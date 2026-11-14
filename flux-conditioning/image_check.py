"""Render Klein 4B images from projected [512, 3072] conditioning on the CPU (2026-11-10). Needs diffusers."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent.parent / "artifacts" / "flux-conditioning"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_noise(args) -> None:
    gen = torch.Generator().manual_seed(args.seed)
    side = args.size // 16
    save_file({"latents": torch.randn(1, 128, side, side, generator=gen)}, str(args.output),
              metadata={"seed": str(args.seed), "size": str(args.size)})
    print(args.output, sha256(args.output))


def cond(path: Path) -> torch.Tensor:
    t = load_file(str(path))
    c = t["projected_conditioning"]
    if tuple(c.shape) != (512, 3072) or not torch.isfinite(c).all():
        raise ValueError(f"Bad conditioning in {path}")
    return c


def pad_rows(path: Path) -> tuple[torch.Tensor, int]:
    t = load_file(str(path))
    return t["projected_conditioning"], int(t["attention_mask"].sum())


def stage1_variants(args) -> None:
    base, nb = pad_rows(args.base)
    enr, ne = pad_rows(args.enriched)
    other, no = pad_rows(args.other)
    pool = [pad_rows(p) for p in sorted(args.pool.glob("*.safetensors")) if not p.stem.startswith("boundary")]
    lo = max(n for _, n in pool)
    glob_mean = torch.stack([c[lo:].float() for c, _ in pool]).mean((0, 1)).to(base.dtype)

    def with_pad(c: torch.Tensor, n: int, pad: torch.Tensor) -> torch.Tensor:
        out = c.clone()
        out[n:] = pad if pad.dim() == 1 else pad[n:]
        return out

    variants = {
        "1-base": base,
        "2-base-padmean": with_pad(base, nb, base[nb:].float().mean(0).to(base.dtype)),
        "3-base-otherpad": with_pad(base, max(nb, no), other),
        "4-base-globalmean": with_pad(base, nb, glob_mean),
        "5-enriched": enr,
        "6-enriched-basepad": with_pad(enr, ne, base),
        "7-enriched-padmean": with_pad(enr, ne, enr[ne:].float().mean(0).to(enr.dtype)),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for name, c in variants.items():
        save_file({"projected_conditioning": c.contiguous()}, str(args.output / f"{name}.safetensors"))
    (args.output / "variants.json").write_text(json.dumps({
        "base": str(args.base), "enriched": str(args.enriched), "other": str(args.other),
        "real_rows": {"base": nb, "enriched": ne, "other": no}, "pool_first_pad_row": lo,
        "files": {p.name: sha256(p) for p in sorted(args.output.glob("*.safetensors"))}}, indent=2))
    print(f"wrote {len(variants)} variants; base rows {nb}, enriched rows {ne}, other rows {no}")


def load_pipe(steps_threads: int):
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel

    torch.set_num_threads(steps_threads)
    transformer = Flux2Transformer2DModel.from_pretrained(ROOT / "teacher" / "transformer", torch_dtype=torch.bfloat16)
    # Inputs are already projected; context_embedder is bias-free, so identity is the exact bypass.
    assert transformer.context_embedder.bias is None
    transformer.context_embedder = torch.nn.Identity()
    vae = AutoencoderKLFlux2.from_pretrained(ROOT / "image-pipeline" / "vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ROOT / "image-pipeline" / "scheduler")
    pipe = Flux2KleinPipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None,
                              transformer=transformer, is_distilled=True)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def render(args) -> None:
    pipe = load_pipe(args.threads)
    noises = {p.stem: load_file(str(p))["latents"] for p in args.noise}
    args.output.mkdir(parents=True, exist_ok=True)
    log = []
    files = sorted(f for d in args.cond for f in (d.glob("*.safetensors") if d.is_dir() else [d]))
    for f in files:
        c = cond(f)[None]
        for nname, lat in noises.items():
            out = args.output / f"{f.stem}__{nname}.png"
            if out.exists():
                continue
            side = lat.shape[-1] * 16
            t0 = time.perf_counter()
            with torch.inference_mode():
                img = pipe(prompt_embeds=c, latents=lat.clone(), height=side, width=side,
                           num_inference_steps=args.steps, guidance_scale=1.0).images[0]
            img.save(out)
            dt = time.perf_counter() - t0
            log.append({"cond": str(f), "cond_sha256": sha256(f), "noise": nname, "png": out.name, "seconds": dt})
            print(f"{out.name} {dt:.1f}s", flush=True)
    with (args.output / "render-log.jsonl").open("a", encoding="utf-8") as fh:
        for row in log:
            fh.write(json.dumps(row) + "\n")


def sheet(args) -> None:
    from PIL import Image, ImageDraw

    pngs = sorted(args.input.glob(args.pattern))
    ims = [Image.open(p).convert("RGB") for p in pngs]
    w, h = ims[0].size
    cols = args.cols
    rows = (len(ims) + cols - 1) // cols
    out = Image.new("RGB", (cols * w, rows * (h + 24)), "white")
    d = ImageDraw.Draw(out)
    for i, (p, im) in enumerate(zip(pngs, ims)):
        x, y = (i % cols) * w, (i // cols) * (h + 24)
        out.paste(im, (x, y + 24))
        d.text((x + 4, y + 4), p.stem[: w // 6], fill="black")
    out.save(args.output)
    print(args.output, len(ims))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    n = sub.add_parser("noise")
    n.add_argument("--seed", type=int, required=True)
    n.add_argument("--size", type=int, default=512)
    n.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("stage1")
    for name in ("base", "enriched", "other", "pool", "output"):
        v.add_argument(f"--{name}", type=Path, required=True)
    r = sub.add_parser("render")
    r.add_argument("--cond", type=Path, nargs="+", required=True)
    r.add_argument("--noise", type=Path, nargs="+", required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--steps", type=int, default=4)
    r.add_argument("--threads", type=int, default=12)
    s = sub.add_parser("sheet")
    s.add_argument("--input", type=Path, required=True)
    s.add_argument("--pattern", default="*.png")
    s.add_argument("--cols", type=int, default=4)
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"noise": make_noise, "stage1": stage1_variants, "render": render, "sheet": sheet}[args.command](args)


if __name__ == "__main__":
    main()
