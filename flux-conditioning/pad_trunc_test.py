"""Klein denoiser on short text sequences: drop or shorten the padding rows and compare images (2026-11-17)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

import image_check


def variants(probe_dir: Path, case_id: str, real: int, keep: list[int]) -> dict[str, torch.Tensor]:
    cond = probe_dir / "cond"
    teacher = load_file(str(cond / f"{case_id}-a-teacher.safetensors"))["projected_conditioning"]
    meanpad = load_file(str(cond / f"{case_id}-b-meanpad.safetensors"))["projected_conditioning"]
    probe = load_file(str(cond / f"{case_id}-c-probe.safetensors"))["projected_conditioning"]
    out = {"a-teacher": teacher, "d-trunc": teacher[:real], "f-probe-trunc": probe[:real]}
    out.update({f"e-pad{k:03d}": meanpad[:max(k, real)] for k in keep})
    return out


def mad(a: Path, b: Path) -> float:
    x = np.asarray(Image.open(a).convert("RGB"), dtype=np.float32)
    y = np.asarray(Image.open(b).convert("RGB"), dtype=np.float32)
    return float(np.abs(x - y).mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path, required=True, help="encoder-probe run folder (cond/, targets/, png/)")
    p.add_argument("--noise", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--keep", type=int, nargs="+", default=[128])
    p.add_argument("--threads", type=int, default=12)
    args = p.parse_args()
    pipe = image_check.load_pipe(args.threads)
    noises = {n.stem: load_file(str(n))["latents"] for n in args.noise}
    out = args.output / "png"
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads((args.probe / "targets" / "manifest.json").read_text(encoding="utf-8"))["cases"]
    rows = []
    for case in cases:
        real = int(load_file(str(args.probe / "targets" / case["file"]))["attention_mask"].sum())
        for tag, c in variants(args.probe, case["id"], real, args.keep).items():
            for nname, lat in noises.items():
                png = out / f"{case['id']}-{tag}__{nname}.png"
                if not png.exists():
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        img = pipe(prompt_embeds=c[None], latents=lat.clone(), height=512, width=512,
                                   num_inference_steps=4, guidance_scale=1.0).images[0]
                    img.save(png)
                    print(f"{png.name} rows {len(c)} {time.perf_counter() - t0:.1f}s", flush=True)
                rows.append({"case": case["id"], "prompt": case["prompt"], "variant": tag, "noise": nname,
                             "text_rows": len(c), "png": png.name})
    teacher = {(r["case"], r["noise"]): out / r["png"] for r in rows if r["variant"] == "a-teacher"}
    meanpad = {c["id"]: args.probe / "png" / f"{c['id']}-b-meanpad__noise-a.png" for c in cases}
    for r in rows:
        r["mad_vs_teacher"] = mad(out / r["png"], teacher[(r["case"], r["noise"])])
        if r["noise"] == "noise-a" and meanpad[r["case"]].exists():
            r["mad_vs_meanpad512"] = mad(out / r["png"], meanpad[r["case"]])
    names = sorted(noises)
    for case in cases:
        if meanpad[case["id"]].exists():
            rows.append({"case": case["id"], "variant": "b-meanpad512", "noise": "noise-a", "text_rows": 512,
                         "mad_vs_teacher": mad(meanpad[case["id"]], teacher[(case["id"], "noise-a")])})
        if len(names) > 1:
            rows.append({"case": case["id"], "variant": "seed-change", "noise": f"{names[0]} vs {names[1]}",
                         "mad_vs_teacher": mad(teacher[(case["id"], names[0])], teacher[(case["id"], names[1])])})
    (args.output / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for tag in sorted({r["variant"] for r in rows}):
        v = [r["mad_vs_teacher"] for r in rows if r["variant"] == tag]
        print(f"{tag}: mean pixel MAD vs teacher {np.mean(v):.1f} (n {len(v)})")


if __name__ == "__main__":
    main()
