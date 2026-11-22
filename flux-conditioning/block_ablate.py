"""Klein denoiser block-removal screen: velocity error per removed block and step, teacher-forced (2026-11-18)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

import image_check


def run_from(tr, inp: dict, skip: set[int], start: int = 0, state=None, record: list | None = None) -> torch.Tensor:
    """The diffusers Flux2 forward (no KV cache, no references), from block `start`, with blocks in `skip` as identity."""
    hs, enc = inp["hidden_states"], inp["encoder_hidden_states"]
    ntxt = enc.shape[1]
    temb = tr.time_guidance_embed(inp["timestep"].to(hs.dtype) * 1000, None)
    mod_img = tr.double_stream_modulation_img(temb)
    mod_txt = tr.double_stream_modulation_txt(temb)
    mod_sgl = tr.single_stream_modulation(temb)
    img_rope, txt_rope = tr.pos_embed(inp["img_ids"][0]), tr.pos_embed(inp["txt_ids"][0])
    rope = (torch.cat([txt_rope[0], img_rope[0]]), torch.cat([txt_rope[1], img_rope[1]]))
    if state is None:
        state = (tr.context_embedder(enc), tr.x_embedder(hs))
    n_dbl = len(tr.transformer_blocks)
    blocks = list(tr.transformer_blocks) + list(tr.single_transformer_blocks)
    for i in range(start, len(blocks)):
        if i == n_dbl and isinstance(state, tuple):
            state = torch.cat(state, dim=1)
        if record is not None:
            record.append(state)
        if i in skip:
            continue
        if i < n_dbl:
            state = blocks[i](hidden_states=state[1], encoder_hidden_states=state[0], temb_mod_img=mod_img,
                              temb_mod_txt=mod_txt, image_rotary_emb=rope, joint_attention_kwargs=None)
        else:
            state = blocks[i](hidden_states=state, encoder_hidden_states=None, temb_mod=mod_sgl,
                              image_rotary_emb=rope, joint_attention_kwargs=None)
    if isinstance(state, tuple):
        state = torch.cat(state, dim=1)
    return tr.proj_out(tr.norm_out(state[:, ntxt:], temb))


def rel_err(v: torch.Tensor, ref: torch.Tensor) -> float:
    v, ref = v.float(), ref.float()
    return float((v - ref).norm() / ref.norm())


def teacher_inputs(pipe, cond: torch.Tensor, noise: torch.Tensor) -> tuple[list[dict], object]:
    tr, calls = pipe.transformer, []
    orig = tr.forward

    def spy(**kw):
        calls.append({k: kw[k] for k in ("hidden_states", "encoder_hidden_states", "timestep", "img_ids", "txt_ids")})
        return orig(**kw)

    tr.forward = spy
    try:
        img = pipe(prompt_embeds=cond[None], latents=noise.clone(), height=512, width=512,
                   num_inference_steps=4, guidance_scale=1.0).images[0]
    finally:
        del tr.forward
    return calls, img


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path, required=True, help="encoder-probe run folder (cond/, targets/)")
    p.add_argument("--noise", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", type=int, default=128, help="meanpad text rows")
    p.add_argument("--skip", type=int, nargs="*", default=[], help="blocks already removed (greedy rounds)")
    p.add_argument("--cases", type=int, default=0, help="first N cases only (0 = all)")
    p.add_argument("--threads", type=int, default=12)
    args = p.parse_args()
    pipe = image_check.load_pipe(args.threads)
    tr = pipe.transformer
    n_blocks = len(tr.transformer_blocks) + len(tr.single_transformer_blocks)
    base = set(args.skip)
    noise = load_file(str(args.noise))["latents"]
    cases = json.loads((args.probe / "targets" / "manifest.json").read_text(encoding="utf-8"))["cases"]
    cases = cases[:args.cases] if args.cases else cases
    args.output.mkdir(parents=True, exist_ok=True)
    tag = "base" + ("-" + "-".join(map(str, sorted(base))) if base else "")
    out_json = args.output / f"screen-{tag}.json"
    rows = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else []
    done = {r["case"] for r in rows}
    for case in cases:
        if case["id"] in done:
            continue
        real = int(load_file(str(args.probe / "targets" / case["file"]))["attention_mask"].sum())
        meanpad = load_file(str(args.probe / "cond" / f"{case['id']}-b-meanpad.safetensors"))["projected_conditioning"]
        t0 = time.perf_counter()
        with torch.inference_mode():
            calls, img = teacher_inputs(pipe, meanpad[:max(args.rows, real)], noise)
            if not base:
                img.save(args.output / f"{case['id']}-teacher{args.rows}.png")
            for s, inp in enumerate(calls):
                states: list = []
                ref = run_from(tr, inp, base, record=states)
                if s == 0 and not base:
                    stock = tr(**inp, guidance=None, joint_attention_kwargs=None, return_dict=False)[0]
                    assert rel_err(ref, stock) < 1e-3, f"replica forward differs: {rel_err(ref, stock)}"
                for k in range(n_blocks):
                    if k in base:
                        continue
                    v = run_from(tr, inp, base | {k}, start=k, state=states[k])
                    rows.append({"case": case["id"], "step": s, "block": k, "rel_err": rel_err(v, ref)})
        out_json.write_text(json.dumps(rows), encoding="utf-8")
        print(f"{case['id']} done {time.perf_counter() - t0:.0f}s", flush=True)
    blocks = sorted({r["block"] for r in rows})
    grid = np.array([[np.mean([r["rel_err"] for r in rows if r["block"] == k and r["step"] == s]) for s in range(4)]
                     for k in blocks])
    print("block  step0  step1  step2  step3   mean")
    for k, g in zip(blocks, grid):
        print(f"{k:5d} " + " ".join(f"{x:6.3f}" for x in g) + f" {g.mean():6.3f}")
    order = [blocks[i] for i in np.argsort(grid.mean(axis=1))]
    print("lowest mean error first:", order)


if __name__ == "__main__":
    main()
