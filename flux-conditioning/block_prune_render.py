"""Render Klein images with denoiser blocks removed, no training; pixel MAD against the 128-row teacher (2026-11-18)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

import image_check
from pad_trunc_test import mad


class SkipDouble(torch.nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, **_):
        return encoder_hidden_states, hidden_states


class SkipSingle(torch.nn.Module):
    def forward(self, hidden_states, **_):
        return hidden_states


def set_removed(tr, original: list, removed: set[int]) -> None:
    n_dbl = len(tr.transformer_blocks)
    for i, block in enumerate(original):
        stub = (SkipDouble() if i < n_dbl else SkipSingle()) if i in removed else block
        if i < n_dbl:
            tr.transformer_blocks[i] = stub
        else:
            tr.single_transformer_blocks[i - n_dbl] = stub


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path, required=True, help="encoder-probe run folder (cond/, targets/)")
    p.add_argument("--teacher-png", type=Path, required=True, help="pad-trunc png/ folder with the e-pad128 images")
    p.add_argument("--noise", type=Path, nargs="+", required=True)
    p.add_argument("--order", type=int, nargs="+", required=True, help="blocks in removal order")
    p.add_argument("--counts", type=int, nargs="+", required=True, help="how many of --order to remove per run")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--threads", type=int, default=12)
    args = p.parse_args()
    pipe = image_check.load_pipe(args.threads)
    tr = pipe.transformer
    original = list(tr.transformer_blocks) + list(tr.single_transformer_blocks)
    noises = {n.stem: load_file(str(n))["latents"] for n in args.noise}
    cases = json.loads((args.probe / "targets" / "manifest.json").read_text(encoding="utf-8"))["cases"]
    out = args.output / "png"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in args.counts:
        removed = set(args.order[:n])
        set_removed(tr, original, removed)
        for case in cases:
            real = int(load_file(str(args.probe / "targets" / case["file"]))["attention_mask"].sum())
            cond = load_file(str(args.probe / "cond" / f"{case['id']}-b-meanpad.safetensors"))["projected_conditioning"]
            cond = cond[:max(args.rows, real)]
            for nname, lat in noises.items():
                png = out / f"{case['id']}-drop{n:02d}__{nname}.png"
                if not png.exists():
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        img = pipe(prompt_embeds=cond[None], latents=lat.clone(), height=512, width=512,
                                   num_inference_steps=4, guidance_scale=1.0).images[0]
                    img.save(png)
                    print(f"{png.name} {time.perf_counter() - t0:.1f}s", flush=True)
                teacher = args.teacher_png / f"{case['id']}-e-pad{args.rows:03d}__{nname}.png"
                rows.append({"case": case["id"], "noise": nname, "removed": sorted(removed), "blocks": 25 - n,
                             "png": png.name, "mad_vs_teacher": mad(png, teacher)})
    set_removed(tr, original, set())
    (args.output / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for n in args.counts:
        v = [r["mad_vs_teacher"] for r in rows if r["blocks"] == 25 - n]
        print(f"{25 - n} blocks (removed {sorted(args.order[:n])}): mean pixel MAD vs teacher {np.mean(v):.1f} (n {len(v)})")


if __name__ == "__main__":
    main()
