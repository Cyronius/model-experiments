"""Linear bridges for removed Klein one-stream blocks: ridge from the modulated input to the gap's residual change, per step (2026-11-18)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

import image_check
from block_ablate import run_from


def gaps_arg(text: str) -> tuple[int, int]:
    a, b = text.split("-")
    return int(a), int(b)


def sgl_mod(tr, inp: dict):
    from diffusers.models.transformers.transformer_flux2 import Flux2Modulation
    temb = tr.time_guidance_embed(inp["timestep"].to(inp["hidden_states"].dtype) * 1000, None)
    return Flux2Modulation.split(tr.single_stream_modulation(temb), 1)[0]


def collect(args) -> None:
    pipe = image_check.load_pipe(args.threads)
    tr = pipe.transformer
    norm = tr.single_transformer_blocks[0].norm
    cases = [json.loads(l) for l in args.cases.read_text(encoding="utf-8").splitlines() if l.strip()]
    d = tr.inner_dim if hasattr(tr, "inner_dim") else 3072
    stats = {(g, s, split): {"G": torch.zeros(d, d, dtype=torch.float64), "C": torch.zeros(d, d, dtype=torch.float64),
                             "yy": 0.0, "n": 0}
             for g in args.gaps for s in range(4) for split in ("train", "validation")}
    gates = {}
    files = {c["id"]: c["file"] for c in json.loads((args.targets / "manifest.json").read_text(encoding="utf-8"))["cases"]}
    for case in cases:
        split = case["split"]
        cond = load_file(str(args.targets / files[case["id"]]))["projected_conditioning"].to(torch.bfloat16)
        step = [0]
        orig = tr.forward

        def spy(**kw):
            inp = {k: kw[k] for k in ("hidden_states", "encoder_hidden_states", "timestep", "img_ids", "txt_ids")}
            states: list = []
            out = run_from(tr, inp, set(), record=states)
            shift, scale, gate = sgl_mod(tr, inp)
            gates[step[0]] = gate.float().reshape(-1)
            for a, b in args.gaps:
                x0 = states[a]
                zn = ((1 + scale) * norm(x0) + shift)[0].double()
                y = (states[b + 1] - x0)[0].double()
                st = stats[((a, b), step[0], split)]
                st["G"] += zn.T @ zn
                st["C"] += zn.T @ y
                st["yy"] += float((y * y).sum())
                st["n"] += zn.shape[0]
            step[0] += 1
            return (out,)

        tr.forward = spy
        try:
            with torch.no_grad():
                pipe(prompt_embeds=cond[None], height=512, width=512, num_inference_steps=4, guidance_scale=1.0,
                     generator=torch.Generator().manual_seed(case["seed"]), output_type="latent")
        finally:
            del tr.forward
        print(f"{case['id']} {split} rows {len(cond)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"stats": stats, "gates": gates, "gaps": args.gaps}, args.output / "stats.pt")


def solve(args) -> None:
    blob = torch.load(args.output / "stats.pt", weights_only=False)
    stats, gates = blob["stats"], blob["gates"]
    out, report = {}, []
    for a, b in blob["gaps"]:
        for s in range(4):
            tr_, va = stats[((a, b), s, "train")], stats[((a, b), s, "validation")]
            G, C = tr_["G"], tr_["C"]
            scale = float(torch.diagonal(G).mean())
            best = None
            for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
                W = torch.linalg.solve(G + lam * scale * torch.eye(G.shape[0], dtype=G.dtype), C)
                sse = va["yy"] - 2 * float((W * va["C"]).sum()) + float((W * (va["G"] @ W)).sum())
                rel = (max(sse, 0.0) / va["yy"]) ** 0.5
                if best is None or rel < best[0]:
                    best = (rel, lam, W)
            rel, lam, W = best
            out[f"{a}-{b}_s{s}"] = W.float().numpy()
            report.append({"gap": f"{a}-{b}", "step": s, "lambda": lam, "val_rel_err": round(rel, 4),
                           "min_abs_gate": round(float(gates[s].abs().min()), 5)})
            print(report[-1], flush=True)
    np.savez(args.output / "bridges.npz", **out, **{f"gate_s{s}": g.numpy() for s, g in gates.items()})
    (args.output / "solve.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--cases", type=Path, required=True)
    c.add_argument("--targets", type=Path, required=True)
    c.add_argument("--gaps", type=gaps_arg, nargs="+", required=True, help="removed block ranges, e.g. 17-21")
    c.add_argument("--threads", type=int, default=12)
    c.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("solve")
    s.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    {"collect": collect, "solve": solve}[args.cmd](args)


if __name__ == "__main__":
    main()
