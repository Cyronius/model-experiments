"""Enricher v2 on the NPU vs PyTorch: --prep exports weights and cases (torch env), --npu runs them (IRON env) (2026-11-21)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

A = Path(__file__).resolve().parents[1] / "artifacts" / "flux-conditioning"
OUT = A / "enricher-npu"
KDIR = Path(r"C:\dev\enricher-kernels")


def prep(n_cases: int, steps: int, guidance: float) -> None:
    import torch
    from safetensors.numpy import save_file
    import enricher_v2 as ev
    import enricher_npu_ref as er

    torch.set_grad_enabled(False)
    model = ev.load_model(A / "enricher-v2" / "run2-cfg" / "cfg.pt", "cpu")[0].float()
    OUT.mkdir(parents=True, exist_ok=True)
    save_file({k: v.float().numpy() for k, v in model.state_dict().items()}, str(OUT / "enricher-v2-cfg.safetensors"))
    data = ev.Data(A / "enricher-v2" / "data", "cpu")
    F, trows = er.fold(model), er.time_rows(model, steps)
    groups = [i for i, g in enumerate(data.groups) if g["split"] == "test"][::97][:n_cases]
    rng = np.random.default_rng(0)
    cases = {"basis": data.basis.float().numpy(), "std": data.std.float().numpy(), "mean": data.mean.float().numpy()}
    for j, gi in enumerate(groups):
        ctx, cm, _, _, n_true = data.gather(torch.tensor([data.groups[gi]["items"][0]]))
        nc, n = int(cm.sum()), int(n_true[0])
        ctx = ctx[0, :nc].numpy()
        noise = np.zeros((ev.WIN, 256), np.float32)
        noise[:n] = rng.standard_normal((n, 256)).astype(np.float32)
        ref, ref_len = er.torch_enrich(model, ctx, n, noise, steps, guidance)
        emu = er.npu_enrich(er.Ops(True), F, trows, ctx, n, noise, steps, guidance)
        cases |= {f"{j}_ctx": ctx, f"{j}_n": np.array(n), f"{j}_noise": noise, f"{j}_ref": ref,
                  f"{j}_ref_len": ref_len, f"{j}_emu": emu, f"{j}_group": np.array(gi)}
    np.savez(OUT / "cases.npz", **cases)
    print(f"{len(groups)} cases -> {OUT / 'cases.npz'}")


def cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def npu(steps: int, guidance: float, reps: int, profile: bool) -> None:
    sys.path.insert(0, r"C:\code\phlegm\open-kernels\open_kernels")
    import flux2_enricher as fe

    r = fe.Runner(OUT / "enricher-v2-cfg.safetensors", KDIR, steps, guidance)
    print(f"loaded in {r.load_s:.1f} s; length head {len(r.length_ops)} ops, sampling {len(r.sample_ops)} ops", flush=True)
    c = np.load(OUT / "cases.npz")
    raw = lambda z: (z @ c["basis"]) * c["std"] + c["mean"]  # noqa: E731
    rows, times = [], []
    j = 0
    while f"{j}_ctx" in c:
        ctx, n, noise = c[f"{j}_ctx"], int(c[f"{j}_n"]), c[f"{j}_noise"]
        t0 = time.perf_counter()
        lg = r.length_logits(ctx)
        t1 = time.perf_counter()
        x = r.sample(ctx, n, noise)
        t2 = time.perf_counter()
        for _ in range(reps - 1):
            t1b = time.perf_counter()
            r.length_logits(ctx)
            r.sample(ctx, n, noise)
            times.append(time.perf_counter() - t1b)
        ref, emu = c[f"{j}_ref"], c[f"{j}_emu"]
        row = {"case": j, "group": int(c[f"{j}_group"]), "nc": len(ctx), "n": n,
               "length_s": round(t1 - t0, 3), "sample_s": round(t2 - t1, 3),
               "pca_cos_min_vs_torch": float(cos(x, ref).min()), "pca_cos_min_vs_emu": float(cos(x, emu).min()),
               "emu_pca_cos_min_vs_torch": float(cos(emu, ref).min()),
               "raw_cos_min_vs_torch": float(cos(raw(x), raw(ref)).min()),
               "len_logit_max_abs": float(np.abs(lg - c[f"{j}_ref_len"]).max()),
               "len_argmax_same": bool(lg.argmax() == c[f"{j}_ref_len"].argmax()),
               "finite": bool(np.isfinite(x).all())}
        rows.append(row)
        print(json.dumps(row), flush=True)
        j += 1
    summary = {"cases": rows, "warm_prompt_s": times}
    if profile:
        prof = {}
        r.sample(c["0_ctx"], int(c["0_n"]), c["0_noise"], profile=prof)
        summary["profile_serialized_s"] = dict(sorted(prof.items(), key=lambda kv: -kv[1]))
        print(json.dumps(summary["profile_serialized_s"], indent=1))
    (OUT / "npu.json").write_text(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--npu", action="store_true")
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()
    if a.prep:
        prep(a.cases, a.steps, a.guidance)
    if a.npu:
        npu(a.steps, a.guidance, a.reps, a.profile)


if __name__ == "__main__":
    main()
