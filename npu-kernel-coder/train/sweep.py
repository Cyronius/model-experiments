"""Build dev-split generators at many shapes, to grow the SFT corpus (compile only, no NPU).

    dibs run build --label ... -- python3 sweep.py [--out ~/.cache/model-experiments/nkc-train/builds] [--jobs 8] [--only ln,gemv_q4]
    python3 -m npuasm corpus <out> --out data/sweep_corpus.jsonl --jobs 12      (from ../npuasm)

Each spec is an IRON design under openflowlm-next/open_kernels, built by its own
`build_design.py` with the shape in environment variables, using the mlir-aie 1.4.2 + Peano
toolchain in phlegm's npu-dev env. Grids use real model widths (hidden sizes, FFN sizes, head
dims), since the use case is porting kernels to a new model's shapes. Shapes a design rejects
(its asserts, program memory) just fail and are logged.

Only dev-split generators (eval/generators.py) are swept: test generators (gemma4 text
kernels, gemma4 multimodal ops, lm_head) must never reach training. Builds go to
<out>/<design>/<name>/ with names like `build_n2048_k1024`, so the eval's name templates group
them. Existing builds are skipped, so a rerun resumes. sweep_log.jsonl records every attempt.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))
import generators  # noqa: E402

OK_ROOT = Path.home() / "code/openflowlm-next/open_kernels"
PY = Path.home() / "code/phlegm/target/npu-dev/env/bin/python"

HIDDEN = (768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120)
FFN = (1024, 2048, 3072, 4096, 5120, 6144, 8192, 9728, 11008, 14336)


def grid(**axes):
    keys = list(axes)
    for vals in itertools.product(*(axes[k] for k in keys)):
        yield dict(zip(keys, vals))


# design dir -> (script, [(name, env)])
def specs() -> dict[str, tuple[str, list[tuple[str, dict]]]]:
    s = {}
    s["ln"] = ("ln.py", [(f"build_n{g['LN_N']}_g{g['LN_GROUPS']}", {**g, "LN_EPS": "1e-6"})
                         for g in grid(LN_N=HIDDEN + (1152, 2304, 6144), LN_GROUPS=(1, 2, 4))])
    s["gemv_q4"] = ("gemv_q4.py", [(f"build_n{g['GEMV_N']}_k{g['GEMV_K']}", g)
                                   for g in grid(GEMV_N=HIDDEN[1:] + FFN[3:], GEMV_K=(1024, 1536, 2048, 2560, 3072, 4096))])
    s["gemm_q4"] = ("gemm_q4.py", [(f"build_m{g['GEMM_M']}_n{g['GEMM_N']}_k{g['GEMM_K']}", g)
                                   for g in grid(GEMM_M=(4, 8), GEMM_N=(2048, 4096, 8192, 11008), GEMM_K=(1024, 2048, 3072, 4096))])
    s["gemm_q4_prefill"] = ("gemm_q4_prefill.py", [(f"build_t{g['GQP_T']}_n{g['GQP_N']}_k{g['GQP_K']}", g)
                                                   for g in grid(GQP_T=(256, 512), GQP_N=(1024, 2048, 2560, 4096), GQP_K=(1024, 2048, 2560, 4096))])
    s["dit_gemm"] = ("dit_gemm.py", [(f"build_m{g['DG_M']}_k{g['DG_K']}_n{g['DG_N']}", g)
                                     for g in grid(DG_M=(1024, 2048, 4096), DG_K=(1536, 2048, 3072), DG_N=(3072, 6144, 9216))])
    s["whisper_gemm"] = ("whisper_gemm.py", [(f"build_m{g['WG_M']}_k{g['WG_K']}_n{g['WG_N']}", g)
                                             for g in grid(WG_M=(512, 1024, 1536), WG_K=(384, 512, 768, 1024, 1280), WG_N=(1280, 1536, 3840, 5120))])
    s["attn_block"] = ("attn_gemm.py", [(f"build_m{g['AG_M']}_k{g['AG_K']}_n{g['AG_N']}", g)
                                        for g in grid(AG_M=(512, 1024, 2048), AG_K=(64, 128, 256), AG_N=(512, 1024, 2048, 4096))])
    s["moe_batch"] = ("moe_batch.py", [(f"build_h{g['MB_HID']}_t{g['MB_NT']}", g)
                                       for g in grid(MB_HID=(1024, 1536, 2048, 2560, 3072, 4096), MB_NT=(8, 16))])
    for d, (script, items) in s.items():
        gen = generators.generator(f"openflowlm-next/open_kernels/designs/{d}", "build_#")
        assert gen not in generators.TEST, (d, gen)
    return s


def build_one(out: Path, design: str, script: str, name: str, env: dict) -> dict:
    dest = out / design / name
    rec = {"design": design, "name": name, "env": env, "out": str(dest)}
    if (dest / "final.prj").is_dir() and (dest / "insts.bin").exists():
        return {**rec, "status": "exists"}
    t = time.time()
    p = subprocess.run([str(PY), "build_design.py", f"designs/{design}/{script}", str(dest)], cwd=OK_ROOT,
                       env={**os.environ, **{k: str(v) for k, v in env.items()}},
                       capture_output=True, text=True, timeout=1800)
    rec.update(seconds=round(time.time() - t, 1), status="ok" if p.returncode == 0 and "BUILD_OK" in p.stdout else "failed")
    if rec["status"] != "ok":
        tail = (p.stderr or p.stdout).strip().splitlines()[-6:]
        rec["error"] = "\n".join(tail)[-800:]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path.home() / ".cache/model-experiments/nkc-train/builds")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    sp = specs()
    if a.only:
        sp = {k: v for k, v in sp.items() if k in a.only.split(",")}
    rev = subprocess.run(["git", "-C", str(OK_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(OK_ROOT), "status", "--porcelain", "open_kernels/designs"],
                                capture_output=True, text=True).stdout.strip())
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(d, script, n, e) for d, (script, items) in sp.items() for n, e in items]
    print(f"{len(jobs)} builds, openflowlm-next {rev[:10]}{' (designs dirty)' if dirty else ''}", flush=True)
    log = (a.out / "sweep_log.jsonl").open("a")
    counts: dict = {}
    with ThreadPoolExecutor(a.jobs) as pool:
        for rec in pool.map(lambda j: build_one(a.out, *j), jobs):
            rec.update(openflowlm_rev=rev, designs_dirty=dirty)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            key = (rec["design"], rec["status"])
            counts[key] = counts.get(key, 0) + 1
    for (d, st), n in sorted(counts.items()):
        print(f"{d:18s} {st:8s} {n}")


if __name__ == "__main__":
    main()
