"""Validate native-width ternary products exactly, then time warm ABBA blocks.

Run under `dibs run npu -- ...`. This measures an arithmetic primitive on one
core with local operands, not a two-layer embedding engine or replica topology.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parent
DRIVER = Path("C:/code/openflowlm-next/open_kernels/harness/out/run_kernel.exe")


def fixture(k, repeats, directory):
    rng = np.random.default_rng(k)
    a = rng.integers(-16, 17, size=(4, k), dtype=np.int8)
    w = rng.integers(-1, 2, size=(32, k), dtype=np.int8)
    # Hardware A is [K/16, 4 rows, 16]; each B byte holds four 2-bit codes.
    packed_a = a.reshape(4, k // 16, 16).transpose(1, 0, 2).copy()
    codes = (w.astype(np.int16) & 3).astype(np.uint8)
    kb, i = np.arange(k // 16)[:, None], np.arange(128)[None, :]
    ki, ni = kb * 16 + i // 8, 2 * (i % 8)
    packed_w = codes[ni, ki] | (codes[ni + 16, ki] << 2)
    packed_w |= (codes[ni + 1, ki] << 4) | (codes[ni + 17, ki] << 6)
    ref = (a.astype(np.int64) @ w.astype(np.int64).T) * (4 * repeats)
    assert np.abs(ref).max() <= np.iinfo(np.int32).max
    expected = np.concatenate([ref[:, :16].reshape(-1), ref[:, 16:].reshape(-1)]).astype(np.int32)
    packed_a.tofile(directory / f"a{k}.bin")
    packed_w.tofile(directory / f"w{k}.bin")
    np.save(directory / f"reference{k}.npy", expected)
    return expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=41)
    args = ap.parse_args()
    result_dir = ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    ks = (384, 512, 1536, 2048)
    expected, lines = {}, ["device"]
    for k in ks:
        build = ROOT / f"build_k{k}_r{args.repeats}"
        for name in ("final.xclbin", "insts.bin", "build.json"):
            assert (build / name).is_file(), build / name
        expected[k] = fixture(k, args.repeats, result_dir)
        lines += [f"xclbin g{k} {build.as_posix()}/final.xclbin",
                  f"kernelx k{k} g{k} {build.as_posix()}/insts.bin",
                  f"buf a{k} {4*k} a{k}.bin", f"buf w{k} {8*k} w{k}.bin",
                  f"buf y{k} 512"]
    # ABBA for each width pair, repeated in reverse order for a second pair.
    order = (384, 512, 512, 384, 1536, 2048, 2048, 1536,
             512, 384, 384, 512, 2048, 1536, 1536, 2048)
    for k in order:
        lines += [f"run k{k} a{k} w{k} y{k}"] * (args.samples + 4)
        lines += [f"dump y{k} y{k}.bin 512"]
    config = result_dir / "bench.cfg"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    env = dict(os.environ)
    env["PATH"] = "C:/dev/xrtNPUfromDLL;C:/Windows/System32/AMD;" + env["PATH"]
    env["HARNESS_TIMEOUT_MS"] = "10000"
    run = subprocess.run([str(DRIVER), str(config)], env=env, capture_output=True,
                         text=True, errors="replace", timeout=180)
    (result_dir / "driver.log").write_text(run.stdout + run.stderr, encoding="utf-8")
    if run.returncode:
        raise RuntimeError(f"NPU driver exited {run.returncode}: {(run.stdout + run.stderr)[-2500:]}")
    checks = {}
    for k in ks:
        got = np.fromfile(result_dir / f"y{k}.bin", dtype=np.int32)
        checks[k] = bool(np.array_equal(got, expected[k]))
        if not checks[k]:
            raise RuntimeError(f"K={k}: exact product mismatch; max error "
                               f"{np.max(np.abs(got.astype(np.int64)-expected[k]))}")
    # Match the harness's output; do not silently accept absent timings.
    rows = re.findall(r"run (k\d+).*?state (\d+).*?\(([0-9.]+) ms\)", run.stdout)
    if len(rows) != len(order) * (args.samples + 4):
        raise RuntimeError(f"Unexpected timing log: {len(rows)} runs; inspect driver.log")
    timing = {k: [] for k in ks}
    blocks = []
    stride = args.samples + 4
    for block, k in enumerate(order):
        selected = rows[block*stride+4:(block+1)*stride]
        assert all(name == f"k{k}" and state == "4" for name, state, ms in selected)
        vals = [float(ms) for name, state, ms in selected]
        timing[k] += vals
        blocks.append({"k": k, "median_ms": float(np.median(vals))})
    stats = {k: {"median_ms": float(np.median(v)), "p10_ms": float(np.percentile(v,10)),
                 "p90_ms": float(np.percentile(v,90)), "n": len(v)} for k,v in timing.items()}
    report = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "One-core, resident-operand 4xKx32 ternary primitive; not full-model throughput",
              "repeats": args.repeats, "exact_validation": checks,
              "statistics": stats, "blocks": blocks,
              "ratio_512_over_384": stats[512]["median_ms"] / stats[384]["median_ms"],
              "ratio_2048_over_1536": stats[2048]["median_ms"] / stats[1536]["median_ms"]}
    (ROOT / "width-result.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
