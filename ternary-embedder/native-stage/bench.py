"""Run all trained four-row projection fixtures under `dibs run npu -- ...`."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from arithmetic import dequantize

ROOT = Path(__file__).resolve().parent
DRIVER = Path("C:/code/openflowlm-next/open_kernels/harness/out/run_kernel.exe")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=ROOT / "fixtures")
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    fixture_dir = args.fixtures.resolve()
    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    out = ROOT / "results" / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True)
    lines, builds = ["device"], {}
    for k, n in ((384, 384), (384, 1536), (1536, 384)):
        tag = f"p{k}n{n}"
        build = ROOT / f"build_k{k}_n{n}"
        builds[tag] = json.loads((build / "build.json").read_text())
        for name, digest in builds[tag]["source_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Rebuild changed source: {name}")
        for name, digest in builds[tag]["artifact_sha256"].items():
            if hashlib.sha256((build / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Build artifact hash changed: {build / name}")
        lines += [f"xclbin g{tag} {build.as_posix()}/final.xclbin",
                  f"kernelx {tag} g{tag} {build.as_posix()}/insts.bin"]
    for key, matrix in manifest["matrices"].items():
        path = fixture_dir / f"{key}.w.bin"
        if hashlib.sha256(path.read_bytes()).hexdigest() != matrix["weight_sha256"]:
            raise ValueError(f"Weight hash changed: {key}")
        lines += [f"buf w{key} {matrix['packed_bytes']} {path.as_posix()}"]
    for case in manifest["fixtures"]:
        name, k, n, groups = (case[x] for x in ("name", "k", "n", "groups"))
        for suffix, digest in (("a.bin", case["activation_sha256"]),
                               ("expected.bin", case["expected_sha256"])):
            if hashlib.sha256((fixture_dir / f"{name}.{suffix}").read_bytes()).hexdigest() != digest:
                raise ValueError(f"Fixture hash changed: {name}.{suffix}")
        size = 4 * n * groups * 4
        lines += [f"buf a{name} {4*k} {fixture_dir.as_posix()}/{name}.a.bin", f"buf y{name} {size}"]
        # The first call warms the shape. Every call still reloads streamed weights.
        lines += [f"run p{k}n{n} a{name} w{case['matrix']} y{name}"] * (args.samples + 1)
        lines += [f"dump y{name} {name}.bin {size}"]
    config = out / "bench.cfg"
    config.write_text("\n".join(lines) + "\n")
    env = dict(os.environ)
    env["PATH"] = "C:/dev/xrtNPUfromDLL;C:/Windows/System32/AMD;" + env["PATH"]
    env["HARNESS_TIMEOUT_MS"] = "10000"
    run = subprocess.run([str(DRIVER), str(config)], env=env, capture_output=True, text=True,
                         errors="replace", timeout=180)
    (out / "driver.log").write_text(run.stdout + run.stderr, encoding="utf-8")
    if run.returncode:
        raise RuntimeError(f"Driver failed ({run.returncode}): {(run.stdout + run.stderr)[-3000:]}")
    timings = re.findall(r"run (p\d+n\d+).*?state (\d+).*?\(([0-9.]+) ms\)", run.stdout)
    stride = args.samples + 1
    if len(timings) != len(manifest["fixtures"]) * stride:
        raise RuntimeError("Unexpected run count; inspect driver.log")
    checks = []
    for index, case in enumerate(manifest["fixtures"]):
        name, n, groups = case["name"], case["n"], case["groups"]
        expected = np.fromfile(fixture_dir / f"{name}.expected.bin", dtype=np.int32)
        got = np.fromfile(out / f"{name}.bin", dtype=np.int32)
        if not np.array_equal(got, expected):
            raise AssertionError(f"Exact integer mismatch: {name}; inspect {out}")
        params = np.load(fixture_dir / f"{case['matrix']}.params.npz")
        fixture = np.load(fixture_dir / f"{name}.dequant.npz")
        y = dequantize(got.reshape(n // 32, groups, 2, 4, 16), fixture["activation_scales"],
                       params["scales"], params["bias"])
        if not np.array_equal(y, fixture["expected"]):
            raise AssertionError(f"Dequantization mismatch: {name}")
        rows = timings[index * stride:(index + 1) * stride]
        if not all(tag == f"p{case['k']}n{n}" and state == "4" for tag, state, _ in rows):
            raise AssertionError(f"Incomplete or misordered execution: {name}")
        values = [float(ms) for _, _, ms in rows[1:]]
        checks.append({**case, "integer_exact": True, "dequant_exact": True,
                       "warm_median_ms": float(np.median(values)), "samples_ms": values})
    report = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "Complete streamed four-row dense projections, not a complete encoder stage",
              "checkpoint_sha256": manifest["checkpoint_sha256"], "builds": builds, "checks": checks,
              "driver_log": str(out / "driver.log")}
    (ROOT / "projection-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"exact_cases": len(checks), "result": str(ROOT / "projection-result.json")}), flush=True)


if __name__ == "__main__":
    main()
