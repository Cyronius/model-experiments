"""Render Klein images on the NPU with DiT blocks skipped, through the resident worker; pixel MAD against no skip (2026-11-18)."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

CFG = json.loads(Path(r"C:\dev\klein-eval\resident-20261118\nodibs.json").read_text())


def worker(gguf: Path | None = None) -> subprocess.Popen:
    cmd = (f". '{CFG['env_script']}'; & '{CFG['python']}' -m open_kernels.flux2_native_generate --serve "
           f"--model '{CFG['model']}' --kernels '{CFG['kernels']}' --transformer-gguf '{gguf or CFG['transformer_gguf']}' "
           f"--gguf-modulation {CFG['gguf_modulation']}")
    p = subprocess.Popen(["powershell.exe", "-NoProfile", "-Command", cmd], cwd=Path(CFG["repo_root"]) / "open-kernels",
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    hello = json.loads(p.stdout.readline())
    assert hello.get("ok"), hello
    return p


def ask(p: subprocess.Popen, q: dict) -> dict:
    p.stdin.write(json.dumps(q) + "\n")
    p.stdin.flush()
    r = json.loads(p.stdout.readline())
    if not r.get("ok"):
        raise RuntimeError(r)
    return r


def mad(a: Path, b: Path) -> float:
    x = np.asarray(Image.open(a).convert("RGB"), dtype=np.float32)
    return float(np.abs(x - np.asarray(Image.open(b).convert("RGB"), dtype=np.float32)).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, required=True, help="encoder-probe targets/manifest.json")
    ap.add_argument("--order", type=int, nargs="+", required=True, help="blocks in removal order")
    ap.add_argument("--counts", type=int, nargs="+", required=True, help="how many of --order to skip per run (0 = teacher)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bridges", type=Path, help="bridges.npz from bridge_fit.py; gaps in it must be skipped")
    ap.add_argument("--teacher", type=Path, help="folder with drop00 teacher images, if not --output")
    ap.add_argument("--gguf", type=Path, help="DiT GGUF instead of the config's, e.g. a healed one")
    args = ap.parse_args()
    # The worker runs in another folder, so relative paths would resolve there.
    args.output = args.output.resolve()
    args.teacher = args.teacher and args.teacher.resolve()
    args.bridges = args.bridges and args.bridges.resolve()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    args.output.mkdir(parents=True, exist_ok=True)
    p = worker(args.gguf and args.gguf.resolve())
    rows = []
    try:
        for n in sorted(set(args.counts) | {0}):
            skip = sorted(args.order[:n])
            for case in cases:
                for seed in args.seeds:
                    base = args.teacher if (n == 0 and args.teacher) else args.output
                    out = base / f"drop{n:02d}" / f"{case['id']}-s{seed}"
                    png = out / "image.png"
                    if not png.exists():
                        r = ask(p, {"prompt": case["prompt"], "size": 512, "steps": 4, "seed": seed,
                                    "skip_blocks": skip, "out": str(out),
                                    "bridges": str(args.bridges) if (args.bridges and n) else None})
                        print(f"drop{n:02d} {case['id']} s{seed} npu {r['npu_s']:.2f}s load {r['load_s']:.1f}s", flush=True)
                    rows.append({"case": case["id"], "seed": seed, "removed": skip, "blocks": 25 - n, "png": str(png)})
    finally:
        p.stdin.close()
        p.wait(timeout=60)
    teacher = {(r["case"], r["seed"]): Path(r["png"]) for r in rows if not r["removed"]}
    for r in rows:
        r["mad_vs_teacher"] = mad(Path(r["png"]), teacher[(r["case"], r["seed"])])
    (args.output / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for n in sorted(set(args.counts) | {0}):
        v = [r["mad_vs_teacher"] for r in rows if r["blocks"] == 25 - n]
        print(f"{25 - n} blocks (removed {sorted(args.order[:n])}): mean pixel MAD vs teacher {np.mean(v):.1f} (n {len(v)})")


if __name__ == "__main__":
    main()
