"""Time Klein on the NPU: stock 25 blocks against a healed GGUF with blocks skipped and bridges; run under dibs --measure (2026-11-20)."""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path

from npu_prune_render import ask, worker

PROMPTS = ["three red apples on a wooden table", "portrait of albert einstein, oil painting", "stylized exterior architecture"]


def run(gguf, skip, bridges, reps: int) -> list[dict]:
    p = worker(gguf)
    out = Path(tempfile.mkdtemp(prefix="npu-time-"))
    rows = []
    try:
        for i in range(reps + 1):
            r = ask(p, {"prompt": PROMPTS[i % len(PROMPTS)], "size": 512, "steps": 4, "seed": 1, "skip_blocks": skip,
                        "out": str(out / f"{i}"), "bridges": bridges})
            if i:
                rows.append({k: v for k, v in r.items() if k.endswith("_s")})
    finally:
        p.stdin.close()
        p.wait(timeout=60)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--skip", type=int, nargs="+", required=True)
    ap.add_argument("--bridges", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=6)
    a = ap.parse_args()
    res = {"stock": run(None, [], None, a.reps),
           "healed": run(a.gguf.resolve(), sorted(a.skip), str(a.bridges.resolve()), a.reps)}
    for k, rows in res.items():
        t = [r["npu_s"] for r in rows]
        print(f"{k}: npu_s median {statistics.median(t):.3f} (min {min(t):.3f}, max {max(t):.3f}, n {len(t)}); "
              f"other keys {sorted(rows[0])}")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
