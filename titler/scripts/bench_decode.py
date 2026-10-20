"""Per-title cost of each decode mode in llama.cpp: one shared prompt pass, then N sequences per step (2026-10-19).

    python scripts/bench_decode.py --gguf model/v3-title-8L.q4_k_m.gguf --threads 2,4
"""

import argparse
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

TOOL = str(Path(".llama-build/bin/Release/llama-batched-bench.exe").resolve())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--threads", default="2,4")
    ap.add_argument("--npp", default="42,258", help="prompt tokens: eval median and p95 (256 cap + BOS + SEP)")
    ap.add_argument("--ntg", default="16,23", help="title tokens with EOS: eval median and p95")
    ap.add_argument("--npl", default="1,2,4,6,8", help="parallel sequences: 1 = greedy, else beams or samples")
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()

    for t in args.threads.split(","):
        cells = defaultdict(list)
        for _ in range(args.reps):
            out = subprocess.run([TOOL, "-m", args.gguf, "-c", "4096", "-b", "2048", "-ub", "512", "-pps", "-t", t,
                                  "-npp", args.npp, "-ntg", args.ntg, "-npl", args.npl, "--output-format", "jsonl"],
                                 capture_output=True, text=True)
            for line in out.stdout.splitlines():
                if line.startswith("{"):
                    r = json.loads(line)
                    cells[(r["pp"], r["tg"], r["pl"])].append(r["t"] * 1000)
        print(f"threads={t}  per-title ms, median of {args.reps}")
        print("  prompt  title  N=" + "  ".join(f"{n:>6}" for n in args.npl.split(",")))
        for pp in args.npp.split(","):
            for tg in args.ntg.split(","):
                ms = [statistics.median(cells[(int(pp), int(tg), int(n))]) for n in args.npl.split(",")]
                print(f"  {pp:>6} {tg:>6}    " + "  ".join(f"{m:6.1f}" for m in ms))


if __name__ == "__main__":
    main()
