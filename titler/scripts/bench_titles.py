"""Per-title latency on real eval prompts, from llama.cpp's own timers (2026-10-19).

Model stays loaded in an app, so the headline is prompt eval + generation; load time is reported separately.

    python scripts/bench_titles.py --gguf model/v3-title-8L.q5_k_m.gguf --tokenizer model/v3-title-8L/tokenizer.json --n 200 --threads 2
"""

import argparse
import json
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from tokenizers import Tokenizer

TOOL = str(Path(".llama-build/bin/Release/llama-completion.exe").resolve())
MAX_PROMPT_TOKENS = 256
TIMER = re.compile(r"(load time|prompt eval time|eval time|total time) =\s+([\d.]+) ms(?: /\s+(\d+))?")


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    rows = [json.loads(line) for line in Path("data/eval/eval_set.jsonl").open(encoding="utf-8") if line.strip()][: args.n]
    f = Path(tempfile.mkdtemp()) / "p.txt"
    compute, load, wall, ptoks, gtoks = [], [], [], [], []
    for r in rows:
        text = r["prompt"].rstrip()
        ids = tok.encode(text).ids
        if len(ids) > MAX_PROMPT_TOKENS:
            text = tok.decode(ids[:MAX_PROMPT_TOKENS]).strip()
        f.write_bytes((text + "<sep>").encode("utf-8"))
        start = time.perf_counter()
        out = subprocess.run([TOOL, "-m", args.gguf, "-f", str(f), "-n", "40", "--temp", "0", "--top-k", "1", "-no-cnv",
                              "--no-display-prompt", "--no-escape", "--no-warmup", "-t", str(args.threads)],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        wall.append((time.perf_counter() - start) * 1000)
        t = {m.group(1): (float(m.group(2)), m.group(3)) for m in TIMER.finditer(out.stderr)}
        if "total time" not in t:
            continue
        compute.append(t["total time"][0])
        load.append(t["load time"][0])
        ptoks.append(int(t["prompt eval time"][1] or 0))
        gtoks.append(int(t["eval time"][1] or 0))
    n = len(compute)
    print(f"{Path(args.gguf).name}  threads={args.threads}  titles={n}")
    print(f"  per-title compute (model loaded): p50 {pct(compute, .5):.1f} ms  p95 {pct(compute, .95):.1f}  p99 {pct(compute, .99):.1f}  max {max(compute):.1f}")
    print(f"  model load: median {statistics.median(load):.1f} ms   process wall (start+load+title): median {statistics.median(wall):.0f} ms")
    print(f"  prompt tokens: median {statistics.median(ptoks):.0f}, max {max(ptoks)}   title tokens: median {statistics.median(gtoks):.0f}, max {max(gtoks)}")


if __name__ == "__main__":
    main()
