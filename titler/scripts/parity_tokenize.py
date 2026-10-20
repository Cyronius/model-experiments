"""Compare llama-tokenize against our Python tokenizer on eval prompts (2026-10-19).

    python scripts/parity_tokenize.py --gguf model/v3-title-8L.f16.gguf --tokenizer model/v3-title-8L/tokenizer.json --n 1000
"""

import argparse
import ast
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

TOOL = str(Path(".llama-build/bin/Release/llama-tokenize.exe").resolve())
BOS = 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    rows = [json.loads(line) for line in Path("data/eval/eval_set.jsonl").open(encoding="utf-8") if line.strip()][: args.n]
    stats, bad = Counter(), []
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "p.txt"
        for r in rows:
            text = r["prompt"][:1500]
            f.write_bytes(text.encode("utf-8"))
            out = subprocess.run([TOOL, "-m", args.gguf, "-f", str(f), "--ids", "--no-escape", "--log-disable"],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
            try:
                got = ast.literal_eval(out.stdout.strip().splitlines()[-1])
            except (ValueError, SyntaxError, IndexError):
                stats["tool failed"] += 1
                continue
            want = [BOS] + tok.encode(text).ids
            while len(want) > 1 and want[-1] == tok.token_to_id("▁"):  # llama.cpp strips trailing whitespace
                want.pop()
            ascii_only = text.isascii()
            kind = "ascii" if ascii_only else "non-ascii"
            if got == want:
                stats[f"{kind} exact"] += 1
            else:
                stats[f"{kind} differ"] += 1
                if len(bad) < 8:
                    first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
                    bad.append((kind, r["id"], len(got), len(want), first))
    for k, v in sorted(stats.items()):
        print(f"{k}: {v}")
    for b in bad:
        print("  differs:", b)


if __name__ == "__main__":
    main()
