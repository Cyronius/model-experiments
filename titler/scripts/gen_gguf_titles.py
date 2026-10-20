"""Greedy titles from a GGUF via llama-completion for the fixed eval set (2026-10-19).

    python scripts/gen_gguf_titles.py --gguf model/v4-ternary.tq2_0.gguf --tokenizer model/v3-title-8L/tokenizer.json --out data/eval/titles_v4_ternary.json
"""

import argparse
import json
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tokenizers import Tokenizer

from common.validation import clean_title

TOOL = str(Path(".llama-build/bin/Release/llama-completion.exe").resolve())
MAX_PROMPT_TOKENS = 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bias", nargs="*", default=[], help="logit bias entries like 123-3 (token-bias)")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    rows = [json.loads(line) for line in Path("data/eval/eval_set.jsonl").open(encoding="utf-8") if line.strip()]
    workdir = Path(tempfile.mkdtemp())

    def one(item):
        n, r = item
        enc = tok.encode(r["prompt"].rstrip())
        text = r["prompt"].rstrip()
        if len(enc.ids) > MAX_PROMPT_TOKENS:  # same head cap as training; offsets are into the normalized text
            text = tok.decode(enc.ids[:MAX_PROMPT_TOKENS]).strip()
        f = workdir / f"p{n}.txt"
        f.write_bytes((text + "<sep>").encode("utf-8"))
        cmd = [TOOL, "-m", args.gguf, "-f", str(f), "-n", "40", "--temp", "0", "--top-k", "1", "-no-cnv",
               "--no-display-prompt", "--no-escape", "--no-warmup", "-t", "1"]
        for b in args.bias:
            cmd += ["--logit-bias", b]
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        raw = re.sub(r"\x1b\[[0-9;]*m", "", out.stdout).replace("[end of text]", "").strip()
        return r["id"], clean_title(raw)

    with ThreadPoolExecutor(args.threads) as pool:
        titles = dict(pool.map(one, enumerate(rows)))
    Path(args.out).write_text(json.dumps(titles, ensure_ascii=False), encoding="utf-8")
    print(f"{len(titles)} titles -> {args.out}")


if __name__ == "__main__":
    main()
