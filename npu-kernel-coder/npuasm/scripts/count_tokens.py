"""Count tokens in the lifted text form, one build per design family.

Needs the `tokenizers` package, e.g.:
  ../../target/manifold-audit/env/bin/python scripts/count_tokens.py \
      results/corpus.jsonl ../../models/manifold-audit/Qwen3.5-9B/tokenizer.json \
      --out results/tokens.md

A family is the directory above `build_*` (e.g. designs/layer_x). Within a
family the build with the most core code is used, so this is a high estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizers import Tokenizer  # noqa: E402

from npuasm import build  # noqa: E402


def family(prj: str) -> str:
    parts = Path(prj).parts
    for i, p in enumerate(parts):
        if p.startswith("build"):
            return "/".join(parts[max(0, i - 2) : i])
    return str(Path(prj).parent.parent)


def kind(name: str) -> str:
    if name.endswith(".s"):
        return "core asm"
    if name == "insts.txn":
        return "insts.txn"
    return "cdo"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("tokenizer", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(a.tokenizer))
    best: dict[str, dict] = {}
    for line in a.results.open():
        r = json.loads(line)
        if r["error"] or not all(r["ok"].values()):
            continue
        f = family(r["prj"])
        if f not in best or r["sizes"].get("core_code", 0) > best[f]["sizes"].get("core_code", 0):
            best[f] = r
    rows = []
    for f, r in sorted(best.items()):
        _, texts, _ = build.lift(Path(r["prj"]), None)
        t = defaultdict(int)
        for name, text in texts.items():
            t[kind(name)] += len(tok.encode(text).ids)
        rows.append((f, r["cores"], r["distinct_cores"], r["sizes"].get("core_code", 0), t))
    lines = [
        "| family | cores | distinct | code bytes | core asm tokens | cdo tokens | insts.txn tokens | total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f, n, d, cb, t in rows:
        total = sum(t.values())
        lines.append(f"| {f} | {n} | {d} | {cb} | {t['core asm']} | {t['cdo']} | {t['insts.txn']} | {total} |")
    s = "\n".join(lines)
    print(s)
    if a.out:
        a.out.write_text(s + "\n")


if __name__ == "__main__":
    main()
