"""Where the CDO tokens go: tokens of the lifted CDO text by register block.

    ../../target/manifold-audit/env/bin/python scripts/cdo_breakdown.py \
        results/corpus.jsonl ../../models/manifold-audit/Qwen3.5-9B/tokenizer.json \
        --out results/cdo-breakdown.md

Same sample as count_tokens.py: one build per design family, the one with
the most core code. Each op line (with its data lines) is tokenized on its
own and charged to the register block at its start address, so the totals
are close to, not equal to, whole-file counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizers import Tokenizer  # noqa: E402

from npuasm import build, cdo, regs  # noqa: E402
from count_tokens import family  # noqa: E402


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
    tokens: Counter = Counter()
    ops: Counter = Counter()
    per_build = []
    for f, r in sorted(best.items()):
        files = build.find(Path(r["prj"]))
        total = 0
        for part, path in files.cdos.items():
            for op in cdo.parse(path.read_bytes()):
                if op.cmd in (cdo.WRITE64, cdo.MASK_WRITE64, cdo.MASK_POLL64, cdo.DMA_WRITE):
                    key = f"{part}/{regs.block(op.args[1])}"
                else:
                    key = f"{part}/{op.name}"
                if op.cmd == cdo.DMA_WRITE and part == "elfs" and regs.block(op.args[1]) == "core:PROGRAM_MEMORY":
                    n = 8  # becomes `dmawrite cCrR+0x20000 @core_cCrR` in the lifted text
                else:
                    n = len(tok.encode(cdo.to_text([op])).ids)
                tokens[key] += n
                ops[key] += 1
                total += n
        per_build.append(total)
    grand = sum(tokens.values())
    lines = [
        f"{len(best)} builds (one per family); {grand:,} CDO tokens in all; median per build {sorted(per_build)[len(per_build) // 2]:,}",
        "",
        "| part / block | ops | tokens | share |",
        "|---|---:|---:|---:|",
    ]
    for k, n in tokens.most_common(25):
        lines.append(f"| {k} | {ops[k]:,} | {n:,} | {100 * n / grand:.1f}% |")
    s = "\n".join(lines)
    print(s)
    if a.out:
        a.out.write_text(s + "\n")


if __name__ == "__main__":
    main()
