"""Attach a worked example to each shape-port task: another pair of builds
of the same design (same directory, name template and core count), shown
as the unified diff between them.

    python3 add_examples.py ../npuasm/results/corpus.jsonl tasks [--max-diff-chars 30000]

The example never includes build B, so the answer cannot leak. Preference:
neither A nor B in it, then the same name positions changing as in A -> B,
then the smallest diff. Writes `example.json` and `example.diff` into each
task directory; tasks without a usable pair get none.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "npuasm"))

from npuasm import build  # noqa: E402
from make_tasks import HASH, template  # noqa: E402


def nums(name: str) -> list[str]:
    return re.findall(r"\d+", HASH.sub("", name))


def changed(a: str, b: str) -> tuple[int, ...]:
    x, y = nums(a), nums(b)
    return tuple(i for i, (p, q) in enumerate(zip(x, y)) if p != q)


def unified(a: dict[str, str], b: dict[str, str]) -> str:
    return "".join(
        "".join(difflib.unified_diff(a[f].splitlines(True), b[f].splitlines(True), f"a/{f}", f"b/{f}", n=2))
        for f in sorted(a)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("tasks", type=Path)
    ap.add_argument("--max-diff-chars", type=int, default=30_000)
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.corpus.open()]
    groups = collections.defaultdict(list)
    for r in rows:
        if r["error"] or not all(r["ok"].values()):
            continue
        b = Path(r["prj"]).parent
        groups[(str(b.parent), template(b.name), r["cores"])].append(r)
    lifted: dict[str, dict[str, str]] = {}

    def text(prj: str) -> dict[str, str]:
        if prj not in lifted:
            lifted[prj] = build.lift(Path(prj), None)[1]
        return lifted[prj]

    index = json.loads((a.tasks / "index.json").read_text())
    have = 0
    for m in index:
        td = a.tasks / m["id"]
        for f in ("example.json", "example.diff"):
            (td / f).unlink(missing_ok=True)
        key = (str(Path(m["a_prj"]).parent.parent), m["template"], m["cores"])
        members = [r["prj"] for r in groups.get(key, []) if r["prj"] != m["b_prj"]]
        want = changed(m["a_name"], m["b_name"])
        best = None
        for c in members:
            for d in members:
                if c == d:
                    continue
                cn, dn = Path(c).parent.name, Path(d).parent.name
                if HASH.sub("", cn) == HASH.sub("", dn):
                    continue
                touches_a = m["a_prj"] in (c, d)
                rank = (touches_a, changed(cn, dn) != want)
                if best is not None and rank > best[0]:
                    continue
                tc, tdd = text(c), text(d)
                if set(tc) != set(tdd) or set(tc) != {p.name for p in (td / "a").iterdir()}:
                    continue
                diff = unified(tc, tdd)
                if not diff or len(diff) > a.max_diff_chars:
                    continue
                cand = (rank, len(diff), cn, dn, diff)
                if best is None or cand[:2] < best[:2]:
                    best = cand
        if best is None:
            continue
        rank, n, cn, dn, diff = best
        (td / "example.diff").write_text(diff)
        (td / "example.json").write_text(json.dumps({"c_name": cn, "d_name": dn, "includes_a": rank[0],
                                                      "same_positions_change": not rank[1], "chars": n}, indent=1) + "\n")
        have += 1
    print(f"{have}/{len(index)} tasks have a worked example")


if __name__ == "__main__":
    main()
