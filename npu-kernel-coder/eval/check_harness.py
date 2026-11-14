"""Oracle check of the scorer: a perfect diff (difflib's own A -> B) must be
exact and rebuild to B's vendor bytes; an empty answer must close no gap.

    python3 check_harness.py tasks
"""
import collections
import difflib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import score


def one(td):
    td = Path(td)
    a = {p.name: p.read_text() for p in (td / "a").iterdir()}
    b = {p.name: p.read_text() for p in (td / "b").iterdir()}
    perfect = "".join("".join(difflib.unified_diff(a[f].splitlines(True), b[f].splitlines(True), f"a/{f}", f"b/{f}", n=3)) for f in sorted(a))
    return td.name, score.score(td, perfect), score.score(td, "")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    ix = json.loads((root / "index.json").read_text())
    st, bad = collections.Counter(), []
    with ProcessPoolExecutor(12) as pool:
        for tid, sp, se in pool.map(one, [str(root / m["id"]) for m in ix]):
            st["perfect: exact"] += sp["exact"]
            st["perfect: bytes_match"] += sp["bytes_match"]
            st["perfect: rules_pass"] += sp["rules_pass"]
            st["empty: gap_closed 0"] += se["gap_closed"] == 0
            st["empty: builds"] += se["builds"]
            if not (sp["exact"] and sp["bytes_match"]):
                bad.append((tid, {k: v for k, v in sp.items() if k != "rule_violations"}))
    print(len(ix), "tasks", dict(st))
    for b in bad[:5]:
        print(b)
