"""Build shape-port tasks from the local corpus of aiecc builds.

A task pairs build A and build B of one design whose build names share a
template apart from their numbers (`attention_d256_r128` -> `_r512`) and
whose core count and lifted file set match. The model sees A's npuasm text
and both names and must return a unified diff that turns A into B. B's real
lifted text (the compiler's output) is the reference.

    python3 make_tasks.py ../npuasm/results/corpus.jsonl tasks/ [--max-chars 80000]

Each task directory holds `task.json`, `a/` (input files), `b/` (the
reference) and both builds' npuasm manifests. The split is by generator script (generators.py), so a training set can hold
out every build a test generator made, in any directory.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "npuasm"))

from npuasm import build  # noqa: E402
import generators  # noqa: E402

HASH = re.compile(r"_[0-9a-f]{8}$")


def template(name: str) -> str:
    return re.sub(r"\d+", "#", HASH.sub("", name))


def numbers(name: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", HASH.sub("", name)))


def diff_lines(a: dict[str, str], b: dict[str, str]) -> int:
    n = 0
    for f in a:
        sm = difflib.SequenceMatcher(None, a[f].splitlines(), b[f].splitlines(), autojunk=False)
        n += sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-chars", type=int, default=80_000)
    ap.add_argument("--max-diff", type=int, default=300)
    ap.add_argument("--per-group", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261103)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = [json.loads(l) for l in a.corpus.open()]
    groups = collections.defaultdict(list)
    for r in rows:
        if r["error"] or not all(r["ok"].values()):
            continue
        b = Path(r["prj"]).parent
        if "#" not in template(b.name):
            continue
        groups[(str(b.parent), template(b.name), r["cores"])].append(r)

    tasks, seen = [], set()
    for (design, tmpl, cores), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        members.sort(key=lambda r: numbers(Path(r["prj"]).parent.name))
        # Neighbouring shapes, in both directions, in random order.
        pairs = [(members[i], members[i + 1]) for i in range(len(members) - 1)]
        pairs += [(y, x) for x, y in pairs]
        rng.shuffle(pairs)
        kept = 0
        for ra, rb in pairs:
            if kept >= a.per_group:
                break
            if HASH.sub("", Path(ra["prj"]).parent.name) == HASH.sub("", Path(rb["prj"]).parent.name):
                continue  # names say nothing about what changed
            if sum(ra["text_chars"].values()) > a.max_chars:
                continue
            ma, ta, _ = build.lift(Path(ra["prj"]), None)
            mb, tb, _ = build.lift(Path(rb["prj"]), None)
            if set(ta) != set(tb):
                continue
            key = hashlib.sha256(json.dumps([tmpl, ta, tb], sort_keys=True).encode()).hexdigest()[:16]
            if key in seen:  # the same pair copied into another project
                continue
            d = diff_lines(ta, tb)
            if not 1 <= d <= a.max_diff:
                continue
            seen.add(key)
            kept += 1
            tasks.append((design, tmpl, ra, rb, ta, tb, d, key, ma, mb))

    a.out.mkdir(parents=True, exist_ok=True)
    families = sorted({t[0] for t in tasks})
    index = []
    for design, tmpl, ra, rb, ta, tb, d, key, ma, mb in tasks:
        tid = f"T{key[:10]}"
        td = a.out / tid
        for side, texts, man in (("a", ta, ma), ("b", tb, mb)):
            (td / side).mkdir(parents=True, exist_ok=True)
            for f, t in texts.items():
                (td / side / f).write_text(t)
            (td / f"manifest_{side}.json").write_text(json.dumps(man, indent=1) + "\n")
        meta = {
            "id": tid,
            "design": design.replace(str(Path.home()) + "/code/", ""),
            "template": tmpl,
            "a_name": Path(ra["prj"]).parent.name,
            "b_name": Path(rb["prj"]).parent.name,
            "a_prj": ra["prj"],
            "b_prj": rb["prj"],
            "cores": ra["cores"],
            "a_chars": sum(len(t) for t in ta.values()),
            "diff_lines": d,
            "difficulty": "easy" if d <= 10 else "medium" if d <= 80 else "hard",
            "generator": generators.generator(design, tmpl),
            "split": generators.split(design, tmpl),
        }
        (td / "task.json").write_text(json.dumps(meta, indent=1) + "\n")
        index.append(meta)
    (a.out / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    c = collections.Counter((m["split"], m["difficulty"]) for m in index)
    print(f"{len(index)} tasks from {len(families)} design families: {dict(sorted(c.items()))}")


if __name__ == "__main__":
    main()
