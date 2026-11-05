"""Do runs fail on different tasks, and are their correct answers different?

    python3 ensemble.py runs/gpt-6.1-sol-medium runs/gpt-6.1-sol-xhigh runs/claude-opus-5-5-xhigh

Coverage: per difficulty, how many tasks each run solves and how many any
run solves (the union, i.e. best-of-N with a perfect selector). Selection here
uses B, which a deployed system doesn't have; it would need a CPU reference
or the hardware check against one.

Variety: for tasks with correct answers from two or more runs, how many
distinct texts they are (after applying each diff to A), whether they equal
B, and how far apart they are (changed lines, as in score.dist). Correct
alternatives to B are possible extra training targets for the same prompt.
"""

from __future__ import annotations

import itertools
import json
import statistics
import sys
from pathlib import Path

import functional
import score

HERE = Path(__file__).parent
DIFFS = ("easy", "medium", "hard")


def predicted(task: str, run: Path) -> dict[str, str]:
    td = HERE / "tasks" / task
    a = {p.name: p.read_text() for p in (td / "a").iterdir()}
    pred = dict(a)
    for f, h in score.parse_diff((run / task / "answer.txt").read_text()).items():
        if f in a:
            pred[f] = "\n".join(score.apply(a[f].splitlines(), h)[0]) + "\n"
    return pred


def main() -> None:
    runs = [Path(p) for p in sys.argv[1:]]
    control = HERE / "runs" / "null-a"
    weak = functional.weak_tasks(control) if control.exists() else set()
    res = {r.name: functional.load(r, weak) for r in runs}
    ids = sorted(set.intersection(*(set(v) for v in res.values())))
    diff = {i: next(iter(res.values()))[i][1]["difficulty"] for i in ids}
    ok = {name: {i for i in ids if res[name][i][0] == "correct"} for name in res}

    print(f"Coverage on {len(ids)} tasks ({len(weak)} with an unverifiable NPU check)\n")
    names = list(res)
    head = " | ".join(names)
    print(f"| Difficulty | Tasks | {head} | Any of them | All of them |")
    print("|---|---:|" + "---:|" * len(names) + "---:|---:|")
    for d in DIFFS + ("all",):
        sel = {i for i in ids if d == "all" or diff[i] == d}
        cells = " | ".join(str(len(ok[n] & sel)) for n in names)
        anyof = set.union(*(ok[n] & sel for n in names))
        allof = set.intersection(*(ok[n] & sel for n in names))
        print(f"| {d.capitalize()} | {len(sel)} | {cells} | **{len(anyof)}** | {len(allof)} |")
    if len(names) > 2:
        print("\nPairs (tasks solved by either):")
        for x, y in itertools.combinations(names, 2):
            print(f"  {x} + {y}: {len(ok[x] | ok[y])}  (only {x}: {len(ok[x] - ok[y])}, only {y}: {len(ok[y] - ok[x])})")
    unsolved = [i for i in ids if not any(i in ok[n] for n in names)]
    print(f"\nSolved by none ({len(unsolved)}): " + ", ".join(f"{i} {diff[i]}" for i in unsolved))

    print("\nVariety among correct answers (tasks with 2+ correct runs):\n")
    print("| Difficulty | Tasks | Distinct correct texts, total | Tasks with 2+ distinct | Texts ≠ B | Programs ≠ B (bytes differ) | Median distance between distinct correct answers (lines) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    detail = []
    for d in DIFFS + ("all",):
        sel = [i for i in ids if (d == "all" or diff[i] == d) and sum(i in ok[n] for n in names) >= 2]
        n_distinct = n_multi = n_not_b = n_prog = 0
        dists = []
        for i in sel:
            b = {p.name: p.read_text() for p in (HERE / "tasks" / i / "b").iterdir()}
            texts, progs = [], []
            for n in names:
                if i in ok[n]:
                    t = predicted(i, Path(next(r for r in runs if r.name == n)))
                    if t not in texts:
                        texts.append(t)
                        progs.append(not res[n][i][1]["score"]["bytes_match"])
            n_distinct += len(texts)
            n_multi += len(texts) >= 2
            n_not_b += sum(t != b for t in texts)
            n_prog += sum(progs)
            pair = [score.dist(x, y) for x, y in itertools.combinations(texts, 2)]
            dists += pair
            if d == "all" and len(texts) >= 2:
                detail.append((i, diff[i], len(texts), sum(progs), pair))
        med = f"{statistics.median(dists):.0f}" if dists else "-"
        print(f"| {d.capitalize()} | {len(sel)} | {n_distinct} | {n_multi} | {n_not_b} | {n_prog} | {med} |")
    if detail:
        print("\nTasks with 2+ distinct correct answers: " + ", ".join(
            f"{i} {d} ({k} texts, {nb} new programs, apart {'/'.join(map(str, p))})" for i, d, k, nb, p in detail))


if __name__ == "__main__":
    main()
