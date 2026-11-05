"""Functional correctness per run, and a paired comparison of runs.

    python3 functional.py [--control runs/null-a] runs/gpt-6.1-sol-medium runs/claude-opus-5-5-xhigh

A task is functionally correct when the answer's text is exact, its rebuild
is byte-identical to B, or `npu_check.py` found it `equal`/`close` to B
(rule-gate alignment repair included). The guarded verdict (npu_guard.json,
out-of-bounds writes caught) replaces the plain one when present.
Control: when A unchanged (the `null` run) itself passes the NPU check on a
task, that task's check can't tell a port from no port, so an NPU-only pass
there counts as "unverifiable", not correct. runs/null-a is used when present.
Failures, in order of precedence: doesn't build, rule gate (fails even after
alignment repair), hang, wrong values (`differs`), unverifiable, not checked
(no npu.json yet), not scored (`vacuous`, `b_nondeterministic`, `b_failed`,
`skipped`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DIFFS = ("easy", "medium", "hard")


def npu_verdict(rd: Path) -> str | None:
    for name in ("npu_guard.json", "npu.json"):
        if (rd / name).exists():
            return json.loads((rd / name).read_text())["verdict"]
    return None


def verdict(rd: Path, weak: set[str] = frozenset()) -> tuple[str, dict]:
    r = json.loads((rd / "result.json").read_text())
    s = r["score"]
    if s["exact"] or s["bytes_match"]:
        return "correct", r
    if not s["builds"]:
        return "doesn't build", r
    if not (s["rules_pass"] or s.get("rules_pass_aligned")):
        return "rule gate", r
    v = npu_verdict(rd)
    if v is None:
        return "not checked", r
    if v in ("equal", "close") and rd.name in weak:
        return "unverifiable", r
    return {"equal": "correct", "close": "correct", "differs": "wrong values", "hang": "hang"}.get(v, "not scored"), r


def load(run: Path, weak: set[str] = frozenset()) -> dict[str, tuple[str, dict]]:
    return {rd.name: verdict(rd, weak) for rd in sorted(p.parent for p in run.glob("*/result.json"))}


def weak_tasks(control: Path) -> set[str]:
    """Tasks where the no-edit answer passes the NPU check."""
    return {rd.name for rd in (p.parent for p in control.glob("*/result.json")) if npu_verdict(rd) in ("equal", "close")}


def table(runs: dict[str, dict]) -> None:
    print("| Run | " + " | ".join(d.capitalize() for d in DIFFS) + " | All | Failures |")
    print("|---|" + "---:|" * (len(DIFFS) + 1) + "---|")
    for name, res in runs.items():
        cells = []
        for d in DIFFS:
            sel = [v for v, r in res.values() if r["difficulty"] == d]
            cells.append(f"{sum(v == 'correct' for v in sel)}/{len(sel)}")
        fails: dict[str, int] = {}
        for v, _ in res.values():
            if v != "correct":
                fails[v] = fails.get(v, 0) + 1
        ok = sum(v == "correct" for v, _ in res.values())
        print(f"| {name} | " + " | ".join(cells) + f" | **{ok}/{len(res)}** | "
              + ", ".join(f"{n} {k}" for k, n in sorted(fails.items(), key=lambda kv: -kv[1])) + " |")


def paired(a_name: str, a: dict, b_name: str, b: dict) -> None:
    ids = sorted(set(a) & set(b))
    print(f"\nPaired on {len(ids)} tasks: {a_name} vs {b_name}\n")
    print(f"| Difficulty | Tasks | {a_name} | {b_name} | Both | {a_name} only | {b_name} only |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for d in DIFFS + ("all",):
        sel = [i for i in ids if d == "all" or a[i][1]["difficulty"] == d]
        ca = {i for i in sel if a[i][0] == "correct"}
        cb = {i for i in sel if b[i][0] == "correct"}
        bold = "**" if d == "all" else ""
        print(f"| {d.capitalize()} | {len(sel)} | {bold}{len(ca)}{bold} | {bold}{len(cb)}{bold} | {len(ca & cb)} | {len(ca - cb)} | {len(cb - ca)} |")
    only_a = [i for i in ids if a[i][0] == "correct" and b[i][0] != "correct"]
    only_b = [i for i in ids if b[i][0] == "correct" and a[i][0] != "correct"]
    for label, xs, other in ((f"{a_name} only", only_a, b), (f"{b_name} only", only_b, a)):
        if xs:
            print(f"\n{label}: " + ", ".join(f"{i} ({a[i][1]['difficulty']}; other: {other[i][0]})" for i in xs))


def main() -> None:
    args = sys.argv[1:]
    control = Path(__file__).parent / "runs" / "null-a"
    if args[:1] == ["--control"]:
        control, args = Path(args[1]), args[2:]
    weak = weak_tasks(control) if control.exists() else set()
    print(f"control {control.name}: {len(weak)} tasks where A unchanged passes; NPU-only passes there don't count\n")
    runs = {Path(p).name: load(Path(p), weak) for p in args}
    table(runs)
    names = list(runs)
    for x, y in zip(names, names[1:]):
        paired(x, runs[x], y, runs[y])


if __name__ == "__main__":
    main()
