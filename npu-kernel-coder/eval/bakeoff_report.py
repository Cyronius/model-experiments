"""Bakeoff tables: local candidates and the ceiling models on the 12 pilot tasks.

    python3 bakeoff_report.py

NPU checks of model answers are off (runs/.npu_disabled, after the
2026-11-05 freeze), so local models are scored without the NPU:
"Correct w/o NPU" counts answers whose text is exact or whose rebuild is
byte-identical to B; "Rule gate" counts answers that build and pass the
rule gate (the ones an NPU check could pass). The reference runs' functional
score (functional.py, NPU checks from before the freeze) is shown for
context. Throughput is llama-server's own timing per request: prompt
(prefill) and generation tokens per second, medians over the tasks.
Degenerate: answers that are mostly one repeated character (the "GGGG"
flash-attention bug).

The second table compares each model with and without one worked example
from the same design (variant ex1) on the 6 tasks that have one.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import functional
import run

HERE = Path(__file__).parent
REFS = ["gpt-6.1-sol-medium", "gpt-6.1-sol-xhigh", "claude-opus-5-5-xhigh", "baseline-scale"]
DEGEN = re.compile(r"(.)\1{200,}")


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, spec):
    return "-" if x is None else format(x, spec)


def stats(rd: Path, tasks: set[str], weak: set[str]) -> dict | None:
    res = {k: v for k, v in functional.load(rd, weak).items() if k in tasks}
    if not res:
        return None
    rows = [r for _, r in res.values()]
    sc = [r["score"] for r in rows]
    tim = [r["run"].get("timings") or {} for r in rows]
    degen = 0
    for k in res:
        a = rd / k / "answer.txt"
        if a.exists() and DEGEN.search(a.read_text(errors="replace")):
            degen += 1
    return {
        "n": len(res),
        "functional": sum(v == "correct" for v, _ in res.values()),
        "textual": sum(bool(s["exact"] or s["bytes_match"]) for s in sc),
        "builds": sum(bool(s["builds"]) for s in sc),
        "gate": sum(bool(s["builds"] and (s["rules_pass"] or s.get("rules_pass_aligned"))) for s in sc),
        "hunks": sum(s.get("hunks_ok", 0) for s in sc) / max(1, sum(s.get("hunks_total", 0) for s in sc)),
        "gap": med([s["gap_closed"] for s in sc]),
        "degen": degen,
        "pre": med([t.get("prompt_per_second") for t in tim]),
        "dec": med([t.get("predicted_per_second") for t in tim]),
        "cap": sum(r["run"].get("finish") == "length" for r in rows),
        "sec": med([r["run"].get("seconds") for r in rows]),
        "unchecked": sum(v == "not checked" for v, _ in res.values()),
    }


def main() -> None:
    index = json.loads((HERE / "tasks/index.json").read_text())
    pilot = {m["id"] for m in run.pilot(index)}
    control = HERE / "runs" / "null-a"
    weak = functional.weak_tasks(control) if control.exists() else set()
    base = [p for p in sorted(HERE.glob("runs/bakeoff-*")) if not p.name.endswith("-ex1")]
    refs = [HERE / "runs" / r for r in REFS]

    print("| Model | Tasks | Correct w/o NPU | Builds | Rule gate | Hunks applied | Gap median | Degenerate | Hit cap | Prefill tok/s | Decode tok/s | Seconds median | Functional (NPU, refs only) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rd in base + refs:
        if not rd.exists() or not (s := stats(rd, pilot, weak)):
            continue
        func = f"{s['functional']}" if rd in refs else "-"
        print(f"| {rd.name.removeprefix('bakeoff-')} | {s['n']} | **{s['textual']}** | {s['builds']} | {s['gate']} | "
              f"{s['hunks']:.0%} | {fmt(s['gap'], '.3f')} | {s['degen']} | {s['cap']} | {fmt(s['pre'], '.0f')} | "
              f"{fmt(s['dec'], '.1f')} | {fmt(s['sec'], '.0f')} | {func} |")

    ex = [p for p in sorted(HERE.glob("runs/bakeoff-*-ex1"))]
    if not ex:
        return
    print()
    print("| Model | Tasks | Correct w/o NPU: base → ex1 | Builds | Rule gate | Hunks applied | Gap median |")
    print("|---|---:|---|---|---|---|---|")
    for e in ex:
        tasks = {p.parent.name for p in e.glob("*/result.json")} & pilot
        b = HERE / "runs" / e.name.removesuffix("-ex1")
        sb, se = stats(b, tasks, weak) if b.exists() else None, stats(e, tasks, weak)
        if not se:
            continue
        if not sb:
            sb = {k: None for k in se}

        def arrow(k, spec="d"):
            return f"{fmt(sb[k], spec)} → {fmt(se[k], spec)}"

        print(f"| {b.name.removeprefix('bakeoff-')} | {se['n']} | **{arrow('textual')}** | {arrow('builds')} | {arrow('gate')} | "
              f"{arrow('hunks', '.0%')} | {arrow('gap', '.3f')} |")


if __name__ == "__main__":
    main()
