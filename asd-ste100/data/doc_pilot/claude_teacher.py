"""Claude Opus 5.5 subagents as the doc teacher, on pool v6, with the v4 gates and an independent meaning check.

Each writer subagent gets one chunk. It reads chunks/cNNN.task.md and the v5 "traps" rewrite prompt,
writes cNNN.out.jsonl ({"id", "rewrite", "uncertain"}; the last line for an id counts), and runs
`check NNN` (marker check, restore, verify_v4) until its rewrites pass or it gives up. A separate
subagent then reads cNNN.judge.md with the traps meaning-check prompt and writes cNNN.judge.out.jsonl
({"id", "differences": [...]}); `judged NNN` writes cNNN.final.jsonl in the pipeline_v5 record format.
`merge` collects accepted.jsonl / rejected.jsonl. Records are teacher candidates, not gold.

    python claude_teacher.py prep START COUNT SIZE    # write chunk files and the prompts
    python claude_teacher.py check NNN | judged NNN | merge
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent)]
from gates_v3 import dropped_words  # noqa: E402
from gates_v4 import marker_problems, restore_any_order, verify_v4  # noqa: E402
from openrouter_v5 import JUDGE_BATCH, REWRITE_BATCH  # noqa: E402
from ste_context_v4 import feedback_v4, judge_message_v4, marker_feedback_v4, source_hints_v4  # noqa: E402
from ste_context_v5 import judge_prompt, rewrite_prompt  # noqa: E402
from synthesize import mask_literals  # noqa: E402
from checker import Lexicon  # noqa: E402
from gen import proper_names  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "data/out/doc-pool-v6/pool.jsonl"
OUT = ROOT / "data/out/doc-teacher-claude"
CH = OUT / "chunks"
TEACHER = "claude-opus-5-5 (subagent)"


def order():
    rows = [json.loads(l) for l in POOL.read_text().splitlines()]
    assert not any(r["family"] == "rust_book" for r in rows)
    key = lambda r: (r["mode"] != "PROC", hashlib.sha256(r["id"].encode()).hexdigest())
    return sorted(rows, key=key)   # every procedure first, then a family mix


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def prep(start, count, size):
    lex = Lexicon()
    CH.mkdir(parents=True, exist_ok=True)
    (OUT / "rewrite-prompt.txt").write_text(rewrite_prompt(lex, "traps") + "\n\n" + REWRITE_BATCH + "\n")
    (OUT / "judge-prompt.txt").write_text(judge_prompt(lex, "traps") + "\n\n" + JUDGE_BATCH + "\n")
    rows = order()
    for n in range(start, start + count):
        part = rows[n * size:(n + 1) * size]
        items, blocks = [], []
        for k, row in enumerate(part):
            marked, literals = mask_literals(row["text"])
            names = proper_names(row["text"])
            hints = source_hints_v4(row["text"], row["mode"], lex, names)
            items.append({"key": f"p{k}", "row": row, "marked": marked, "literals": literals, "names": sorted(names), "hints": hints})
            blocks.append(f"### id: p{k}\nMode: {row['mode']}\nDocument family: {row['family']}\nDocument path: "
                          f"{row['doc_path']}\nChecker findings on this passage:\n{hints}\nPassage:\n{marked}")
        (CH / f"c{n:03d}.in.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
        (CH / f"c{n:03d}.task.md").write_text("\n\n".join(blocks) + "\n")
    print(f"chunks {start}..{start + count - 1}, {size} passages each")


def latest(path, field):
    out = {}
    for e in load(path):
        if isinstance(e, dict) and "id" in e and field in e:
            out[str(e["id"])] = e
    return out


def check(n):
    lex = Lexicon()
    items = load(CH / f"c{n:03d}.in.jsonl")
    outs = latest(CH / f"c{n:03d}.out.jsonl", "rewrite")
    gates, judge, passed = [], [], 0
    for it in items:
        it["names"] = set(it["names"])
        row, e = it["row"], outs.get(it["key"])
        rewrite = (e or {}).get("rewrite") or ""
        target, issues, note = None, [], ""
        if not rewrite.strip():
            issues, note = ["missing"], "no rewrite yet"
        elif marker_problems(rewrite, len(it["literals"])):
            issues, note = ["marker_change"], marker_feedback_v4(rewrite, it["marked"], len(it["literals"]))
        else:
            target = restore_any_order(rewrite, it["literals"])
            issues = verify_v4(row["text"], target, row["mode"], lex, bool(e.get("uncertain")), it["names"])
            if issues:
                note = feedback_v4(issues, row["text"], target, row["mode"], lex, it["names"], [], False)
        gates.append({"key": it["key"], "rewrite": rewrite or None, "target": target, "issues": issues})
        if target and not issues:
            passed += 1
            judge.append(f"### id: {it['key']}\n" + judge_message_v4(row["text"], target, row["mode"], lex, it["names"]))
        else:
            print(f"FAIL {it['key']}: {', '.join(issues)}\n{note}\n")
    (CH / f"c{n:03d}.gates.jsonl").write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in gates))
    (CH / f"c{n:03d}.judge.md").write_text("\n\n".join(judge) + "\n")
    print(f"PASSED {passed} of {len(items)}; judge input: {CH / f'c{n:03d}.judge.md'}")


def judged(n):
    lex = Lexicon()
    items = {i["key"]: i for i in load(CH / f"c{n:03d}.in.jsonl")}
    gates = load(CH / f"c{n:03d}.gates.jsonl")
    checks = latest(CH / f"c{n:03d}.judge.out.jsonl", "differences")
    final = []
    for g in gates:
        it, row = items[g["key"]], items[g["key"]]["row"]
        it["names"] = set(it["names"])
        issues, d = list(g["issues"]), None
        if g["target"] and not issues:
            d = (checks.get(g["key"]) or {}).get("differences")
            issues = ["meaning_check_unparsed"] if not isinstance(d, list) else [f"meaning_check:{len(d)}"] if d else []
        final.append({**row, "ste": g["target"], "issues": issues, "review_status": "teacher_candidate_owner_unverified",
                      "teacher_model": TEACHER, "judge_model": TEACHER + ", independent context", "source_hints": it["hints"],
                      "meaning_check": d, "chunk": n,
                      "review_flags": {"dropped_words": dropped_words(row["text"], g["target"], lex, it["names"])
                                       if g["target"] else []}})
    (CH / f"c{n:03d}.final.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in final))
    print(f"accepted {sum(not r['issues'] for r in final)} of {len(final)}")


def merge():
    rows = [r for p in sorted(CH.glob("c*.final.jsonl")) for r in load(p)]
    for name, keep in (("accepted.jsonl", lambda r: not r["issues"]), ("rejected.jsonl", lambda r: r["issues"])):
        (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if keep(r)))
    print(json.dumps({"chunks": len(list(CH.glob("c*.final.jsonl"))), "passages": len(rows),
                      "accepted": sum(not r["issues"] for r in rows)}))


if __name__ == "__main__":
    cmd, rest = sys.argv[1], [int(x) for x in sys.argv[2:]]
    {"prep": prep, "check": check, "judged": judged, "merge": merge}[cmd](*rest)
