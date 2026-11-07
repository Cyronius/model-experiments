"""Inputs for a repair pass over the accepted rewrites of the final set.

Reads data/out/doc-teacher-v5/<final>/accepted.jsonl (merge_v5.py) and writes, for each
rewrite to repair, the pool row of the passage with repair_prev (the rewrite with its
literal markers, from the attempts of its run), repair_note (the feedback) and repair_of
(the rewrite, its teacher and run). pipeline_v5.py starts such a row as a retry.
--reason second_judge: the rewrites whose second meaning check found differences; the
note is the v4 feedback built from them. --reason model: the unflagged rewrites that
the student model cannot encode (model_gate_v5.py), except those that drop a plain word
or whose input does not encode; the note is the gate's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent)]
from ste_context_v4 import feedback_v4  # noqa: E402
from checker import Lexicon  # noqa: E402
from gen import proper_names  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/out/doc-teacher-v5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", default="final")
    parser.add_argument("--pool", default=str(ROOT / "data/out/doc-pool-v5/pool.jsonl"))
    parser.add_argument("--out", default=str(BASE / "repair-inputs.jsonl"))
    parser.add_argument("--reason", choices=("second_judge", "model"), default="second_judge")
    parser.add_argument("--model-gate", default="", help="vocabulary directory for --reason model")
    args = parser.parse_args()
    lex = Lexicon()
    pool = {r["id"]: r for r in map(json.loads, Path(args.pool).read_text().splitlines())}
    rows = [json.loads(l) for l in (BASE / args.final / "accepted.jsonl").read_text().splitlines()]
    notes = {}
    if args.reason == "second_judge":
        for r in rows:
            d = (r["review_flags"]["second_judge"] or {}).get("differences")
            if d:
                notes[r["id"]] = feedback_v4([f"meaning_check:{len(d)}"], r["text"], r["ste"], r["mode"], lex,
                                             proper_names(r["text"]), d)
    else:
        from model_gate_v5 import TOKENIZER, ModelGate
        gate = ModelGate(lex, Path(args.model_gate) if args.model_gate else TOKENIZER)
        for r in rows:
            if (r["review_flags"]["second_judge"] or {}).get("differences"):
                continue
            issues, note = gate.problems(r["text"], r["ste"], r["mode"])
            if issues and note and not any(i.startswith(("model_dropped_word", "model_input")) for i in issues):
                notes[r["id"]] = note
    flagged = [r for r in rows if r["id"] in notes]
    marked = {}
    for run in sorted({r["run"] for r in flagged}):
        for a in map(json.loads, (BASE / run / "attempts.jsonl").read_text().splitlines()):
            if a.get("target") and not a["issues"]:
                marked[(a["id"], a["target"])] = a["rewrite_with_markers"]
    with Path(args.out).open("w", encoding="utf-8") as f:
        for r in flagged:
            row = {**pool[r["id"]], "repair_prev": marked[(r["id"], r["ste"])], "repair_note": notes[r["id"]],
                   "repair_of": {"ste": r["ste"], "teacher_model": r["teacher_model"], "run": r["run"]}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"reason": args.reason, "rows": len(flagged), "out": args.out}))


if __name__ == "__main__":
    main()
