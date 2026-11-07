"""v5 production run on OpenRouter: batched rewrites, the v4 gates, batched meaning checks, retries.

The v4 method, with passages in batches: each batch of --batch passages goes through up
to --attempts rounds. A round sends one rewrite request for the passages still open (a
retry carries the previous rewrite and the checks it failed, as in v4), runs the v4
marker check and gates on each rewrite, and sends one meaning-check request for those
that passed (--judge, a model of a different family than the rewriter is the point).
A passage is accepted when its rewrite passes the gates and the check finds no
difference. --rewriter can name several models: batches take them in turn.
--judge2 adds a second check after a clean first one; a passage then needs both clean.
A row with repair_prev and repair_note starts as a retry of that rewrite.
--model-gate adds the student model's contract (model_gate_v5.py) to the gates.
--max-cost stops new requests once the spend reported by OpenRouter reaches it.
Outputs go to data/out/doc-teacher-v5/<run>/ (accepted, rejected, attempts, summary);
a rerun skips the passages already recorded (also those of --skip-runs). Records are teacher candidates, not gold.
"""

from __future__ import annotations

import sys
from pathlib import Path

# data/doc_pilot/queue.py shadows the standard library's queue, which the thread pool needs.
_path, sys.path = sys.path, [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
import queue  # noqa: E402,F401
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
sys.path = _path

import argparse  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

sys.path[:0] = [str(Path(__file__).parent)]
from gates_v3 import dropped_words  # noqa: E402
from gates_v4 import marker_problems, restore_any_order, verify_v4  # noqa: E402
from openrouter_v5 import JUDGE_BATCH, REWRITE_BATCH, ask_batch  # noqa: E402
from ste_context_v4 import feedback_v4, judge_message_v4, marker_feedback_v4, source_hints_v4  # noqa: E402
from ste_context_v5 import judge_prompt, rewrite_prompt  # noqa: E402
from synthesize import append, mask_literals, sha  # noqa: E402
from checker import Lexicon  # noqa: E402
from gen import proper_names  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/out/doc-teacher-v5"
RETRY = ("Your previous rewrite of this passage:\n{prev}\nIt failed these checks:\n{note}\nFix every point. Keep the "
         "full meaning and every literal marker. If you cannot, copy the input and set uncertain to true.")


class Run:
    def __init__(self, args):
        self.args, self.lex = args, Lexicon()
        self.out = BASE / args.run
        self.out.mkdir(parents=True, exist_ok=True)
        self.rewrite_system = rewrite_prompt(self.lex, args.variant) + "\n\n" + REWRITE_BATCH
        self.judge_system = judge_prompt(self.lex, args.variant) + "\n\n" + JUDGE_BATCH
        self.rewriters = args.rewriter.split(",")
        self.gate = None
        if args.model_gate:
            from model_gate_v5 import ModelGate
            self.gate = ModelGate(self.lex, Path(args.model_gate))
        self.lock, self.cost, self.stats = threading.Lock(), 0.0, Counter()

    def ask(self, model, system, blocks, key, reasoning, kind):
        with self.lock:
            if self.cost >= self.args.max_cost:
                raise RuntimeError("max cost reached")
        entries, usage, seconds = ask_batch(model, system, blocks, key, reasoning)
        with self.lock:
            self.cost += usage.get("cost") or 0
            self.stats.update({f"{kind}_requests": 1, f"{kind}_items": len(blocks),
                               f"{kind}_prompt_tokens": usage.get("prompt_tokens") or 0,
                               f"{kind}_output_tokens": usage.get("completion_tokens") or 0,
                               f"{kind}_ms": int(seconds * 1000)})
        return entries

    def batch(self, index: int, rows: list[dict]) -> list[dict]:
        a, lex = self.args, self.lex
        model = self.rewriters[index % len(self.rewriters)]
        items = {}
        for k, row in enumerate(rows):
            marked, literals = mask_literals(row["text"])
            names = proper_names(row["text"])
            hints = source_hints_v4(row["text"], row["mode"], lex, names)
            items[f"p{k}"] = {"row": row, "marked": marked, "literals": literals, "names": names, "hints": hints,
                              "base": f"Mode: {row['mode']}\nDocument family: {row['family']}\nDocument path: "
                                      f"{row['doc_path']}\nChecker findings on this passage:\n{hints}\nPassage:\n{marked}",
                              "prev": row.get("repair_prev"), "note": row.get("repair_note"),
                              "earlier": [row["repair_prev"]] if row.get("repair_prev") else [],
                              "final": None, "gates_only": None, "done": False}
        for attempt in range(1, a.attempts + 1):
            open_ = [k for k, it in items.items() if not it["done"]]
            if not open_:
                break
            blocks = [(k, items[k]["base"] + ("\n" + RETRY.format(prev=items[k]["prev"], note=items[k]["note"])
                                             if items[k]["prev"] else "")) for k in open_]
            try:
                entries = self.ask(model, self.rewrite_system, blocks, "rewrites", a.rewriter_reasoning, "rewrite")
            except RuntimeError as exc:
                for k in open_:
                    items[k]["final"] = {"attempt": attempt, "issues": [f"request_failed: {exc}"[:200]], "target": None}
                    items[k]["done"] = True
                break
            to_judge = []
            for k in open_:
                it, row = items[k], items[k]["row"]
                e = entries.get(k) or {}
                rewrite = (e.get("rewrite") or "").strip() if isinstance(e.get("rewrite"), str) else ""
                target, markers, issues, gate_note = None, None, [], ""
                if not rewrite:
                    issues = ["missing_from_batch"]
                elif marker_problems(rewrite, len(it["literals"])):
                    issues, markers = ["marker_change"], marker_feedback_v4(rewrite, it["marked"], len(it["literals"]))
                else:
                    target = restore_any_order(rewrite, it["literals"])
                    issues = verify_v4(row["text"], target, row["mode"], lex, bool(e.get("uncertain")), it["names"])
                    if not issues and self.gate:
                        issues, gate_note = self.gate.problems(row["text"], target, row["mode"])
                repeated = bool(rewrite) and rewrite in it["earlier"]
                it["earlier"].append(rewrite)
                it["final"] = {"attempt": attempt, "model": model, "rewrite_with_markers": rewrite or None,
                               "target": target, "issues": issues, "repeated": repeated, "meaning_check": None}
                if target and not issues:
                    to_judge.append(k)
                    if it["gates_only"] is None:
                        it["gates_only"] = {"attempt": attempt, "ste": target}
                elif rewrite:
                    it["prev"] = rewrite
                    it["note"] = ("- " + markers) if markers else gate_note if gate_note else (
                        feedback_v4(issues, row["text"], target, row["mode"], lex, it["names"], [], repeated) if target
                        else "- " + ", ".join(issues))
                    if any(i.startswith(("model_dropped_word", "model_input")) for i in issues):
                        it["done"] = True     # not repaired: see model_gate_v5
            if to_judge and a.judge != "none":
                jblocks = [(k, judge_message_v4(items[k]["row"]["text"], items[k]["final"]["target"], items[k]["row"]["mode"],
                                                lex, items[k]["names"])) for k in to_judge]
                try:
                    checks = self.ask(a.judge, self.judge_system, jblocks, "checks", a.judge_reasoning, "judge")
                except RuntimeError as exc:
                    checks = {}
                    self.stats[f"judge_failed: {str(exc)[:60]}"] += 1
                clean = []
                for k in to_judge:
                    it, row = items[k], items[k]["row"]
                    d = (checks.get(k) or {}).get("differences")
                    it["final"]["meaning_check"] = d
                    if not isinstance(d, list):
                        it["final"]["issues"] = ["meaning_check_unparsed"]
                        it["done"] = True     # as in v4: no retry without the check's findings
                    elif d:
                        it["final"]["issues"] = [f"meaning_check:{len(d)}"]
                        it["prev"] = it["final"]["rewrite_with_markers"]
                        it["note"] = feedback_v4(it["final"]["issues"], row["text"], it["final"]["target"], row["mode"],
                                                 lex, it["names"], d, it["final"]["repeated"])
                    else:
                        clean.append(k)
                if a.judge2 and clean:
                    message, checks2 = dict(jblocks), {}
                    for _ in range(2):     # the second time for the checks missing from the replies
                        todo = [k for k in clean if not isinstance((checks2.get(k) or {}).get("differences"), list)]
                        for i in range(0, len(todo), a.judge2_batch):
                            part = todo[i:i + a.judge2_batch]
                            try:
                                checks2.update(self.ask(a.judge2, self.judge_system, [(k, message[k]) for k in part],
                                                        "checks", a.judge2_reasoning, "judge2"))
                            except RuntimeError as exc:
                                self.stats[f"judge2_failed: {str(exc)[:60]}"] += 1
                    for k in clean:
                        it, row = items[k], items[k]["row"]
                        d = (checks2.get(k) or {}).get("differences")
                        it["final"]["meaning_check2"] = d
                        if not isinstance(d, list):
                            it["final"]["issues"] = ["meaning_check2_unparsed"]
                        elif d:
                            it["final"]["issues"] = [f"meaning_check2:{len(d)}"]
                            it["prev"] = it["final"]["rewrite_with_markers"]
                            it["note"] = feedback_v4(it["final"]["issues"], row["text"], it["final"]["target"],
                                                     row["mode"], lex, it["names"], d, it["final"]["repeated"])
                            continue
                        it["done"] = True
                else:
                    for k in clean:
                        items[k]["done"] = True
            else:
                for k in to_judge:
                    items[k]["done"] = True
            with self.lock:
                for k in open_:
                    it = items[k]
                    append(self.out / "attempts.jsonl", {"id": it["row"]["id"], "batch": index, "attempt": attempt,
                                                         "time_utc": datetime.now(timezone.utc).isoformat(),
                                                         **{x: it["final"][x] for x in ("model", "rewrite_with_markers",
                                                                                      "target", "issues", "repeated",
                                                                                      "meaning_check", "meaning_check2")
                                                            if x in it["final"]},
                                                         "feedback": it["note"] if not it["done"] else None})
        records = []
        for k, it in items.items():
            f, row = it["final"], it["row"]
            if f is None or any(i.startswith("request_failed") for i in f["issues"]):
                continue   # not recorded, so a rerun tries the passage again
            record = {**{x: v for x, v in row.items() if x not in ("repair_prev", "repair_note")}, "ste": f["target"], "attempts": f["attempt"], "issues": f["issues"],
                      "review_status": "teacher_candidate_owner_unverified", "teacher_model": f.get("model", model),
                      "judge_model": a.judge, "source_hints": it["hints"], "gates_only": it["gates_only"],
                      "meaning_check": f.get("meaning_check"),
                      **({"judge2_model": a.judge2, "meaning_check2": f.get("meaning_check2")} if a.judge2 else {}),
                      "review_flags": {"dropped_words": dropped_words(row["text"], f["target"], lex, it["names"])
                                       if f["target"] else []}}
            records.append(record)
        with self.lock:
            for r in records:
                append(self.out / ("accepted.jsonl" if not r["issues"] else "rejected.jsonl"), r)
        return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--inputs", default=str(ROOT / "data/out/doc-pool-v5/pool.jsonl"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--rewriter", required=True, help="OpenRouter model id(s), comma-separated")
    parser.add_argument("--rewriter-reasoning", default="off")
    parser.add_argument("--judge", required=True, help="OpenRouter model id, or none")
    parser.add_argument("--judge-reasoning", default="off")
    parser.add_argument("--judge2", default="", help="a second judge: a rewrite passes only when both find no difference")
    parser.add_argument("--judge2-reasoning", default="low")
    parser.add_argument("--judge2-batch", type=int, default=12, help="checks a request (Cerebras stops a reply at 8,192 tokens)")
    parser.add_argument("--variant", default="traps")
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=1.0, help="US dollars for this invocation")
    parser.add_argument("--skip-runs", default="", help="other run directories whose recorded passages to skip")
    parser.add_argument("--model-gate", default="", help="a model vocabulary directory: rewrites must encode with it")
    args = parser.parse_args()
    run = Run(args)
    rows = [json.loads(l) for l in Path(args.inputs).read_text().splitlines()][args.offset:args.offset + args.limit]
    if any(r["family"] == "rust_book" for r in rows):
        raise ValueError("the Rust Book is the untouched holdout")
    dirs = [run.out] + [BASE / d for d in args.skip_runs.split(",") if d]
    seen = {json.loads(l)["id"] for d in dirs for n in ("accepted.jsonl", "rejected.jsonl") if (d / n).exists()
            for l in (d / n).read_text().splitlines()}
    todo = [r for r in rows if r["id"] not in seen]
    chunks = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    t0 = time.time()
    with ThreadPoolExecutor(args.parallel) as pool:
        for records in pool.map(lambda ic: run.batch(*ic), enumerate(chunks)):
            print(json.dumps({"batch": len(records), "accepted": sum(not r["issues"] for r in records),
                              "spent": round(run.cost, 4)}), flush=True)
    load = lambda n: [json.loads(l) for l in (run.out / n).read_text().splitlines()] if (run.out / n).exists() else []
    accepted, rejected = load("accepted.jsonl"), load("rejected.jsonl")
    summary = {"schema": "asd-ste100-doc-teacher-v5-pipeline", "run": args.run, "rewriter": run.rewriters,
               "judge": args.judge, "judge2": args.judge2 or None, "model_gate": args.model_gate or None, "variant": args.variant, "batch": args.batch, "attempts": args.attempts,
               "passages_this_run": len(todo), "wall_s": round(time.time() - t0, 1), "cost_usd_this_run": round(run.cost, 4),
               "stats": dict(run.stats), "accepted": len(accepted), "rejected": len(rejected),
               "gates_only": sum(1 for r in accepted + rejected if r["gates_only"]),
               "issues": dict(Counter(i.split(":")[0] for r in rejected for i in r["issues"])),
               "accepted_by_teacher": dict(Counter(r["teacher_model"] for r in accepted)),
               "rewrite_prompt_sha256": sha(run.rewrite_system.encode()), "judge_prompt_sha256": sha(run.judge_system.encode()),
               "code_sha256": {n: sha(Path(__file__).with_name(n).read_bytes())
                               for n in ("pipeline_v5.py", "openrouter_v5.py", "ste_context_v5.py", "gates_v4.py",
                                         "ste_context_v4.py")}}
    (run.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("passages_this_run", "accepted", "rejected", "cost_usd_this_run", "wall_s")}),
          flush=True)


if __name__ == "__main__":
    main()
