"""v5 teacher speed runs: the v4 method with parallel slots, a shorter prompt and other models.

The gates, hints, feedback, attempts and meaning check are the v4 ones. What changes:
- `--parallel N` runs N passages at once, one llama-server slot each. Before each
  request the worker restores the cached system prefix into its own slot (the v4
  slot mechanism, per slot). Batched slots can change the floating-point results,
  so a rewrite can differ from a one-slot run of the same model.
- `--variant` picks the system prompt (ste_context_v5): the v4 prompt, or a shorter
  one with its own meaning-check prefix.
- `--profile` picks the request settings of a model (chat template switches, token
  limits). The served model is recorded by its file size and the server's report.
Outputs go to data/out/doc-teacher-v5/<run>/ with timings for a passages-per-hour
figure. Outputs are review candidates only, never training-ready or owner-verified gold.
"""

from __future__ import annotations

import sys
from pathlib import Path

# data/doc_pilot/queue.py shadows the standard library's queue, which the thread pool needs.
_path, sys.path = sys.path, [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
import queue  # noqa: E402,F401
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
sys.path = _path

import argparse
import json
import threading
import time
from collections import Counter
from datetime import datetime, timezone

sys.path[:0] = [str(Path(__file__).parent)]
from gates_v4 import marker_problems, restore_any_order, verify_v4  # noqa: E402
from gates_v3 import dropped_words  # noqa: E402
from ste_context_v4 import feedback_v4, judge_message_v4, marker_feedback_v4, source_hints_v4  # noqa: E402
from ste_context_v5 import VARIANTS, judge_prompt, rewrite_prompt  # noqa: E402
from ste_context_v3 import parse_judge  # noqa: E402
from synthesize import append, mask_literals, parse_reply, sha  # noqa: E402
from synthesize_v2 import post  # noqa: E402
from checker import Lexicon  # noqa: E402
from gen import proper_names  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "data/out/doc-teacher-pilot-v4"
BASE = ROOT / "data/out/doc-teacher-v5"
SLOTS = BASE / "slots"   # the server's --slot-save-path
CONFIG = {"temperature": 0.0, "retry_temperature": 0.7, "seed": 42, "max_attempts": 3}
PROFILES = {
    "bonsai": {"kwargs": {"enable_thinking": False}, "max_tokens": 320, "judge_max_tokens": 256},
    # Qwen3.6 writes long differences: 256 tokens cut 7 of 45 checks short (unparsed) in qwen36-core-p8.
    "qwen36": {"kwargs": {"enable_thinking": False}, "max_tokens": 320, "judge_max_tokens": 768},
    "gptoss": {"kwargs": {"reasoning_effort": "low"}, "max_tokens": 1536, "judge_max_tokens": 1536},
}
TURN_MARKS = ("<|im_start|>", "<|start|>")


def system_prefix(a: str, b: str) -> str:
    """The rendered text before the first user turn, from two renders that differ only in the user turn."""
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    cut = max(a.rfind(mark, 0, n) for mark in TURN_MARKS)
    if cut <= 0 or not any(a.startswith(mark + "user", cut) for mark in TURN_MARKS):
        raise ValueError("unexpected chat template layout")
    return a[:cut]


def prepare_prefix(url: str, prompt: str, profile: dict, model_id: str, timeout: int) -> dict:
    """Warm slot 0 with the system prefix once and save it; any slot can restore the file later."""
    # A slot file loads only into a server with the same KV stream count, so the servers run with
    # --kv-unified (one stream for every slot count).
    key = sha((f"{model_id}\nkv-unified\n" + json.dumps(profile["kwargs"], sort_keys=True) + "\n" + prompt).encode())[:20]
    meta_path, name = SLOTS / f"{key}.json", f"{key}.bin"
    if meta_path.exists() and (SLOTS / name).exists():
        return json.loads(meta_path.read_text())
    renders = [post(url, "/apply-template", {"messages": [{"role": "system", "content": prompt},
                                                          {"role": "user", "content": probe}],
                                          "chat_template_kwargs": profile["kwargs"]}, 60)["prompt"]
               for probe in ("first probe", "second probe")]
    prefix = system_prefix(*renders)
    warm = post(url, "/completion", {"prompt": prefix, "n_predict": 0, "cache_prompt": True, "id_slot": 0}, timeout)
    saved = post(url, "/slots/0?action=save", {"filename": name}, 600)
    meta = {"file": name, "model": model_id, "system_prompt_sha256": sha(prompt.encode()),
            "prefix_sha256": sha(prefix.encode()), "prefix_tokens": warm.get("tokens_evaluated"),
            "warm_timings": warm.get("timings"), "save": saved}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


class Teacher:
    def __init__(self, args, lex):
        self.url, self.args, self.lex = args.url, args, lex
        self.profile = PROFILES[args.profile]
        self.model_id = args.model_id
        self.rewrite_system = rewrite_prompt(lex, args.variant)
        self.judge_system = judge_prompt(lex, args.variant)
        if args.backend == "llama":
            self.rewrite_prefix = prepare_prefix(self.url, self.rewrite_system, self.profile, self.model_id,
                                                 args.warm_timeout)
            self.judge_prefix = (self.rewrite_prefix if self.judge_system == self.rewrite_system else
                                 prepare_prefix(self.url, self.judge_system, self.profile, self.model_id, args.warm_timeout))
        else:   # an OpenAI-compatible server (vLLM) caches shared prefixes by itself
            self.rewrite_prefix = self.judge_prefix = {}
        self.lock = threading.Lock()
        self.free = list(range(args.parallel))
        self.stats = Counter()

    def chat(self, slot: int, prefix: dict, messages: list[dict], max_tokens: int, attempt: int) -> dict:
        t0 = time.time()
        llama = self.args.backend == "llama"
        restored = post(self.url, f"/slots/{slot}?action=restore", {"filename": prefix["file"]}, 600) if llama else None
        t1 = time.time()
        body = {"model": self.args.served_model, "messages": messages,
                "temperature": CONFIG["temperature"] if attempt == 1 else CONFIG["retry_temperature"],
                "seed": CONFIG["seed"] + attempt - 1, "max_tokens": max_tokens,
                "chat_template_kwargs": self.profile["kwargs"], **({"id_slot": slot, "cache_prompt": True} if llama else {})}
        result = post(self.url, "/v1/chat/completions", body, self.args.timeout)
        timings = result.get("timings") or {}
        if not llama:
            usage = result.get("usage") or {}
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            timings = {"prompt_n": (usage.get("prompt_tokens") or 0) - cached, "predicted_n": usage.get("completion_tokens", 0),
                       "cache_n": cached}
        with self.lock:
            self.stats.update({"requests": 1, "restore_ms": int((t1 - t0) * 1000),
                               "request_ms": int((time.time() - t1) * 1000),
                               "prompt_tokens": timings.get("prompt_n", 0), "output_tokens": timings.get("predicted_n", 0),
                               "cached_tokens": timings.get("cache_n", 0)})
            if (timings.get("cache_n") or 0) < 0.9 * (prefix.get("prefix_tokens") or 0):
                self.stats["prefix_misses"] += 1
        message = result["choices"][0]["message"]
        return {"request_sha256": sha(json.dumps(body, sort_keys=True).encode()), "restored": restored,
                "body": result, "content": message.get("content") or "", "reasoning": message.get("reasoning_content")}

    def meaning_check(self, slot: int, row: dict, target: str, names) -> tuple[dict, list[str]]:
        message = judge_message_v4(row["text"], target, row["mode"], self.lex, names)
        response = self.chat(slot, self.judge_prefix, [{"role": "system", "content": self.judge_system},
                                                        {"role": "user", "content": message}],
                             self.profile["judge_max_tokens"], 1)
        trace = {"request_sha256": response["request_sha256"], "message": message, "raw_reply": response["content"],
                 "usage": response["body"].get("usage"), "timings": response["body"].get("timings")}
        try:
            differences = parse_judge(response["content"])
        except Exception as exc:
            trace["error"] = type(exc).__name__ + ": " + str(exc)
            return trace, ["meaning_check_unparsed"]
        trace["differences"] = differences
        return trace, [f"meaning_check:{len(differences)}"] if differences else []

    def passage(self, row: dict, out: Path) -> dict:
        with self.lock:
            slot = self.free.pop()
        try:
            return self._passage(slot, row, out)
        finally:
            with self.lock:
                self.free.append(slot)

    def _passage(self, slot: int, row: dict, out: Path) -> dict:
        lex = self.lex
        marked, literals = mask_literals(row["text"])
        names = proper_names(row["text"])
        hints = source_hints_v4(row["text"], row["mode"], lex, names)
        user = (f"Mode: {row['mode']}\nDocument family: {row['family']}\nDocument path: {row['doc_path']}\n\n"
                f"Checker findings on this passage:\n{hints}\n\nPassage:\n{marked}\n\n"
                "Rewrite the passage in STE. Keep the meaning exactly. Reply with the JSON object only.")
        messages = [{"role": "system", "content": self.rewrite_system}, {"role": "user", "content": user}]
        final, earlier, gates_only = None, [], None
        for attempt in range(1, CONFIG["max_attempts"] + 1):
            stamp = datetime.now(timezone.utc).isoformat()
            response, rewrite, target, note, judge, differences, markers = None, "", "", None, None, [], None
            try:
                response = self.chat(slot, self.rewrite_prefix, messages, self.profile["max_tokens"], attempt)
                rewrite, uncertain = parse_reply(response["content"])
                if marker_problems(rewrite, len(literals)):
                    issues, markers = ["marker_change"], marker_feedback_v4(rewrite, marked, len(literals))
                else:
                    target = restore_any_order(rewrite, literals)
                    issues = verify_v4(row["text"], target, row["mode"], lex, uncertain, names)
                    if not issues and not self.args.no_check:
                        judge, issues = self.meaning_check(slot, row, target, names)
                        differences = judge.get("differences", [])
            except Exception as exc:
                issues = [type(exc).__name__ + ": " + str(exc)]
            if target and gates_only is None and all(i.startswith("meaning_check") for i in issues):
                gates_only = {"attempt": attempt, "ste": target}
            repeated = bool(rewrite) and rewrite in earlier
            earlier.append(rewrite)
            if markers and attempt < CONFIG["max_attempts"]:
                note = "- " + markers + ("\n- Your rewrite is the same as an earlier attempt that failed these checks. "
                                         "Write a different rewrite." if repeated else "")
            elif issues and target and attempt < CONFIG["max_attempts"] and issues != ["meaning_check_unparsed"]:
                note = feedback_v4(issues, row["text"], target, row["mode"], lex, names, differences, repeated)
            trace = {"id": row["id"], "attempt": attempt, "time_utc": stamp, "slot": slot,
                     "request_sha256": response["request_sha256"] if response else None,
                     "raw_reply": response["content"] if response else None,
                     "reasoning": response["reasoning"] if response else None,
                     "usage": response["body"].get("usage") if response else None,
                     "timings": response["body"].get("timings") if response else None,
                     "rewrite_with_markers": rewrite or None, "target": target or None, "issues": issues,
                     "repeated": repeated,
                     "dropped_words": dropped_words(row["text"], target, lex, names) if target else None,
                     "meaning_check": judge, "feedback": note}
            with self.lock:
                append(out / "attempts.jsonl", trace)
            final = trace
            if not issues or not note:
                break
            messages += [{"role": "assistant", "content": response["content"]},
                         {"role": "user", "content": "The candidate failed these checks:\n" + note +
                          "\nFix every point. Keep the full meaning and every literal marker. "
                          "If you cannot, copy the input and set uncertain to true. Reply with the JSON object only."}]
        record = {**row, "ste": final["target"], "attempts": final["attempt"],
                  "review_status": "teacher_candidate_owner_unverified", "teacher_model": self.model_id,
                  "issues": final["issues"], "source_hints": hints, "gates_only": gates_only,
                  "meaning_check": (final["meaning_check"] or {}).get("differences"),
                  "review_flags": {"dropped_words": final["dropped_words"] or []}}
        with self.lock:
            append(out / ("accepted.jsonl" if not final["issues"] else "rejected.jsonl"), record)
        return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--model-id", required=True, help="the served model file and its size, for the records")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--backend", choices=("llama", "openai"), default="llama",
                        help="llama: llama-server slots restored per request; openai: a server with its own prefix cache")
    parser.add_argument("--served-model", default="teacher", help="the model name for an OpenAI-compatible server")
    parser.add_argument("--inputs", default=str(V4 / "teacher_inputs.jsonl"), help="JSONL passages (v4 input schema)")
    parser.add_argument("--limit", type=int, default=132)
    parser.add_argument("--no-check", action="store_true", help="gates only, no meaning check")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--warm-timeout", type=int, default=7200)
    args = parser.parse_args()
    out = BASE / args.run
    out.mkdir(parents=True, exist_ok=True)
    SLOTS.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in Path(args.inputs).read_text().splitlines()][:args.limit]
    plan_path = V4 / "synthesis-plan.json"   # absent on a pod, where the inputs come from build_pool_v5.py
    if plan_path.exists() and json.loads(plan_path.read_text())["splits"]["untouched_holdout"] != ["rust_book"]:
        raise ValueError("holdout split changed")
    if any(r["family"] == "rust_book" for r in rows):
        raise ValueError("the Rust Book is the untouched holdout")
    lex = Lexicon()
    teacher = Teacher(args, lex)
    (out / "system-prompt.txt").write_text(teacher.rewrite_system + "\n")
    (out / "judge-prompt.txt").write_text(teacher.judge_system + "\n")
    seen = {json.loads(l)["id"] for n in ("accepted.jsonl", "rejected.jsonl") if (out / n).exists()
            for l in (out / n).read_text().splitlines()}
    todo = [r for r in rows if r["id"] not in seen]
    t0 = time.time()
    with ThreadPoolExecutor(args.parallel) as pool:
        for record in pool.map(lambda r: teacher.passage(r, out), todo):
            print(json.dumps({"id": record["id"][:12], "attempts": record["attempts"], "accepted": not record["issues"],
                              "issues": record["issues"]}), flush=True)
    wall = time.time() - t0
    load = lambda n: [json.loads(l) for l in (out / n).read_text().splitlines()] if (out / n).exists() else []
    accepted, rejected, traces = load("accepted.jsonl"), load("rejected.jsonl"), load("attempts.jsonl")
    checks = [t["meaning_check"] for t in traces if t.get("meaning_check")]
    by_set = lambda records: dict(Counter(r.get("pilot_set", "all") for r in records))
    summary = {"schema": "asd-ste100-doc-teacher-v5", "run": args.run, "model_id": args.model_id,
               "profile": args.profile, "variant": args.variant, "parallel": args.parallel, "backend": args.backend,
               "config": CONFIG,
               "prefix_tokens": {"rewrite": teacher.rewrite_prefix.get("prefix_tokens"),
                                 "judge": teacher.judge_prefix.get("prefix_tokens")},
               "passages_this_run": len(todo), "wall_s": round(wall, 1),
               "passages_per_hour": round(len(todo) / wall * 3600, 1) if todo else None,
               "stats": dict(teacher.stats),
               "accepted": len(accepted), "rejected": len(rejected),
               "accepted_by_set": by_set(accepted),
               "gates_only_by_set": by_set([r for r in accepted + rejected if r["gates_only"]]),
               "first_attempt_gate_pass": sum(1 for r in accepted + rejected if (r["gates_only"] or {}).get("attempt") == 1),
               "meaning_checks": {"requests": len(checks),
                                  "found_differences": sum(1 for c in checks if c.get("differences")),
                                  "unparsed": sum(1 for c in checks if "error" in c)},
               "issues": dict(Counter(i.split(":")[0] for r in rejected for i in r["issues"])),
               "code_sha256": {n: sha(Path(__file__).with_name(n).read_bytes())
                               for n in ("synthesize_v5.py", "ste_context_v5.py", "gates_v4.py", "ste_context_v4.py")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("passages_per_hour", "accepted", "rejected", "gates_only_by_set")}), flush=True)


if __name__ == "__main__":
    main()
