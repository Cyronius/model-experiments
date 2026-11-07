"""v3 teacher pilot: the v2 method with the v2-review fixes and a meaning check.

Inputs are the frozen v1 teacher inputs, in the v2 order, so the first 12 are
the v2 passages. A rewrite that passes the deterministic gates goes to a
MEANING CHECK request (same model, same cached system prompt). Differences it
reports count as a failed check and go back to the teacher as feedback. A
rewrite with missing, repeated or reordered literal markers also gets a retry
(v1 and v2 stopped there). The first attempt is greedy. Retries sample at
`retry_temperature` with a fixed seed for each attempt: at temperature 0 the
teacher repeated a failed rewrite word for word in 5 of the 7 v2 passages that
needed a retry, and again on the first v3 passage, even with specific feedback.
Outputs are review candidates only, never training-ready or owner-verified gold.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "data/out/doc-teacher-pilot-v1"
OUT = ROOT / "data/out/doc-teacher-pilot-v3"
SLOT = "ste-prefix.bin"
sys.path[:0] = [str(Path(__file__).parent)]
from gates_v3 import dropped_words, verify_v3  # noqa: E402
from ste_context_v3 import (build_system_prompt_v3, context_manifest_v3, feedback_v3, judge_message,  # noqa: E402
                            marker_feedback, parse_judge, source_hints_v3)
from synthesize import MODEL, SLOT as MARKER, append, mask_literals, parse_reply, restore, sha  # noqa: E402
from synthesize_v2 import TEMPLATE_KWARGS, first_rows, post, system_prefix  # noqa: E402
from checker import Lexicon  # noqa: E402
from gen import proper_names  # noqa: E402

CONFIG = {"temperature": 0.0, "retry_temperature": 0.7, "seed": 42, "max_tokens": 320, "judge_max_tokens": 256,
          "max_attempts": 3}
CODE = ("synthesize_v3.py", "gates_v3.py", "ste_context_v3.py", "gates_v2.py", "ste_context.py")


def prepare_slot(url: str, prompt: str, timeout: int) -> dict:
    meta_path = OUT / "slots/ste-prefix.json"
    prompt_sha = sha(prompt.encode())
    if (OUT / "slots" / SLOT).exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("system_prompt_sha256") == prompt_sha and meta.get("model_sha256") == MODEL["sha256"]:
            return meta
    renders = [post(url, "/apply-template", {"messages": [{"role": "system", "content": prompt},
                                                          {"role": "user", "content": probe}],
                                          "chat_template_kwargs": TEMPLATE_KWARGS}, 60)["prompt"]
               for probe in ("first probe", "second probe")]
    prefix = system_prefix(renders[0])
    if system_prefix(renders[1]) != prefix:
        raise ValueError("system prefix depends on the user turn")
    warm = post(url, "/completion", {"prompt": prefix, "n_predict": 0, "cache_prompt": True, "id_slot": 0}, timeout)
    saved = post(url, "/slots/0?action=save", {"filename": SLOT}, 600)
    meta = {"system_prompt_sha256": prompt_sha, "model_sha256": MODEL["sha256"],
            "prefix_sha256": sha(prefix.encode()), "prefix_tokens": warm.get("tokens_evaluated"),
            "warm_timings": warm.get("timings"), "save": saved}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def chat(url: str, messages: list[dict], timeout: int, max_tokens: int, attempt: int = 1) -> dict:
    restored = post(url, "/slots/0?action=restore", {"filename": SLOT}, 600)
    body = {"model": "Bonsai2-PQ2_0", "messages": messages, "id_slot": 0, "cache_prompt": True,
            "temperature": CONFIG["temperature"] if attempt == 1 else CONFIG["retry_temperature"],
            "seed": CONFIG["seed"] + attempt - 1, "max_tokens": max_tokens, "chat_template_kwargs": TEMPLATE_KWARGS}
    result = post(url, "/v1/chat/completions", body, timeout)
    return {"request_sha256": sha(json.dumps(body, sort_keys=True).encode()), "restored": restored,
            "body": result, "content": result["choices"][0]["message"].get("content") or ""}


def cached(response: dict | None) -> int | None:
    usage = (response or {}).get("body", {}).get("usage") or {}
    return (usage.get("prompt_tokens_details") or {}).get("cached_tokens")


def meaning_check(url: str, prompt: str, row: dict, target: str, lex, names, timeout: int) -> tuple[dict, list[str]]:
    message = judge_message(row["text"], target, row["mode"], lex, names)
    response = chat(url, [{"role": "system", "content": prompt}, {"role": "user", "content": message}],
                    timeout, CONFIG["judge_max_tokens"])
    trace = {"request_sha256": response["request_sha256"], "message": message, "raw_reply": response["content"],
             "usage": response["body"].get("usage"), "timings": response["body"].get("timings")}
    try:
        differences = parse_judge(response["content"])
    except Exception as exc:
        trace["error"] = type(exc).__name__ + ": " + str(exc)
        return trace, ["meaning_check_unparsed"]
    trace["differences"] = differences
    return trace, [f"meaning_check:{len(differences)}"] if differences else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--limit", type=int, default=36)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--warm-timeout", type=int, default=7200)
    args = parser.parse_args()
    if not 1 <= args.limit <= 36:
        raise ValueError("pilot limit must be 1..36")
    plan = json.loads((V1 / "synthesis-plan.json").read_text())
    if plan["splits"]["untouched_holdout"] != ["rust_book"]:
        raise ValueError("holdout split changed")
    (OUT / "slots").mkdir(parents=True, exist_ok=True)
    lex = Lexicon()
    prompt = build_system_prompt_v3(lex)
    (OUT / "system-prompt.txt").write_text(prompt + "\n")
    manifest = context_manifest_v3(prompt)
    (OUT / "context-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    slot = prepare_slot(args.url, prompt, args.warm_timeout)
    print(json.dumps({"slot_ready": True, "prefix_tokens": slot.get("prefix_tokens")}), flush=True)

    seen = set()
    for name in ("accepted.jsonl", "rejected.jsonl"):
        if (OUT / name).exists():
            seen |= {json.loads(line)["id"] for line in (OUT / name).read_text().splitlines()}
    for row in first_rows(args.limit):
        if row["id"] in seen:
            continue
        marked, literals = mask_literals(row["text"])
        names = proper_names(row["text"])
        hints = source_hints_v3(row["text"], row["mode"], lex, names)
        user = (f"Mode: {row['mode']}\nDocument family: {row['family']}\nDocument path: {row['doc_path']}\n\n"
                f"Checker findings on this passage:\n{hints}\n\nPassage:\n{marked}\n\n"
                "Rewrite the passage in STE. Keep the meaning exactly. Reply with the JSON object only.")
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user}]
        final, earlier = None, []
        for attempt in range(1, CONFIG["max_attempts"] + 1):
            stamp = datetime.now(timezone.utc).isoformat()
            response, rewrite, target, note, judge, differences, markers = None, "", "", None, None, [], None
            try:
                response = chat(args.url, messages, args.timeout, CONFIG["max_tokens"], attempt)
                rewrite, uncertain = parse_reply(response["content"])
                if MARKER.findall(rewrite) != [f"[[L{i}]]" for i in range(len(literals))]:
                    issues, markers = ["marker_change"], marker_feedback(rewrite, len(literals))
                else:
                    target = restore(rewrite, literals)
                    issues = verify_v3(row["text"], target, row["mode"], lex, uncertain, names)
                    if not issues:
                        judge, issues = meaning_check(args.url, prompt, row, target, lex, names, args.timeout)
                        differences = judge.get("differences", [])
            except Exception as exc:
                issues = [type(exc).__name__ + ": " + str(exc)]
            repeated = bool(rewrite) and rewrite in earlier
            earlier.append(rewrite)
            if markers and attempt < CONFIG["max_attempts"]:
                note = "- " + markers + ("\n- Your rewrite is the same as an earlier attempt that failed these checks. "
                                         "Write a different rewrite." if repeated else "")
            elif issues and target and attempt < CONFIG["max_attempts"] and issues != ["meaning_check_unparsed"]:
                note = feedback_v3(issues, row["text"], target, row["mode"], lex, names, differences, repeated)
            trace = {"id": row["id"], "attempt": attempt, "time_utc": stamp,
                     "request_sha256": response["request_sha256"] if response else None,
                     "raw_reply": response["content"] if response else None,
                     "usage": response["body"].get("usage") if response else None,
                     "timings": response["body"].get("timings") if response else None,
                     "rewrite_with_markers": rewrite or None, "target": target or None, "issues": issues,
                     "repeated": repeated,
                     "dropped_words": dropped_words(row["text"], target, lex, names) if target else None,
                     "meaning_check": judge, "feedback": note}
            append(OUT / "attempts.jsonl", trace)
            final = trace
            hit = cached(response)
            if attempt == 1 and slot.get("prefix_tokens") and hit is not None and hit < 0.9 * slot["prefix_tokens"]:
                raise RuntimeError(f"slot prefix was not reused: {hit} cached of {slot['prefix_tokens']}")
            if not issues or not note:
                break
            messages += [{"role": "assistant", "content": response["content"]},
                         {"role": "user", "content": "The candidate failed these checks:\n" + note +
                          "\nFix every point. Keep the full meaning and every literal marker. "
                          "If you cannot, copy the input and set uncertain to true. Reply with the JSON object only."}]
        record = {**row, "ste": final["target"], "attempts": final["attempt"],
                  "review_status": "teacher_candidate_owner_unverified", "teacher_model": MODEL,
                  "issues": final["issues"], "source_hints": hints,
                  "meaning_check": (final["meaning_check"] or {}).get("differences"),
                  "review_flags": {"dropped_words": final["dropped_words"] or []}}
        append(OUT / ("accepted.jsonl" if not final["issues"] else "rejected.jsonl"), record)
        print(json.dumps({"family": row["family"], "id": row["id"][:12], "attempts": final["attempt"],
                          "accepted": not final["issues"], "issues": final["issues"]}), flush=True)

    def rows(name):
        return [json.loads(line) for line in (OUT / name).read_text().splitlines()] if (OUT / name).exists() else []
    accepted, traces = rows("accepted.jsonl"), rows("attempts.jsonl")
    checks = [t["meaning_check"] for t in traces if t.get("meaning_check")]
    summary = {"schema": "asd-ste100-doc-teacher-pilot-v3", "model": MODEL, "config": CONFIG,
               "context": manifest, "slot": {k: slot.get(k) for k in ("prefix_sha256", "prefix_tokens")},
               "requested_limit": args.limit, "accepted": len(accepted), "rejected": len(rows("rejected.jsonl")),
               "accepted_by_attempt": {str(n): sum(1 for r in accepted if r["attempts"] == n)
                                       for n in range(1, CONFIG["max_attempts"] + 1)},
               "meaning_checks": {"requests": len(checks),
                                  "found_differences": sum(1 for c in checks if c.get("differences")),
                                  "unparsed": sum(1 for c in checks if "error" in c)},
               "repeated_attempts": sum(1 for t in traces if t.get("repeated")),
               "inputs_sha256": sha((V1 / "teacher_inputs.jsonl").read_bytes()),
               "source_manifest_sha256": plan["source_manifest_sha256"],
               "code_sha256": {name: sha(Path(__file__).with_name(name).read_bytes()) for name in CODE},
               "limits": ["All accepted records are teacher candidates, not owner-verified meaning gold.",
                          "The gates count meaning classes of modal, negation and condition words; they cannot certify semantics.",
                          "The meaning check is the same model as the teacher: a second look, not an independent judge.",
                          "The 'should' policy (advice -> 'we recommend that you', requirement -> 'must') is the assistant default, pending owner review.",
                          "The v3 fixes were designed on the first 12 inputs; the other 24 measure how they generalize.",
                          "Held-out Rust family is never sent to the teacher."]}
    (OUT / "synthesis-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"accepted": summary["accepted"], "rejected": summary["rejected"]}), flush=True)


if __name__ == "__main__":
    main()
