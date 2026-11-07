"""v2 teacher pilot: full STE context, per-passage checker hints, word-level retries.

Inputs are the frozen v1 teacher inputs, in the same order. The long system
prompt is evaluated once and saved as a llama-server slot; every request
restores that slot, so only the passage and its retries are evaluated again.
Outputs are review candidates only, never training-ready or owner-verified gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "data/out/doc-teacher-pilot-v1"
OUT = ROOT / "data/out/doc-teacher-pilot-v2"
SLOT = "ste-prefix.bin"
sys.path[:0] = [str(Path(__file__).parent)]
from gates_v2 import verify_v2  # noqa: E402
from ste_context import build_system_prompt, context_manifest, feedback, source_hints  # noqa: E402
from synthesize import MODEL, append, mask_literals, parse_reply, restore, sha  # noqa: E402
from checker import Lexicon  # noqa: E402
from gen import proper_names  # noqa: E402

CONFIG = {"temperature": 0.0, "seed": 42, "max_tokens": 320, "max_attempts": 3}
USER_MARK = "<|im_start|>user"
# The template renders the system turn differently with thinking on, so the
# cached prefix must be rendered with the same options as every request.
TEMPLATE_KWARGS = {"enable_thinking": False}


def post(url: str, path: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def system_prefix(rendered: str) -> str:
    # The chat template text up to the first user turn: what the slot caches.
    if not rendered.startswith("<|im_start|>system") or USER_MARK not in rendered:
        raise ValueError("unexpected chat template layout")
    return rendered[:rendered.index(USER_MARK)]


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


def chat(url: str, messages: list[dict], timeout: int) -> dict:
    restored = post(url, "/slots/0?action=restore", {"filename": SLOT}, 600)
    body = {"model": "Bonsai2-PQ2_0", "messages": messages, "id_slot": 0, "cache_prompt": True,
            "temperature": CONFIG["temperature"], "seed": CONFIG["seed"], "max_tokens": CONFIG["max_tokens"],
            "chat_template_kwargs": TEMPLATE_KWARGS}
    result = post(url, "/v1/chat/completions", body, timeout)
    return {"request_sha256": sha(json.dumps(body, sort_keys=True).encode()), "restored": restored,
            "body": result, "content": result["choices"][0]["message"].get("content") or ""}


def first_rows(limit: int) -> list[dict]:
    rows = [json.loads(line) for line in (V1 / "teacher_inputs.jsonl").open()]
    by_family = {f: [r for r in rows if r["family"] == f] for f in ("curl_cli", "postgres_docs", "python_docs")}
    return [r for position in range(12) for f in by_family for r in by_family[f][position:position + 1]][:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--warm-timeout", type=int, default=7200)
    args = parser.parse_args()
    if not 1 <= args.limit <= 36:
        raise ValueError("pilot limit must be 1..36")
    plan = json.loads((V1 / "synthesis-plan.json").read_text())
    if plan["splits"]["untouched_holdout"] != ["rust_book"]:
        raise ValueError("holdout split changed")
    (OUT / "slots").mkdir(parents=True, exist_ok=True)
    prompt = build_system_prompt()
    (OUT / "system-prompt.txt").write_text(prompt + "\n")
    manifest = context_manifest(prompt)
    (OUT / "context-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    slot = prepare_slot(args.url, prompt, args.warm_timeout)
    print(json.dumps({"slot_ready": True, "prefix_tokens": slot.get("prefix_tokens")}), flush=True)

    seen = set()
    for name in ("accepted.jsonl", "rejected.jsonl"):
        if (OUT / name).exists():
            seen |= {json.loads(line)["id"] for line in (OUT / name).open()}
    lex = Lexicon()
    for row in first_rows(args.limit):
        if row["id"] in seen:
            continue
        marked, literals = mask_literals(row["text"])
        names = proper_names(row["text"])
        hints = source_hints(row["text"], row["mode"], lex, names)
        user = (f"Mode: {row['mode']}\nDocument family: {row['family']}\nDocument path: {row['doc_path']}\n\n"
                f"Checker findings on this passage:\n{hints}\n\nPassage:\n{marked}\n\n"
                "Rewrite the passage in STE. Keep the meaning exactly. Reply with the JSON object only.")
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user}]
        final = None
        for attempt in range(1, CONFIG["max_attempts"] + 1):
            stamp = datetime.now(timezone.utc).isoformat()
            response, target, note = None, "", None
            try:
                response = chat(args.url, messages, args.timeout)
                rewrite, uncertain = parse_reply(response["content"])
                target = restore(rewrite, literals)
                issues = verify_v2(row["text"], target, row["mode"], lex, uncertain, names)
            except Exception as exc:
                issues = [type(exc).__name__ + ": " + str(exc)]
            if issues and target and attempt < CONFIG["max_attempts"]:
                note = feedback(issues, target, row["mode"], lex, names)
            usage = response["body"].get("usage") if response else None
            trace = {"id": row["id"], "attempt": attempt, "time_utc": stamp,
                     "request_sha256": response["request_sha256"] if response else None,
                     "raw_reply": response["content"] if response else None, "usage": usage,
                     "timings": response["body"].get("timings") if response else None,
                     "target": target or None, "issues": issues, "feedback": note}
            append(OUT / "attempts.jsonl", trace)
            final = trace
            cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
            if attempt == 1 and slot.get("prefix_tokens") and cached is not None and cached < 0.9 * slot["prefix_tokens"]:
                raise RuntimeError(f"slot prefix was not reused: {cached} cached of {slot['prefix_tokens']}")
            if not issues or not note:
                break
            messages += [{"role": "assistant", "content": response["content"]},
                         {"role": "user", "content": "The candidate failed these checks:\n" + note +
                          "\nFix every point. Keep the full meaning and every literal marker. "
                          "If you cannot, copy the input and set uncertain to true. Reply with the JSON object only."}]
        record = {**row, "ste": final["target"], "attempts": final["attempt"],
                  "review_status": "teacher_candidate_owner_unverified", "teacher_model": MODEL,
                  "issues": final["issues"], "source_hints": hints}
        append(OUT / ("accepted.jsonl" if not final["issues"] else "rejected.jsonl"), record)
        print(json.dumps({"family": row["family"], "id": row["id"][:12], "attempts": final["attempt"],
                          "accepted": not final["issues"], "issues": final["issues"]}), flush=True)

    def count(name):
        return sum(1 for _ in (OUT / name).open()) if (OUT / name).exists() else 0
    accepted = [json.loads(line) for line in (OUT / "accepted.jsonl").open()] if (OUT / "accepted.jsonl").exists() else []
    summary = {"schema": "asd-ste100-doc-teacher-pilot-v2", "model": MODEL, "config": CONFIG,
               "context": manifest, "slot": {k: slot.get(k) for k in ("prefix_sha256", "prefix_tokens")},
               "requested_limit": args.limit, "accepted": count("accepted.jsonl"), "rejected": count("rejected.jsonl"),
               "accepted_by_attempt": {str(n): sum(1 for r in accepted if r["attempts"] == n)
                                       for n in range(1, CONFIG["max_attempts"] + 1)},
               "inputs_sha256": sha((V1 / "teacher_inputs.jsonl").read_bytes()),
               "source_manifest_sha256": plan["source_manifest_sha256"],
               "code_sha256": {p.name: sha(p.read_bytes()) for p in
                               (Path(__file__), Path(__file__).with_name("gates_v2.py"), Path(__file__).with_name("ste_context.py"))},
               "limits": ["All accepted records are teacher candidates, not owner-verified meaning gold.",
                          "The v2 gates count meaning classes of modal, negation and condition words; they cannot certify semantics.",
                          "The 'should' policy (advice -> 'we recommend that you', requirement -> 'must') is the assistant default, pending owner review.",
                          "Held-out Rust family is never sent to the teacher."]}
    (OUT / "synthesis-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"accepted": summary["accepted"], "rejected": summary["rejected"]}), flush=True)


if __name__ == "__main__":
    main()
