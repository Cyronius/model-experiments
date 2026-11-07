"""A model on the NPU as an independent meaning check, scored on the reviewed v3 rewrites.

The teacher's meaning check is Bonsai 2 judging its own rewrites. This runs the
same check with a different model, served by `oflm serve` (Granite 4.2 3B from
an OpenFlowLM-Next build of PR #209, K2-Horizon-7B from the main build, each
with its Uno drafter), on every v3 rewrite that has an assistant verdict: the
first rewrite of each passage that passed the gates, and the final rewrite
when the meaning check made it differ. Two prompt arms:
- `v3`: the MEANING CHECK text and examples that Bonsai used in v3, so the
  result compares directly with Bonsai's v3 self-check on the same rewrites;
- `v4`: the v4 MEANING CHECK. Its "do not report" list was written from the v3
  false alarms, so on these rewrites it is tuned, not tested.
The judge does not see the 28k-token STE word lists, only a short STE note.
Requests are greedy (top_k 1, no penalty, no thinking) by default, so the Uno
drafter applies and the output equals plain greedy decoding. K2 always reasons
first; `--reasoning-effort low` asks for its shortest reasoning, and the reply
content after the reasoning is what is parsed. Greedy K2 reasoning repeats the
same lines until the token budget ends, so `--repetition-penalty` (which turns
speculation off) is there for K2. A reply that is not valid JSON
but lists differences counts as a flag, and the summary counts those replies.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

from ste_context_v3 import _judge_section, judge_message, parse_judge
from ste_context_v4 import _judge_section_v4, judge_message_v4
from checker import Lexicon
from gen import proper_names
from synthesize import sha

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "data/out/doc-teacher-pilot-v1"
V3 = ROOT / "data/out/doc-teacher-pilot-v3"
STE_NOTE = """The rewrites are in ASD-STE100 Simplified Technical English (STE). STE uses only approved words, short sentences, the active voice and simple tenses. It writes modal words by sense ("MODAL WORDS"): may or might -> can or possibly; should (advice) -> "we recommend that you"; should (a requirement) -> must; whether -> if; once -> when; never -> do not; except -> but."""
ARMS = {"v3": (_judge_section, judge_message), "v4": (_judge_section_v4, judge_message_v4)}


def cases() -> list[dict]:
    """Every v3 rewrite with an assistant verdict, with Bonsai's own meaning check on it."""
    inputs = {json.loads(l)["id"]: json.loads(l) for l in (V1 / "teacher_inputs.jsonl").read_text().splitlines()}
    attempts = [json.loads(l) for l in (V3 / "attempts.jsonl").read_text().splitlines()]
    review = json.loads((V3 / "all-36-assistant-review.json").read_text())["pairs"]
    out = []
    for pair in review:
        tries = {t["attempt"]: t for t in attempts if t["id"].startswith(pair["id"])}
        for arm in ("gates_only", "with_meaning_check"):
            verdict = pair[arm]
            if not verdict["accepted"] or any(c["id"] == pair["id"] and c["attempt"] == verdict["attempt"] for c in out):
                continue
            trace = tries[verdict["attempt"]]
            row = inputs[trace["id"]]
            out.append({"n": pair["n"], "id": pair["id"], "attempt": verdict["attempt"], "family": row["family"],
                        "mode": row["mode"], "source": row["text"], "target": trace["target"],
                        "label": verdict["verdict"], "bonsai_differences": (trace["meaning_check"] or {}).get("differences")})
    return out


GREEDY = {"temperature": 0.0, "top_k": 1, "repetition_penalty": 1.0}


def ask(url: str, model: str, system: str, user: str, max_tokens: int, timeout: int, effort: str | None = None,
        sampling: dict | None = None) -> dict:
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **(sampling or GREEDY), "think": False, "max_tokens": max_tokens, "stream": False}
    if effort:
        body["reasoning_effort"] = effort
    request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    message = result["choices"][0]["message"]
    return {"seconds": time.monotonic() - start, "request_sha256": sha(json.dumps(body, sort_keys=True).encode()),
            "content": message.get("content") or "", "reasoning": message.get("reasoning_content"),
            "finish_reason": result["choices"][0].get("finish_reason"), "usage": result.get("usage")}


def flagged(row: dict, key: str) -> bool:
    """Differences found; a malformed reply counts when it is not an empty list."""
    if row[key] is not None:
        return bool(row[key])
    return "[]" not in row["raw_reply"].replace(" ", "") and "differences" in row["raw_reply"]


def score(rows: list[dict], key: str) -> dict:
    changed = [r for r in rows if r["label"] == "repair"]
    kept = [r for r in rows if r["label"] == "kept"]
    return {"meaning_changes_flagged": sum(1 for r in changed if flagged(r, key)), "of_meaning_changes": len(changed),
            "good_rewrites_flagged": sum(1 for r in kept if flagged(r, key)), "of_good_rewrites": len(kept),
            "missed_meaning_changes_n": [r["n"] for r in changed if not flagged(r, key)],
            "unparsed": sum(1 for r in rows if r[key] is None)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="granite4.2:3b")
    parser.add_argument("--out", type=Path, default=ROOT / "data/out/doc-judge-granite-v1")
    parser.add_argument("--arms", default="v3,v4")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--reasoning-effort", default=None, help="K2: low, medium or high")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="above 1 stops greedy reasoning loops; it turns speculative decoding off")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    lex = Lexicon()
    data = cases()
    summary = {"schema": "asd-ste100-doc-judge-npu-v1", "model": args.model, "reasoning_effort": args.reasoning_effort,
               "max_tokens": args.max_tokens, "sampling": {**GREEDY, "repetition_penalty": args.repetition_penalty},
               "cases": len(data),
               "labels": dict(Counter(c["label"] for c in data)), "bonsai_v3_self_check": score(data, "bonsai_differences"),
               "arms": {}}
    for arm in args.arms.split(","):
        section, message = ARMS[arm]
        system = STE_NOTE + "\n\n" + section(lex)
        (out / f"system-prompt-{arm}.txt").write_text(system + "\n")
        rows = []
        with (out / f"judgements-{arm}.jsonl").open("w", encoding="utf-8") as f:
            for case in data:
                user = message(case["source"], case["target"], case["mode"], lex, proper_names(case["source"]))
                reply = ask(args.url, args.model, system, user, args.max_tokens, args.timeout, args.reasoning_effort,
                            {**GREEDY, "repetition_penalty": args.repetition_penalty})
                try:
                    differences, error = parse_judge(reply["content"]), None
                except Exception as exc:
                    differences, error = None, type(exc).__name__ + ": " + str(exc)
                row = {**case, "arm": arm, "message": user, "raw_reply": reply["content"], "reasoning": reply["reasoning"],
                       "finish_reason": reply["finish_reason"], "judge_differences": differences,
                       "error": error, "seconds": reply["seconds"], "usage": reply["usage"],
                       "request_sha256": reply["request_sha256"]}
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
        summary["arms"][arm] = {"system_prompt_sha256": sha(system.encode()), **score(rows, "judge_differences"),
                                "seconds_total": round(sum(r["seconds"] for r in rows), 1),
                                "seconds_mean": round(sum(r["seconds"] for r in rows) / len(rows), 2)}
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
