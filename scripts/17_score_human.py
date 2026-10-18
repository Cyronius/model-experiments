"""Score the human read of v2 vs v1.5 against the hidden key, next to both LLM judges (2026-10-18)."""

import json
from collections import Counter
from pathlib import Path

HR = Path("data/eval/human_read")
key = json.loads((HR / "key.json").read_text(encoding="utf-8"))
votes = json.loads((HR / "human_votes.json").read_text(encoding="utf-8"))
skip = {n for n, v in votes.items() if v == "R"}  # rejected prompts leave the score


def verdicts(path: str) -> dict[str, str]:
    try:
        with Path(path).open(encoding="utf-8") as f:
            return {r["id"]: r["verdict"] for r in map(json.loads, f)}
    except FileNotFoundError:
        return {}


qwen, granite = verdicts("data/eval/judge_8Lb_vs_v15_qwen.jsonl"), verdicts("data/eval/judge_8Lb_vs_v15_granite.jsonl")
# judge dumps hold a = v2 (title file A); human A/B is randomized, so convert to "v2"/"v1.5"
to_side = {"a": "v2", "b": "v1.5", "tie": "tie"}


def human_side(n: str) -> str | None:
    v = votes.get(n)
    if v in (None, "R"):
        return None
    if v == "T":
        return "tie"
    if v == "X":
        return "both bad"
    return key[n]["a_is"] if v == "A" else ("v1.5" if key[n]["a_is"] == "v2" else "v2")


def tally(rows: list[tuple[str, str]], label: str) -> None:
    c = Counter(side for _, side in rows)
    decided = c["v2"] + c["v1.5"]
    share = f"{c['v2'] / decided:.1%} of {decided} decided" if decided else "n/a"
    print(f"{label:28} n={len(rows):3}  v2 {c['v2']:3}  v1.5 {c['v1.5']:3}  tie {c['tie']:3}  both-bad {c['both bad']:2}  -> v2 {share}")


human = [(n, s) for n in key if (s := human_side(n))]
tally(human, "human, all")
tally([(n, s) for n, s in human if key[n]["bucket"] != "hard"], "human, test prompts")
tally([(n, s) for n, s in human if key[n]["bucket"] == "hard"], "human, hard cases")

granite.update(verdicts("data/eval/judge_8Lb_hard_granite.jsonl"))
for name, j in (("Qwen3-14B", qwen), ("Granite 4.2 8B", granite)):
    rows = [(n, human_side(n), to_side[j[key[n]["id"]]]) for n in key if key[n]["id"] in j and human_side(n)]
    if rows:
        tally([(n, s) for n, _, s in rows], f"{name} on the same pairs")
        both_decided = [(h, s) for _, h, s in rows if h in ("v2", "v1.5") and s in ("v2", "v1.5")]
        agree = sum(h == s for h, s in both_decided)
        print(f"{'':28} agrees with human on {agree} of {len(both_decided)} pairs both decided")
print("answered:", len(votes), "of", len(key), "| missing:", sorted(set(key) - set(votes), key=int), "| rejected:", sorted(skip, key=int))
