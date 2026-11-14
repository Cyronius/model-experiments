"""Validate the 500M sampled smoke review and compare unchanged pilot ratings."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAG = "qwen-20261023"


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf-8-sig"))


base = read("qwen_sampling_20261023_ratings.json")
review = read("qwen_sampling_500m_20261023_ratings.json")
checkable = set(base["method"]["checkable_ids"])
assert len(checkable) == 31
all_ratings = {m: base["ratings"][m] for m in ("d200M", "d250M", "d300M", "stock08")}
all_ratings["d500M"] = review["ratings"]
summary = {}
for model, ratings in all_ratings.items():
    raw = [json.loads(s) for s in (HERE / f"answers_{model}-sampled_{TAG}.jsonl").read_text(encoding="utf-8").splitlines()]
    meta = read(f"metadata_{model}-sampled_{TAG}.json")
    assert len(raw) == len(ratings) == 40
    assert sorted(r["id"] for r in raw) == sorted(r[0] for r in ratings) == list(range(40))
    assert {r[0] for r in ratings if r[3] != "na"} == checkable
    assert all(type(r[1]) is int and 1 <= r[1] <= 5 and type(r[2]) is bool and r[3] in ("yes", "no", "na") and r[4] for r in ratings)
    assert meta["sampling"] == base["method"]["sampling"]
    assert meta["max_tokens"] == 400 and meta["thinking"] is False and meta["seed_offset"] == 0
    assert meta["slots"] == 4 and meta["context"] == 16384
    assert all(0 < r["tokens"] <= 400 and r["finish"] in ("stop", "length") for r in raw)
    correct = [r[0] for r in ratings if r[3] == "yes"]
    loops = [r[0] for r in ratings if r[2]]
    summary[model] = {
        "coherence_mean": sum(r[1] for r in ratings) / 40,
        "coherent_4plus": sum(r[1] >= 4 for r in ratings),
        "correct": len(correct), "checkable": 31, "accuracy": len(correct) / 31,
        "correct_ids": correct, "loops": len(loops), "loop_ids": loops,
        "loop_rate": len(loops) / 40,
        "token_limit_hits": sum(r["finish"] == "length" for r in raw),
        "total_generated_tokens": sum(r["tokens"] for r in raw),
    }
runtime = read("runtime_sampling_d500M_qwen-20261023.json")
assert len(runtime) == 4
for slot in runtime:
    params = slot["params"]
    assert all(params[k] == v for k, v in base["method"]["sampling"].items())
    assert params["repeat_last_n"] == 64 and params["max_tokens"] == 400
paired = {}
for model in ("d200M", "d250M", "d300M", "stock08"):
    previous, current = summary[model], summary["d500M"]
    paired[model] = {
        "newly_correct_ids": sorted(set(current["correct_ids"]) - set(previous["correct_ids"])),
        "lost_correct_ids": sorted(set(previous["correct_ids"]) - set(current["correct_ids"])),
        "coherence_delta": round(current["coherence_mean"] - previous["coherence_mean"], 4),
        "new_loop_ids": sorted(set(current["loop_ids"]) - set(previous["loop_ids"])),
        "resolved_loop_ids": sorted(set(previous["loop_ids"]) - set(current["loop_ids"])),
    }
result = {
    "method": base["method"] | {"extension": "500M final checkpoint; prior ratings unchanged; same primary assistant and rubric."},
    "summary": summary, "paired_500m_against": paired,
    "format_checks_500m": review["format_checks"],
    "borderline_judgments": review["borderline_judgments"],
}
(HERE / "qwen_sampling_500m_20261023_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"summary": summary, "paired_500m_against": paired}, indent=2))
