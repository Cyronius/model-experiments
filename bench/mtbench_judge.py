# MT-Bench judging in two steps: write blind judge batches for Claude Haiku subagents, then merge their scores into the run results (2026-10-19).
"""
Usage:
  python mtbench_judge.py batches --runid R [--models a,b]   # writes results/R_work/mtbench_judge/batch_NN.json + INSTRUCTIONS.md
  (have a Haiku subagent judge each batch_NN.json into batch_NN_out.json, following INSTRUCTIONS.md)
  python mtbench_judge.py merge --runid R                    # adds mtbench_haiku to each result JSON and rewrites summary.md

Scores use Haiku, not GPT-4, so they compare our models with each other, not with published MT-Bench numbers.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from run_bench import write_summary

BENCH = Path(__file__).parent
RESULTS = BENCH / "results"
BATCH = 20

INSTRUCTIONS = """# MT-Bench judging instructions

You are an impartial judge of AI assistant answers. Each item in the batch file has:
- "item": an id to copy into your output,
- "conversation": the user and assistant turns so far; judge ONLY the final assistant turn,
- "reference": a correct reference answer for that turn, or null.

For each item, rate the final assistant turn from 1 to 10 for helpfulness, relevance, accuracy, depth and
clarity, as a strict MT-Bench judge would. When a reference is given, correctness against the reference
dominates: a wrong final answer to a maths, reasoning or coding question scores at most 3. Check arithmetic
and logic yourself. Answers that repeat themselves or degenerate into loops score at most 2. An answer cut off
at a length limit is judged on what is there. Do not reward length for its own sake.

Write ONLY a JSON array, one object per item in the same order, each {"item": ..., "score": <1-10 integer>,
"reason": "<at most 15 words>"}, to the output path you were given.
"""


def batches(runid, models):
    work = RESULTS / f"{runid}_work"
    items, key = [], {}
    refs = {q["question_id"]: q.get("reference") or [] for q in
            map(json.loads, (BENCH / "data" / "mtbench.jsonl").read_text(encoding="utf-8").splitlines())}
    for model_dir in sorted((work).iterdir()):
        if not (model_dir / "mtbench").is_dir() or (models and model_dir.name not in models):
            continue
        for f in sorted((model_dir / "mtbench").glob("*.json")):
            q = json.loads(f.read_text(encoding="utf-8"))
            prompts = next(x["prompt"] for x in map(json.loads, (BENCH / "data" / "mtbench.jsonl").read_text(encoding="utf-8").splitlines())
                           if x["question_id"] == q["question_id"])
            conv = []
            for t, ans in enumerate(q.get("answers", [])):
                conv = conv + [{"role": "user", "content": prompts[t]}, {"role": "assistant", "content": ans["content"]}]
                ref = refs[q["question_id"]]
                items.append(({"model": model_dir.name, "question_id": q["question_id"], "turn": t + 1,
                               "category": q["category"]},
                              {"conversation": conv, "reference": ref[t] if t < len(ref) else None}))
    random.Random(runid).shuffle(items)
    out = work / "mtbench_judge"
    out.mkdir(exist_ok=True)
    (out / "INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8")
    for b in range(0, len(items), BATCH):
        batch = []
        for n, (meta, body) in enumerate(items[b:b + BATCH], start=b):
            key[f"m{n:04d}"] = meta
            batch.append({"item": f"m{n:04d}", **body})
        (out / f"batch_{b // BATCH:02d}.json").write_text(json.dumps(batch, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(items)} turns in {-(-len(items) // BATCH)} batches under {out}")


def merge(runid):
    out = RESULTS / f"{runid}_work" / "mtbench_judge"
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    scores = defaultdict(lambda: defaultdict(list))
    for f in sorted(out.glob("batch_*_out.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            meta = key[r["item"]]
            scores[meta["model"]][meta["turn"]].append(r["score"])
            scores[meta["model"]][meta["category"]].append(r["score"])
    for model, by in scores.items():
        f = RESULTS / f"{runid}_result_{model}.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text(encoding="utf-8"))
        allv = by[1] + by[2]
        res["mtbench_haiku"] = round(sum(allv) / len(allv), 2)
        res["mtbench_haiku_detail"] = {str(k): round(sum(v) / len(v), 2) for k, v in by.items()} | {"n": len(allv)}
        f.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(model, res["mtbench_haiku"], res["mtbench_haiku_detail"])
    write_summary(RESULTS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["batches", "merge"])
    ap.add_argument("--runid", required=True)
    ap.add_argument("--models", default="")
    a = ap.parse_args()
    batches(a.runid, set(filter(None, a.models.split(",")))) if a.mode == "batches" else merge(a.runid)
