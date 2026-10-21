# Build blind judge batches: coherence of every answer (models shuffled, opaque ids) and agreement of each shrunk answer with the stock one (2026-10-19).
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = sys.argv[2:] or ["stock", "mlp75", "mlp50-selfchat"]
TAG = sys.argv[1] if len(sys.argv) > 1 else "all"
prompts = {p["id"]: p for p in map(json.loads, (HERE / "prompts.jsonl").open(encoding="utf-8"))}
ans = {m: {a["id"]: a for a in map(json.loads, (HERE / f"answers_{m}.jsonl").open(encoding="utf-8"))} for m in MODELS}
rng = random.Random(TAG)

items, key = [], {}
for m in MODELS:
    for i in prompts:
        items.append((m, i))
rng.shuffle(items)
coh = []
for n, (m, i) in enumerate(items):
    iid = f"{TAG[0]}c{n:03d}"
    key[iid] = {"model": m, "id": i}
    coh.append({"item": iid, "prompt": prompts[i]["prompt"], "answer": ans[m][i]["answer"]})

pairs = [(m, i) for m in MODELS[1:] for i in prompts]
rng.shuffle(pairs)
agr = []
for n, (m, i) in enumerate(pairs):
    iid = f"{TAG[0]}a{n:03d}"
    key[iid] = {"model": m, "id": i}
    agr.append({"item": iid, "prompt": prompts[i]["prompt"], "reference": ans["stock"][i]["answer"], "candidate": ans[m][i]["answer"]})

jd = HERE / "judge" / TAG
jd.mkdir(parents=True, exist_ok=True)
for b in range(0, len(coh), 20):
    (jd / f"coherence_{b // 20}.json").write_text(json.dumps(coh[b:b + 20], ensure_ascii=False, indent=1), encoding="utf-8")
for b in range(0, len(agr), 20):
    (jd / f"agreement_{b // 20}.json").write_text(json.dumps(agr[b:b + 20], ensure_ascii=False, indent=1), encoding="utf-8")
(jd / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
print(len(coh), len(agr))
