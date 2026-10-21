# Tally the Haiku judge outputs per model and per prompt category (2026-10-19).
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
jd = HERE / "judge" / (sys.argv[1] if len(sys.argv) > 1 else "first")
key = json.loads((jd / "key.json").read_text(encoding="utf-8"))
prompts = {p["id"]: p for p in map(json.loads, (HERE / "prompts.jsonl").open(encoding="utf-8"))}
coh, agr = defaultdict(list), defaultdict(list)
for f in sorted(jd.glob("*_out.json")):
    for r in json.loads(f.read_text(encoding="utf-8")):
        k = key[r["item"]]
        r.update(model=k["model"], id=k["id"], cat=prompts[k["id"]]["cat"])
        (coh if f.name.startswith("coherence") else agr)[k["model"]].append(r)
out = {}
for m, rs in coh.items():
    chk = [r for r in rs if r["correct"] in ("yes", "no")]
    fmt = [r for r in rs if r.get("follows_format") is not None]
    out[m] = {"n": len(rs), "coherence_mean": sum(r["coherent"] for r in rs) / len(rs),
              "coherent_4plus": sum(r["coherent"] >= 4 for r in rs) / len(rs),
              "loop": sum(bool(r["loop"]) for r in rs) / len(rs), "on_topic": sum(bool(r["on_topic"]) for r in rs) / len(rs),
              "correct": f"{sum(r['correct'] == 'yes' for r in chk)}/{len(chk)}",
              "follows_format": f"{sum(bool(r['follows_format']) for r in fmt)}/{len(fmt)}",
              "by_cat": {c: round(sum(r["coherent"] for r in rs if r["cat"] == c) / max(1, sum(r["cat"] == c for r in rs)), 2)
                         for c in sorted({r["cat"] for r in rs})}}
for m, rs in agr.items():
    out[m]["same_substance"] = dict(Counter(r["same_substance"] for r in rs))
    out[m]["better"] = dict(Counter(r["better"] for r in rs))
(jd / "tally.json").write_text(json.dumps({"summary": out, "coherence": coh, "agreement": agr}, indent=1), encoding="utf-8")
for m, s in out.items():
    print(m, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()})
