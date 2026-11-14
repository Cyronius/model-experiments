"""Stage 2 step 1: hosted-teacher vectors (no instruction, bare strings) for
all vocabulary entries ("##" stripped; special and [unused*] tokens skipped)
and the surface strings of the T1 bigrams and trigrams (or T2 with --tier T2).

Writes training-v2/work/ngram-space/stage2/seedvec_{vocab,n2_T1,n3_T1}.npy
(float16 [n, 4096], rows aligned with the vocabulary / the ranked key files)
and seed_items.json (counts, spend before/after).

Usage: DEEPINFRA_API_KEY=... python -I stage2/seed_embed.py [--probe 200] [--tier T1]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

S1 = C.WORK / "stage1"
OUT = C.WORK / "stage2"
TIERS = {"T1": (0, 250_000), "T2": (250_000, 1_000_000)}


def vocab_items():
    tok = C.tokenizer()
    items = []
    for i in range(C.VOCAB):
        s = tok.id_to_token(i)
        if i in C.SPECIAL_IDS or (s.startswith("[") and s.endswith("]")):
            items.append(None)
            continue
        s = s[2:] if s.startswith("##") else s
        items.append(s if s else None)
    return items


def ngram_items(n, lo, hi):
    with open(S1 / f"ranked_n{n}_surface.jsonl") as fh:
        surf = [json.loads(l) for l in fh]
    return surf[lo:hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--tier", default="T1")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    sh = C.shared()
    sh.WORKERS = a.workers          # concurrency of this process only; shared.py unchanged
    OUT.mkdir(parents=True, exist_ok=True)
    lo, hi = TIERS[a.tier]
    groups = {"vocab": vocab_items(), f"n2_{a.tier}": ngram_items(2, lo, hi), f"n3_{a.tier}": ngram_items(3, lo, hi)}
    if a.tier != "T1":
        groups.pop("vocab")
    before = sh.spend("B")
    t0 = time.time()
    if a.probe:
        texts = [t for t in groups.get("vocab", groups[f"n2_{a.tier}"]) if t][:a.probe // 2] + \
                [t for t in groups[f"n3_{a.tier}"] if t][:a.probe - a.probe // 2]
        C.embed(texts)
        after = sh.spend("B")
        tokens = after["tokens"] - before["tokens"]
        print(json.dumps({"probe_items": len(texts), "tokens": tokens, "dollars": after["dollars"] - before["dollars"],
                          "tokens_per_item": tokens / len(texts), "spend_B": after}))
        return
    info = {}
    for g, items in groups.items():
        idx = [i for i, t in enumerate(items) if t]
        vecs = np.zeros((len(items), 4096), np.float16)
        for b in range(0, len(idx), 20000):
            part = idx[b:b + 20000]
            vecs[part] = sh.embed([items[i] for i in part], plan="B")
            print(f"{time.time()-t0:6.0f}s {g} {b + len(part)}/{len(idx)} spend {sh.spend('B')}", flush=True)
        np.save(OUT / f"seedvec_{g}.npy", vecs)
        info[g] = {"items": len(items), "embedded": len(idx), "unique_strings": len(set(items[i] for i in idx))}
    after = sh.spend("B")
    info["spend_before"], info["spend_after"] = before, after
    info["tokens"] = after["tokens"] - before["tokens"]
    info["dollars"] = after["dollars"] - before["dollars"]
    json.dump(info, open(OUT / f"seed_items_{a.tier}.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
