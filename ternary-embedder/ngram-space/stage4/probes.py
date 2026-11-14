"""Stage 4 step 4: probes relative to the teacher (report only, not a gate).

Pairs are built only from public text the student was not trained on: the
dev-task documents (SciFact, NFCorpus, ArguAna, 504 contract) and the Jaswanth
rows outside the Plan B training set (validation, over-504-token, leakage).

Kinds (about 200 pairs each, seed 0):
  negation   a sentence and the same sentence with "not" after its first
             auxiliary (is/are/was/were/can/does/do/did/will/may/has/have)
  swap       "A <verb> B" sentences with A and B exchanged
  reorder    a 3-8 sentence passage and the same sentences shuffled
  entity     two sentences with the same ambiguous name in different senses
             (sense found by keywords), plus same-sense pairs as the control

Usage:
  DEEPINFRA_API_KEY=... python -I stage4/probes.py build      # pairs + teacher
  python -I stage4/probes.py student ARM                      # student cosines
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

OUT = C.WORK / "stage4" / "probes"
N = 200
AUX = r"\b(is|are|was|were|can|does|do|did|will|may|has|have)\b(?! not)"
VERBS = ("increases|decreases|reduces|causes|inhibits|activates|promotes|prevents|regulates|"
         "induces|blocks|enhances|suppresses|predicts|requires|affects|improves")
PREP = re.compile(r"^(with|in|by|to|from|after|before|during|for|of|on|at|over|under|than|when|if|that|whether)\b", re.I)
SWAP = re.compile(r"^([A-Za-z][\w\- ,()]{2,60}?) (" + VERBS + r") ([\w\- ,()]{3,60}?)\.?$")
SENSES = {  # name: (sense A keywords, sense B keywords)
    "apple": (["iphone", "ipad", "mac", "company", "inc", "itunes", "ios", "store"],
              ["fruit", "tree", "juice", "pie", "orchard", "eat", "cider"]),
    "amazon": (["prime", "kindle", "order", "seller", "shipping", "aws", "book"],
               ["river", "rainforest", "basin", "brazil", "jungle", "forest"]),
    "python": (["code", "programming", "script", "library", "function", "django"],
               ["snake", "reptile", "species", "prey", "venom", "zoo"]),
    "java": (["code", "programming", "class", "jvm", "oracle", "method"],
             ["island", "indonesia", "coffee", "jakarta", "volcano"]),
    "jaguar": (["car", "xj", "engine", "sedan", "land rover", "dealer"],
               ["cat", "species", "prey", "rainforest", "predator"]),
    "mercury": (["planet", "orbit", "sun", "nasa", "venus"],
                ["poisoning", "fish", "toxic", "exposure", "thermometer", "levels"]),
    "mars": (["planet", "rover", "nasa", "orbit", "mission"],
             ["bar", "chocolate", "bruno", "candy"]),
    "turkey": (["istanbul", "ankara", "erdogan", "country", "greece", "syria"],
               ["thanksgiving", "roast", "meat", "sandwich", "bird", "cook"]),
    "cell": (["phone", "mobile", "battery", "carrier", "call"],
             ["membrane", "tissue", "protein", "tumor", "stem", "receptor"]),
    "bank": (["loan", "account", "money", "deposit", "interest", "credit"],
             ["river", "shore", "erosion", "flood", "stream"]),
    "virus": (["computer", "antivirus", "malware", "software", "infected files"],
              ["infection", "vaccine", "hiv", "influenza", "patients", "viral"]),
    "pitch": (["baseball", "pitcher", "batter", "inning"],
              ["frequency", "voice", "tone", "sound", "note"]),
    "crane": (["construction", "lift", "load", "tower", "operator"],
              ["bird", "wetland", "species", "migration", "feathers"]),
    "seal": (["envelope", "rubber", "leak", "airtight", "gasket"],
             ["pup", "marine", "ice", "colony", "harbor"]),
    "shell": (["oil", "gas", "petroleum", "royal dutch"],
              ["egg", "snail", "beach", "turtle", "crab"]),
}


def pool():
    """Public texts outside the Plan B training set."""
    texts = []
    for t in ("SciFact", "NFCorpus", "ArguAna"):
        texts += json.load(open(C.WORK / "dev" / f"{t}-504-texts.json"))
    train = set(np.load(C.WORK / "stage0" / "train_rows.npy").tolist())
    with open(C.WORK / "jaswanth" / "texts.jsonl") as fh:
        for i, line in enumerate(fh):
            if i not in train:
                texts.append(json.loads(line))
    return [t for t in texts if isinstance(t, str) and t.strip()]


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def build_pairs(texts, rng):
    sents = list({s for t in texts for s in sentences(t) if 6 <= len(s.split()) <= 40})
    sents.sort()
    rng.shuffle(sents)
    pairs = {"negation": [], "swap": [], "reorder": [], "entity_diff": [], "entity_same": []}
    for s in sents:
        if len(pairs["negation"]) < N and re.search(AUX, s) and " not " not in s and "n't" not in s:
            pairs["negation"].append((s, re.sub(AUX, r"\1 not", s, count=1)))
        m = SWAP.match(s)
        if len(pairs["swap"]) < N and m and len(s.split()) <= 25 and m.group(1).lower() != m.group(3).lower() \
                and not PREP.match(m.group(3)) and len(m.group(1).split()) <= 8 and len(m.group(3).split()) <= 8:
            pairs["swap"].append((s, f"{m.group(3)} {m.group(2)} {m.group(1)}."))
    order = list(range(len(texts)))
    rng.shuffle(order)
    for i in order:
        ss = sentences(texts[i])
        if 3 <= len(ss) <= 8 and 30 <= len(texts[i].split()) <= 200:
            p = rng.permutation(len(ss))
            if (p == np.arange(len(ss))).all():
                p = p[::-1]
            pairs["reorder"].append((" ".join(ss), " ".join(ss[j] for j in p)))
            if len(pairs["reorder"]) >= N:
                break
    # entity: sense-tagged sentences per name
    tagged = {k: ([], []) for k in SENSES}
    for s in sents:
        low = s.lower()
        for name, (a, b) in SENSES.items():
            if re.search(rf"\b{name}\b", low):
                ina = any(re.search(rf"\b{w}\b", low) for w in a)
                inb = any(re.search(rf"\b{w}\b", low) for w in b)
                if ina != inb:
                    tagged[name][0 if ina else 1].append(s)
    per = {k: (len(v[0]), len(v[1])) for k, v in tagged.items()}
    usable = [k for k, (x, y) in per.items() if x >= 2 and y >= 2]
    k = 0
    while len(pairs["entity_diff"]) < N and usable and k < 50 * N:
        name = usable[k % len(usable)]
        A, B = tagged[name]
        a1, a2 = rng.choice(len(A), 2, replace=False)
        b1 = rng.integers(len(B))
        pairs["entity_diff"].append((A[a1], B[b1], name))
        pairs["entity_same"].append((A[a1], A[a2], name))
        k += 1
    return pairs, per


def build():
    rng = np.random.default_rng(0)
    pairs, per = build_pairs(pool(), rng)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"pairs": pairs, "entity_sense_counts": per}, open(OUT / "pairs.json", "w"), indent=1)
    texts = sorted({t for v in pairs.values() for p in v for t in p[:2]})
    vecs = C.unit(C.embed(texts))
    np.save(OUT / "teacher.npy", vecs.astype(np.float16))
    json.dump(texts, open(OUT / "texts.json", "w"))
    print({k: len(v) for k, v in pairs.items()}, "texts", len(texts))
    print(json.dumps(C.shared().spend("B")))


def report(student_vecs, arm):
    d = json.load(open(OUT / "pairs.json"))
    texts = json.load(open(OUT / "texts.json"))
    idx = {t: i for i, t in enumerate(texts)}
    T = np.load(OUT / "teacher.npy").astype(np.float32)[:, :1024]
    T = C.unit(T)
    S = C.unit(student_vecs[:, :1024])
    res = {}
    for kind, ps in d["pairs"].items():
        a = np.array([idx[p[0]] for p in ps]); b = np.array([idx[p[1]] for p in ps])
        tc = (T[a] * T[b]).sum(1); sc = (S[a] * S[b]).sum(1)
        res[kind] = {"n": len(ps), "teacher_cos_mean": float(tc.mean()), "student_cos_mean": float(sc.mean()),
                     "teacher_sep": float(1 - tc.mean()), "student_sep": float(1 - sc.mean()),
                     "kept_sep": float((1 - sc.mean()) / max(1 - tc.mean(), 1e-9)),
                     "pearson": float(np.corrcoef(tc, sc)[0, 1]) if len(ps) > 2 else None,
                     "teacher_separates_share": float((tc < 0.95).mean())}
    if "entity_diff" in res:
        tg = res["entity_same"]["teacher_cos_mean"] - res["entity_diff"]["teacher_cos_mean"]
        sg = res["entity_same"]["student_cos_mean"] - res["entity_diff"]["student_cos_mean"]
        res["entity_gap"] = {"teacher": tg, "student": sg, "kept": sg / tg if tg > 0 else None}
    res["arm"] = arm
    json.dump(res, open(C.PLAN / "stage4" / f"probes_{arm}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def student(arm, device="cpu"):
    sys.path.insert(0, str(C.PLAN))
    from eval_arm import load_arm
    from train import encode
    m, _ = load_arm(arm, device)
    texts = json.load(open(OUT / "texts.json"))
    seqs = [s[:504] if len(s) else np.array([C.UNK], np.uint16) for s in C.encode(texts)]
    report(encode(m, seqs, device), arm)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        student(sys.argv[2])
