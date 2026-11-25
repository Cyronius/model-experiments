"""Generate tokenized Big Muff-family pedal datasets with solved node waveforms.

One token per component and one per node, in random order. Components point at their terminals'
node labels; nodes carry their own label. Targets (node tokens only): voltage at 16 phases of the
last of 4 input-sine periods, starting from the DC operating point.

  python gen_analog.py            # writes cache/analog/{train,test,ood}.npz
"""
import multiprocessing as mp
import os
import sys

import numpy as np

import analog

TOK = 96
LABELS = 64
NONE = LABELS
TYPES = ["NODE", "GND", "R", "C", "V", "VS", "D", "Q", "PAD"]
TI = {t: i for i, t in enumerate(TYPES)}
ROLES = ["gnd", "vcc", "src", "in", "base", "coll", "emit", "clip", "tone_lp", "tone_hp", "tone_out", "out"]


def features(kind, v):
    if kind == "R":
        return [(np.log10(v[0]) - 4) / 2, 0, 0]
    if kind == "C":
        return [(np.log10(v[0]) + 7) / 2, 0, 0]
    if kind == "V":
        return [v[0] / 9, 0, 0]
    if kind == "VS":
        return [np.log10(v[0]) + 1, np.log10(v[1]) - 2.5, 0]
    if kind == "D":
        return [(np.log10(v[0]) + 12) / 6, v[1] - 1.5, 0]
    if kind == "Q":
        return [np.log10(v[0]) + 14, (v[2 - 1] - 350) / 250, 0]
    raise ValueError(kind)


def tokenize(c, wave, rng):
    n = c["n_nodes"]
    labels = rng.permutation(LABELS)[:n]
    rows = []
    for k in range(n):
        rows.append(dict(type=TI["GND" if k == 0 else "NODE"], cont=[0, 0, 0], t=[labels[k], NONE, NONE],
                         node=1, target=wave[k], depth=c["depth"][k], role=ROLES.index(c["roles"][k])))
    for kind, term, v in c["comps"]:
        t = [labels[x] for x in term] + [NONE] * (3 - len(term))
        rows.append(dict(type=TI[kind], cont=features(kind, v), t=t, node=0, target=np.zeros(wave.shape[-1]), depth=-1, role=-1))
    assert len(rows) <= TOK, len(rows)
    order = rng.permutation(len(rows))
    out = {
        "type": np.full(TOK, TI["PAD"], np.int64), "cont": np.zeros((TOK, 3), np.float32),
        "t1": np.full(TOK, NONE, np.int64), "t2": np.full(TOK, NONE, np.int64), "t3": np.full(TOK, NONE, np.int64),
        "node": np.zeros(TOK, bool), "target": np.zeros((TOK, wave.shape[-1]), np.float32),
        "depth": np.full(TOK, -1, np.int64), "role": np.full(TOK, -1, np.int64), "pad": np.zeros(TOK, bool),
    }
    for i, j in enumerate(order):
        r = rows[j]
        out["type"][i] = r["type"]
        out["cont"][i] = r["cont"]
        out["t1"][i], out["t2"][i], out["t3"][i] = r["t"]
        out["node"][i] = r["node"]
        out["target"][i] = r["target"]
        out["depth"][i] = r["depth"]
        out["role"][i] = r["role"]
    out["pad"][len(rows):] = True
    return out


def work(job):
    seed, n, kw, dc = job
    rng = np.random.default_rng(seed)
    circs = [analog.make_circuit(rng, **kw) for _ in range(n)]
    b = analog.Batch(circs)
    if dc:  # DC operating point only (signal source at 0 V): [B, N, 1]
        x, ok = b.dc()
        wave = x[:, :b.N, None]
    else:
        wave, ok = b.transient()
    toks = [tokenize(c, wave[i], rng) for i, c in enumerate(circs) if ok[i]]
    return {k: np.stack([t[k] for t in toks]) for k in toks[0]}, int((~ok).sum())


def build(name, n, seed, kw, procs=24, chunk=128, dc=False):
    jobs = [(seed * 100000 + i, chunk, kw, dc) for i in range(n // chunk)]
    parts, bad = [], 0
    with mp.Pool(procs) as pool:
        for i, (p, b) in enumerate(pool.imap_unordered(work, jobs)):
            parts.append(p)
            bad += b
            if i % 100 == 0:
                print(f"{name}: {i}/{len(jobs)} chunks, {bad} dropped", flush=True)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out = "cache/analog-dc" if dc else "cache/analog"
    os.makedirs(out, exist_ok=True)
    np.savez(f"{out}/{name}.npz", **data)
    print(f"{name}: {len(data['type'])} circuits, {bad} dropped (no convergence)", flush=True)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    dc = "--dc" in sys.argv  # DC bias-point dataset in cache/analog-dc/
    which = [a for a in sys.argv[1:] if a != "--dc"] or ["test", "ood", "train"]
    if "test" in which:
        build("test", 4096, 1, {}, dc=dc)
    if "ood" in which:
        build("ood", 4096, 2, {"muff_p": 0.0, "stages_lo": 5, "stages_hi": 6}, dc=dc)
    if "train" in which:
        build("train", 400000, 3, {}, dc=dc)
