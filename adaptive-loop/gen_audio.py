"""Audio datasets for the real-time pedal model: guitar-like input through pedal circuits at 48 kHz.

  python gen_audio.py muff      # one fixed Big Muff, many input clips: cache/audio/muff-{train,test}.npz
  python gen_audio.py family    # random family circuits, one clip each, plus circuit tokens:
                                #   cache/audio/family-{train,test}.npz
Input: x [n, S] source volts. Output: y [n, S] volts at the circuit's output node (100 kOhm load).
"""
import multiprocessing as mp
import os
import sys

import numpy as np

import analog
from gen_analog import tokenize

FS = 48000
S = 4800  # 0.1 s


def pluck(rng, f, S):
    """Karplus-Strong string: noise burst through a delay line with a lowpass, decaying."""
    n = max(2, int(round(FS / f)))
    buf = rng.uniform(-1, 1, n)
    out = np.zeros(S)
    damp = rng.uniform(0.990, 0.999)
    for i in range(S):
        out[i] = buf[i % n]
        buf[i % n] = damp * 0.5 * (buf[i % n] + buf[(i + 1) % n])
    return out / (np.abs(out).max() + 1e-9)


def signal(rng):
    kind = rng.choice(["pluck", "chord", "sine", "chirp", "mute"], p=[0.4, 0.25, 0.15, 0.15, 0.05])
    t = np.arange(S) / FS
    if kind == "pluck":
        x = pluck(rng, 10 ** rng.uniform(np.log10(80), np.log10(1000)), S)
    elif kind == "chord":
        x = sum(pluck(rng, 10 ** rng.uniform(np.log10(80), np.log10(700)), S) for _ in range(rng.integers(2, 4)))
        x /= np.abs(x).max() + 1e-9
    elif kind == "sine":
        x = np.sin(2 * np.pi * 10 ** rng.uniform(np.log10(60), np.log10(3000)) * t + rng.uniform(0, 2 * np.pi))
    elif kind == "chirp":
        f0, f1 = 10 ** rng.uniform(np.log10(50), np.log10(300)), 10 ** rng.uniform(np.log10(1000), np.log10(6000))
        x = np.sin(2 * np.pi * f0 * S / FS / np.log(f1 / f0) * (np.exp(t / (S / FS) * np.log(f1 / f0)) - 1))
    else:  # palm-muted: short burst then near silence
        x = pluck(rng, 10 ** rng.uniform(np.log10(80), np.log10(300)), S) * np.exp(-t / rng.uniform(0.005, 0.02))
    onset = int(rng.integers(0, S // 4)) if kind in ("pluck", "chord", "mute") else 0
    x = np.concatenate([np.zeros(onset), x[:S - onset]])
    return x * 10 ** rng.uniform(-2, 0)  # 10 mV to 1 V peak, like pickups


def fixed_muff():
    rng = np.random.default_rng(12345)
    while True:
        c = analog.make_circuit(rng, muff_p=1.0)
        if c["muff"]:
            return c


def fixed_pool(k):
    rng = np.random.default_rng(777)
    return [analog.make_circuit(rng) for _ in range(k)]


def work(job):
    mode, seed, n = job
    rng = np.random.default_rng(seed)
    if mode == "muff":
        circs = [fixed_muff()] * n
    elif mode.startswith("fixed"):  # "fixed16": 16 fixed family circuits, clips spread over them
        pool = fixed_pool(int(mode[5:]))
        circs = [pool[i] for i in rng.integers(0, len(pool), n)]
    else:
        circs = [analog.make_circuit(rng) for _ in range(n)]
    x = np.stack([signal(rng) for _ in range(n)])
    y, ok = analog.Batch(circs).audio(x, FS, nodes=[c["out"] for c in circs])
    res = {"x": x[ok].astype(np.float32), "y": y[ok].astype(np.float32)}
    if mode != "muff":
        toks = [tokenize(c, np.zeros((c["n_nodes"], 1)), rng) for c, k in zip(circs, ok) if k]
        for k in ("type", "cont", "t1", "t2", "t3", "node", "role", "pad", "depth"):
            res[k] = np.stack([t[k] for t in toks])
        res["is_out"] = res["role"] == 11  # ROLES.index("out")
    return res, int((~ok).sum())


def build(mode, name, n, seed, chunk=64, procs=24):
    jobs = [(mode, seed * 100000 + i, chunk) for i in range(n // chunk)]
    with mp.Pool(procs) as pool:
        parts = pool.map(work, jobs)
    data = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    bad = sum(p[1] for p in parts)
    os.makedirs("cache/audio", exist_ok=True)
    np.savez(f"cache/audio/{mode}-{name}.npz", **data)
    print(f"{mode}-{name}: {len(data['x'])} clips, {bad} dropped (no convergence)", flush=True)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    mode = sys.argv[1]
    if mode == "muff":
        build("muff", "test", 512, 1)
        build("muff", "train", 8192, 2)
    elif mode.startswith("fixed"):
        build(mode, "test", 512, 1)
        build(mode, "train", 8192, 2)
    else:
        build("family", "test", 1024, 1)
        build("family", "train", 49152, 2)
