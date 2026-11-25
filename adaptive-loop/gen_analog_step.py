"""Time-step pedal data: given every node's voltage at sample n and the input at n+1, predict each
node's change in voltage over one audio sample (48 kHz).

Same circuits and tokens as gen_analog.py. Node tokens carry their voltage at n as continuous
features; the signal source token carries the input at n+1 and n (not its amplitude or frequency,
so a trained model can be driven by any audio). Target (node tokens): v[n+1] - v[n], volts.

Ground truth: the solver's backward-Euler step at dt = 1/48000, from the DC point, 480 samples
(10 ms). Capacitor charge is fully set by node voltages, so v[n] and the input are the whole state.

  python gen_analog_step.py     # writes cache/analog-step/{train,test,ood}.npz and {roll,roll_ood}.npz
"""
import multiprocessing as mp
import os
import sys

import numpy as np

import analog
from gen_analog import TI, tokenize

FS = 48000
NSAMP = 480
PER_CIRCUIT = 8  # training examples drawn from each simulated circuit


def simulate(circs):
    """States [B, NSAMP+1, N] from the DC point, input [B, NSAMP+1], ok flags."""
    b = analog.Batch(circs)
    x, ok = b.dc()
    dt = 1.0 / FS
    dtb = np.full((b.B, 1), dt)
    states = np.zeros((b.B, NSAMP + 1, b.N))
    states[:, 0] = x[:, :b.N]
    for s in range(1, NSAMP + 1):
        A, rhs = b.linear(dt=dtb, vprev=x, t=np.full((b.B, 1), s * dt))
        x_new, ok_s = b.newton(A, rhs, x, iters=60, tol=1e-6)
        if not ok_s.all():  # substep retry for failed circuits only, as in analog.Batch.transient
            first, xs = ok_s, x
            ok_s = np.ones(b.B, bool)
            for q in range(1, 9):
                A, rhs = b.linear(dt=dtb / 8, vprev=xs, t=np.full((b.B, 1), (s - 1 + q / 8) * dt))
                xs, o = b.newton(A, rhs, xs, iters=100, tol=1e-6)
                ok_s &= o | first
            x_new = np.where(first[:, None], x_new, xs)
        x = x_new
        ok &= ok_s
        states[:, s] = x[:, :b.N]
    amp = b.val["VS"][:, 0, 0]
    f = b.val["VS"][:, 0, 1]
    vin = amp[:, None] * np.sin(2 * np.pi * f[:, None] * np.arange(NSAMP + 1)[None] / FS)
    return states, vin, ok


def layout(c, rng):
    """Tokenize once; node rows get their node index in target[:, 0] so samples can be filled in."""
    lay = tokenize(c, np.repeat(np.arange(c["n_nodes"], dtype=np.float32)[:, None], 16, 1), rng)
    nidx = np.where(lay["node"], lay["target"][:, 0].astype(np.int64), -1)
    vs = int(np.flatnonzero(lay["type"] == TI["VS"])[0])
    return lay, nidx, vs


def fill(lay, nidx, vs, v_now, v_next, u_now, u_next):
    t = {k: v.copy() for k, v in lay.items() if k != "target"}
    m = nidx >= 0
    t["cont"][m] = np.stack([v_now[nidx[m]] / 9, v_now[nidx[m]], np.zeros(m.sum())], -1)
    t["cont"][vs] = [u_next, u_now, 0.0]
    t["target"] = np.zeros((len(nidx), 1), np.float32)
    t["target"][m, 0] = v_next[nidx[m]] - v_now[nidx[m]]
    return t


def work(job):
    seed, n, kw, mode = job
    rng = np.random.default_rng(seed)
    circs = [analog.make_circuit(rng, **kw) for _ in range(n)]
    states, vin, ok = simulate(circs)
    keep = [i for i in range(n) if ok[i] and np.isfinite(states[i]).all()]
    if mode == "roll":  # whole trajectories, for closed-loop evaluation
        lays = [layout(circs[i], rng) for i in keep]
        out = {k: np.stack([l[0][k] for l in lays]) for k in lays[0][0]}
        out["nidx"] = np.stack([l[1] for l in lays])
        out["vs"] = np.array([l[2] for l in lays])
        N = states.shape[2]
        st = np.zeros((len(keep), NSAMP + 1, 40), np.float32)
        st[:, :, :N] = states[keep]
        out["states"], out["vin"] = st, vin[keep].astype(np.float32)
        return out, n - len(keep)
    toks = []
    for i in keep:
        lay, nidx, vs = layout(circs[i], rng)
        for s in rng.integers(0, NSAMP, PER_CIRCUIT):
            toks.append(fill(lay, nidx, vs, states[i, s], states[i, s + 1], vin[i, s], vin[i, s + 1]))
    return {k: np.stack([t[k] for t in toks]) for k in toks[0]}, n - len(keep)


def build(name, n, seed, kw, mode="step", procs=16, chunk=64):
    jobs = [(seed * 100000 + i, chunk, kw, mode) for i in range(n // chunk)]
    parts, bad = [], 0
    with mp.Pool(procs) as pool:
        for i, (p, b) in enumerate(pool.imap_unordered(work, jobs)):
            parts.append(p)
            bad += b
            if i % 100 == 0:
                print(f"{name}: {i}/{len(jobs)} chunks, {bad} dropped", flush=True)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    os.makedirs("cache/analog-step", exist_ok=True)
    np.savez(f"cache/analog-step/{name}.npz", **data)
    print(f"{name}: {len(data['type'])} rows, {bad} circuits dropped", flush=True)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ood = {"muff_p": 0.0, "stages_lo": 5, "stages_hi": 6}
    which = sys.argv[1:] or ["test", "ood", "roll", "roll_ood", "train"]
    if "test" in which:
        build("test", 512, 11, {})
    if "ood" in which:
        build("ood", 512, 12, ood)
    if "roll" in which:
        build("roll", 512, 13, {}, mode="roll")
    if "roll_ood" in which:
        build("roll_ood", 256, 14, ood, mode="roll")
    if "train" in which:
        build("train", 50048, 15, {})
