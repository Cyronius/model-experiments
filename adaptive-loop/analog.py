"""Batched analog circuit solver (MNA + Newton, backward-Euler transient) and a Big Muff-family generator.

Components (terminals as node indices, node 0 = ground):
  R  (a, b)        resistance
  C  (a, b)        capacitance (open at DC; backward Euler in transient)
  V  (p, n)        DC source
  VS (p, n)        sine source A*sin(2*pi*f*t), 0 at DC
  D  (anode, cath) Shockley diode Is, n
  Q  (c, b, e)     NPN, Ebers-Moll transport model Is, betaF, betaR

Unknowns per circuit: node voltages 1..N-1 and one current per voltage source. Circuits in a batch
are padded to the same node/component counts; padding components connect ground to ground, so their
stamps land only in the dropped ground row/column.
"""
import numpy as np

VT = 0.02585
GMIN = 1e-9
GMIN_D = 1e-6  # siemens across each diode (added 2026-11-22, after the transient data): an unbiased diode pair
#                otherwise leaves a node floating at DC, and GMIN to ground then drags it toward 0 V


def lexp(x):
    """exp, linearised above 40 to avoid overflow. Returns (value, derivative)."""
    big = x > 40.0
    e = np.exp(np.minimum(x, 40.0))
    return np.where(big, e * (1 + x - 40.0), e), e


class Batch:
    def __init__(self, circuits):
        self.B = len(circuits)
        self.N = max(c["n_nodes"] for c in circuits)
        kinds = ["R", "C", "V", "VS", "D", "Q"]
        nterm = {"R": 2, "C": 2, "V": 2, "VS": 2, "D": 2, "Q": 3}
        self.idx, self.val = {}, {}
        for k in kinds:
            comps = [[x for x in c["comps"] if x[0] == k] for c in circuits]
            m = max(1, max(len(x) for x in comps))
            idx = np.zeros((self.B, m, nterm[k]), np.int64)
            nv = {"R": 1, "C": 1, "V": 1, "VS": 2, "D": 2, "Q": 3}[k]
            val = np.zeros((self.B, m, nv))
            if k in ("R", "C"):
                val[:] = 1.0
            if k == "D":
                val[:] = (1e-14, 1.0)
            if k == "Q":
                val[:] = (1e-14, 100.0, 1.0)
            for i, cl in enumerate(comps):
                for j, (_, t, v) in enumerate(cl):
                    idx[i, j] = t
                    val[i, j] = v
            self.idx[k], self.val[k] = idx, val
        self.M = self.idx["V"].shape[1] + self.idx["VS"].shape[1]
        self.n = self.N + self.M  # including ground row 0, dropped at solve time
        self.rows = np.arange(self.B)[:, None]

    # --- stamping helpers (full system with ground at index 0) ---
    def _stamp_g(self, A, a, b, g):
        r = self.rows
        np.add.at(A, (r, a, a), g)
        np.add.at(A, (r, b, b), g)
        np.add.at(A, (r, a, b), -g)
        np.add.at(A, (r, b, a), -g)

    def linear(self, dt=None, vprev=None, t=0.0, scale=1.0):
        """Linear part A x = b (resistors, caps, sources). Returns A [B,n,n], rhs [B,n]."""
        A = np.zeros((self.B, self.n, self.n))
        rhs = np.zeros((self.B, self.n))
        r = self.rows
        ia = np.arange(1, self.N)
        A[:, ia, ia] += GMIN
        R = self.idx["R"]
        self._stamp_g(A, R[..., 0], R[..., 1], 1.0 / self.val["R"][..., 0])
        if dt is not None:
            Ci = self.idx["C"]
            g = self.val["C"][..., 0] / dt
            self._stamp_g(A, Ci[..., 0], Ci[..., 1], g)
            hist = g * (vprev[r, Ci[..., 0]] - vprev[r, Ci[..., 1]])
            np.add.at(rhs, (r, Ci[..., 0]), hist)
            np.add.at(rhs, (r, Ci[..., 1]), -hist)
        src = [("V", self.val["V"][..., 0] * scale)]
        vs = self.val["VS"]
        src.append(("VS", vs[..., 0] * np.sin(2 * np.pi * vs[..., 1] * t) if dt is not None else 0 * vs[..., 0]))
        k0 = self.N
        for kind, volts in src:
            I = self.idx[kind]
            m = I.shape[1]
            kk = k0 + np.arange(m)[None].repeat(self.B, 0)
            p, q = I[..., 0], I[..., 1]
            real = (p != q).astype(float)  # padding sources (0,0) become i = 0 rows
            np.add.at(A, (r, p, kk), real)
            np.add.at(A, (r, q, kk), -real)
            np.add.at(A, (r, kk, p), real)
            np.add.at(A, (r, kk, q), -real)
            np.add.at(A, (r, kk, kk), 1.0 - real)
            rhs[r, kk] = volts * real
            k0 += m
        return A, rhs

    def nonlinear(self, x):
        """Currents leaving each node through diodes/BJTs, and their Jacobian. x [B,n] (full)."""
        F = np.zeros((self.B, self.n))
        J = np.zeros((self.B, self.n, self.n))
        r = self.rows
        D = self.idx["D"]
        Is, nf = self.val["D"][..., 0], self.val["D"][..., 1]
        a, k = D[..., 0], D[..., 1]
        e, de = lexp((x[r, a] - x[r, k]) / (nf * VT))
        i = Is * (e - 1) + GMIN_D * (x[r, a] - x[r, k])
        g = Is * de / (nf * VT) + GMIN_D
        np.add.at(F, (r, a), i)
        np.add.at(F, (r, k), -i)
        self._stamp_g(J, a, k, g)
        Q = self.idx["Q"]
        Is, bf, br = self.val["Q"][..., 0], self.val["Q"][..., 1], self.val["Q"][..., 2]
        c, b, em = Q[..., 0], Q[..., 1], Q[..., 2]
        ebe, dbe = lexp((x[r, b] - x[r, em]) / VT)
        ebc, dbc = lexp((x[r, b] - x[r, c]) / VT)
        ic = Is * (ebe - ebc) - Is / br * (ebc - 1)
        ib = Is / bf * (ebe - 1) + Is / br * (ebc - 1)
        # partial derivatives wrt vbe, vbc
        gbe, gbc = Is * dbe / VT, Is * dbc / VT
        dic = (gbe, -gbc - gbc / br)
        dib = (gbe / bf, gbc / br)
        np.add.at(F, (r, c), ic)
        np.add.at(F, (r, b), ib)
        np.add.at(F, (r, em), -(ic + ib))
        # vbe = vb - ve, vbc = vb - vc
        for node, (d1, d2), sgn in ((c, dic, 1), (b, dib, 1), (em, (dic[0] + dib[0], dic[1] + dib[1]), -1)):
            np.add.at(J, (r, node, b), sgn * (d1 + d2))
            np.add.at(J, (r, node, em), -sgn * d1)
            np.add.at(J, (r, node, c), -sgn * d2)
        return F, J

    def newton(self, A, rhs, x, iters=60, tol=1e-7, max_step=0.3):
        """Solve A x - rhs + I_nl(x) = 0. Returns x and a converged flag per circuit."""
        ok = np.zeros(self.B, bool)
        for _ in range(iters):
            Fn, Jn = self.nonlinear(x)
            F = np.einsum("bij,bj->bi", A, x) - rhs + Fn
            J = A + Jn
            dx = np.zeros_like(x)
            dx[:, 1:] = np.linalg.solve(J[:, 1:, 1:], -F[:, 1:, None])[..., 0]
            vmax = np.abs(dx[:, 1:self.N]).max(1)
            s = np.minimum(1.0, max_step / np.maximum(vmax, 1e-30))
            x = x + s[:, None] * dx
            ok = (vmax < tol) & (s == 1.0)
            if ok.all():
                break
        return x, ok

    def dc(self, steps=6):
        """DC operating point, with source stepping (supply ramped up in `steps`)."""
        x = np.zeros((self.B, self.n))
        for sc in np.linspace(1.0 / steps, 1.0, steps):
            A, rhs = self.linear(scale=sc)
            x, ok = self.newton(A, rhs, x, iters=100)
        return x, ok

    def transient(self, periods=4, per_period=64, keep=16):
        """Start at the DC point, drive the sine sources, return node voltages at `keep` evenly spaced
        phases of the last period: [B, N, keep]. Assumes one frequency per circuit (first VS)."""
        x, ok = self.dc()
        f = self.val["VS"][:, 0, 1]
        dt = (1.0 / f) / per_period  # per-circuit step
        dtb = dt[:, None]
        out = np.zeros((self.B, self.N, keep))
        stride = per_period // keep
        nsteps = periods * per_period
        for s in range(1, nsteps + 1):
            t = (s * dt)[:, None]
            A, rhs = self.linear(dt=dtb, vprev=x, t=t)
            x_new, ok_s = self.newton(A, rhs, x, iters=60, tol=1e-6)
            if not ok_s.all():  # retry the step in 8 substeps; failures are rare
                ok_s_first = ok_s
                xs = x
                ok_s = np.ones(self.B, bool)
                for q in range(1, 9):
                    A, rhs = self.linear(dt=dtb / 8, vprev=xs, t=((s - 1 + q / 8) * dt)[:, None])
                    xs, o = self.newton(A, rhs, xs, iters=100, tol=1e-6)
                    ok_s &= o | ok_s_first
                x_new = np.where(ok_s_first[:, None], x_new, xs)  # only failed circuits take the substep path
            x = x_new
            ok &= ok_s
            j = s - (periods - 1) * per_period
            if j > 0 and j % stride == 0:
                out[:, :, j // stride - 1] = x[:, :self.N]
        return out, ok


    def audio(self, x_in, fs=48000, nodes=None, iters=100):
        """Drive each circuit's first VS with an arbitrary waveform x_in [B, S] (volts) at sample rate fs,
        starting from the DC point. Backward Euler, one step per sample. Returns voltages [B, S] at
        `nodes` [B] (default: each circuit's last node) and a converged flag per circuit."""
        x, ok = self.dc()
        B, S = x_in.shape
        r = self.rows
        dt = np.full((B, 1), 1.0 / fs)
        A, rhs_base = self.linear(dt=dt, vprev=np.zeros_like(x), t=0.0)  # A does not depend on vprev or t
        Ci = self.idx["C"]
        g = self.val["C"][..., 0] / dt
        vs_row = self.N + self.idx["V"].shape[1]
        real_vs = (self.idx["VS"][:, 0, 0] != self.idx["VS"][:, 0, 1]).astype(float)
        nodes = np.full(B, self.N - 1) if nodes is None else np.asarray(nodes)
        out = np.zeros((B, S))
        for s in range(S):
            rhs = rhs_base.copy()
            hist = g * (x[r, Ci[..., 0]] - x[r, Ci[..., 1]])
            np.add.at(rhs, (r, Ci[..., 0]), hist)
            np.add.at(rhs, (r, Ci[..., 1]), -hist)
            rhs[:, vs_row] = x_in[:, s] * real_vs
            x, ok_s = self.newton(A, rhs, x, iters=iters, tol=1e-6)
            ok &= ok_s
            out[:, s] = x[np.arange(B), nodes]
        return out, ok


# ---------------------------------------------------------------- generator

DIODES = {"si": (2.52e-9, 1.75), "ge": (2e-6, 1.3), "led": (1e-18, 2.0)}


def make_circuit(rng, muff_p=0.25, stages_lo=1, stages_hi=4):
    """Random Big Muff-family pedal. Returns dict(n_nodes, comps, roles, out)."""
    comps = []
    n = [2]  # 0 gnd, 1 vcc
    roles = ["gnd", "vcc"]

    depth = [0, 0]

    def node(role):
        roles.append(role)
        depth.append(stage_i[0])
        n[0] += 1
        return n[0] - 1

    stage_i = [0]
    muff = rng.random() < muff_p
    sp = 0.3 if muff else 1.0

    def j(x, w=1.0):  # log-uniform jitter, factor 2**(w*sp)
        return x * 2 ** rng.uniform(-w * sp, w * sp)

    comps.append(("V", (1, 0), (j(9.0, 0.15),)))
    src = node("src")
    amp = 10 ** rng.uniform(-2, 0)
    freq = 10 ** rng.uniform(np.log10(80), np.log10(1200))
    comps.append(("VS", (src, 0), (amp, freq)))
    inp = node("in")
    comps.append(("R", (src, inp), (j(5e3, 1.5),)))
    if muff:
        stages = ["ce", "clip", "clip", "tone", "ce"]
    else:
        stages = list(rng.choice(["ce", "clip", "tone"], size=rng.integers(stages_lo, stages_hi + 1), p=[0.4, 0.4, 0.2]))
    dt = rng.choice(list(DIODES))
    cur = inp
    for si, st in enumerate(stages):
        stage_i[0] = si + 1
        if st == "tone":
            a, b, w = node("tone_lp"), node("tone_hp"), node("tone_out")
            t = rng.uniform(0.05, 0.95)
            pot = j(100e3)
            comps += [("R", (cur, a), (j(22e3),)), ("C", (a, 0), (j(4e-9),)),
                      ("C", (cur, b), (j(10e-9),)), ("R", (b, 0), (j(22e3),)),
                      ("R", (a, w), (t * pot,)), ("R", (w, b), ((1 - t) * pot,))]
            cur = w
            continue
        b, c, e = node("base"), node("coll"), node("emit")
        comps += [("C", (cur, b), (j(100e-9),)), ("R", (c, b), (j(470e3),)), ("R", (1, c), (j(15e3),)),
                  ("R", (e, 0), (j(150, 1.5),)),
                  ("Q", (c, b, e), (j(1e-14, 2), rng.uniform(100, 600), 1.0))]
        if muff or rng.random() < 0.7:
            comps.append(("C", (c, b), (j(470e-12),)))
        if st == "clip":
            x = node("clip")
            Is, nf = DIODES[dt if muff or rng.random() < 0.7 else rng.choice(list(DIODES))]
            comps += [("C", (c, x), (j(1e-6),)), ("D", (x, b), (Is, nf)), ("D", (b, x), (Is, nf))]
        cur = c
    stage_i[0] = len(stages) + 1
    out = node("out")
    comps += [("C", (cur, out), (j(100e-9),)), ("R", (out, 0), (j(100e3),))]
    return {"n_nodes": n[0], "comps": comps, "roles": roles, "depth": depth, "out": out, "muff": muff,
            "stages": stages}


if __name__ == "__main__":
    import time

    # 1. RC low-pass, sine drive: compare last-period samples to the exact backward-Euler recursion.
    R, C, f, A = 1e3, 1e-6, 200.0, 1.0
    circ = {"n_nodes": 3, "comps": [("VS", (1, 0), (A, f)), ("R", (1, 2), (R,)), ("C", (2, 0), (C,))]}
    out, ok = Batch([circ]).transient(periods=4, per_period=64, keep=16)
    dt = 1 / f / 64
    v, ref = 0.0, []
    for s in range(1, 257):
        vin = A * np.sin(2 * np.pi * f * s * dt)
        v = (v * C / dt + vin / R) / (C / dt + 1 / R)
        if s > 192 and (s - 192) % 4 == 0:
            ref.append(v)
    print("RC max err", np.abs(out[0, 2] - np.array(ref)).max(), "ok", ok)

    # 2. Diode + resistor from 5 V: compare to scalar Newton.
    Is, nf = DIODES["si"]
    circ = {"n_nodes": 3, "comps": [("V", (1, 0), (5.0,)), ("R", (1, 2), (1e3,)), ("D", (2, 0), (Is, nf))]}
    x, ok = Batch([circ]).dc()
    vd = 0.6
    for _ in range(100):
        fval = (5 - vd) / 1e3 - Is * (np.exp(vd / (nf * VT)) - 1)
        fp = -1e-3 - Is / (nf * VT) * np.exp(vd / (nf * VT))
        vd -= fval / fp
    print("diode V", x[0, 2], "ref", vd, "ok", ok)

    # 3. Random pedals: convergence rate, KCL residual at the DC point, timing; Big Muff bias.
    rng = np.random.default_rng(0)
    circs = [make_circuit(rng) for _ in range(256)]
    bt = Batch(circs)
    t = time.time()
    x, ok = bt.dc()
    A, rhs = bt.linear()
    Fn, _ = bt.nonlinear(x)
    res = np.abs(np.einsum("bij,bj->bi", A, x) - rhs + Fn)[:, 1:].max(1)
    print(f"DC: {ok.mean():.3f} converged, max KCL residual {res[ok].max():.2e} A, {time.time() - t:.1f}s")
    t = time.time()
    w, okt = bt.transient()
    print(f"transient: {okt.mean():.3f} converged, {time.time() - t:.1f}s for 256, N={bt.N}")
    mi = next(i for i, c in enumerate(circs) if c["muff"])
    c = circs[mi]
    print("muff stages", c["stages"])
    for k, role in enumerate(c["roles"]):
        print(f"  node {k:2d} {role:9s} dc {x[mi, k]:7.3f}  wave {np.round(w[mi, k, ::2], 3)}")
