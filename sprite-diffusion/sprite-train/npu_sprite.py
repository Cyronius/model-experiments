# Sprite sheet generator on the NPU (or its numpy model): DiT steps on dit_gemm / dit_fa / dit_ew, host CFG + Euler + palette snap; 2026-10-19.
#   . C:\dev\mlir-aie\iron_env.ps1
#   python npu_sprite.py --weights weights-p2.npz --sets sets-p2 check --golden goldens\p2.npz [--np]
#   python npu_sprite.py --weights weights-p2.npz --sets sets-p2 sheet --cat food --steps 16 --out sheet.png
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ['OFLM_ROOT'])   # OpenFlowLM checkout: dit_gemm's pack.py and the pyxrt harness
sys.path.insert(0, str(ROOT / 'open_kernels' / 'designs' / 'dit_gemm'))
sys.path.insert(0, str(ROOT / 'open_kernels' / 'harness'))
EL, H, DEPTH = 384, 384, 10
CATS = ['creature', 'item', 'weapon', 'food', 'effect', 'terrain', 'structure', 'nature']
NONE = 8
CFG = 3.0


def patchify(x, p):          # (B,4,32,32) -> (B, T, 4p^2), the order of dit.py's unfold
    B, n = len(x), 32 // p
    return x.reshape(B, 4, n, p, n, p).transpose(0, 2, 4, 1, 3, 5).reshape(B, n * n, 4 * p * p)


def unpatchify(v, p):
    B, n = len(v), 32 // p
    return v.reshape(B, n, n, 4, p, p).transpose(0, 3, 1, 4, 2, 5).reshape(B, 4, 32, 32)


class Model:
    """The schedule, shared by both backends. Rows: sprites 0..n-1 conditional, then n..2n-1 unconditional."""

    def __init__(self, W, sprites):
        self.W, self.meta = W, json.loads(str(W['meta']))
        self.p, self.P, self.side = self.meta['patch'], self.meta['P'], self.meta['side']
        self.T = self.side ** 2
        self.n = sprites
        self.M = 2 * sprites * self.T
        self.half = self.M // 2

    def mods(self, steps, k, cat):
        """(ln0 [half][2][H], res [half][depth][4][H]) for step k: conditional = cat, unconditional = none."""
        ln0, res = self.W[f'ln0_s{steps}'][k], self.W[f'res_s{steps}'][k]
        return [ln0[cat], ln0[NONE]], [res[cat], res[NONE]]

    def eval(self, latent_patches, steps, k, cat, hook=None):
        """Velocity (2n, T, P) for latent patches (n, T, P), shared by both halves."""
        ln0, res = self.mods(steps, k, cat)
        self.write_input(np.concatenate([latent_patches, latent_patches]))
        self.gemm('inp')
        for h in range(2):
            self.ln(h, ln0[h])
        if hook: hook('inp', self)
        for i in range(DEPTH):
            self.gemm('lin1', i)
            self.qk(i)
            self.attn()
            self.swiglu()
            self.gemm('lin2', i)
            for h in range(2):
                self.res(h, i, res[h][i])
            if hook: hook(f'block{i}', self)
        self.gemm('outp')
        return self.read_output()


class NumpyBackend(Model):
    """fp32 model of the NPU dataflow (same buffers and layouts), for checking the layouts."""

    def __init__(self, W, sprites):
        super().__init__(W, sprites)
        M = self.M
        self.X, self.XM = np.zeros((M, 1024), np.float32), np.zeros((M, 512), np.float32)
        self.H1, self.L2IN = np.zeros((M, 4096), np.float32), np.zeros((M, 1536), np.float32)
        self.Y2, self.VOUT = np.zeros((M, 1024), np.float32), np.zeros((M, 1024), np.float32)
        self.AIN = np.zeros((M, self.P + 1), np.float32)
        self.XM[:, H] = 1.0

    def write_input(self, x):
        self.AIN[:, :self.P] = x.reshape(self.M, self.P)
        self.AIN[:, self.P] = 1.0

    def read_output(self):
        return self.VOUT[:, :self.P].reshape(2 * self.n, self.T, self.P)

    def gemm(self, name, i=None):
        w = self.W[name if i is None else f'{name}_{i}']
        if name == 'inp':
            self.X[:] = self.AIN @ w[:self.P + 1]
        elif name == 'lin1':
            self.H1[:] = self.XM @ w
        elif name == 'lin2':
            self.Y2[:] = self.L2IN @ w
        else:
            self.VOUT[:] = self.XM @ w

    @staticmethod
    def _ln(x):
        mu = x.mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + 1e-6)

    def ln(self, h, m):
        r = slice(h * self.half, (h + 1) * self.half)
        self.XM[r, :H] = self._ln(self.X[r, :H]) * (1 + m[1]) + m[0]

    def res(self, h, i, m):
        r = slice(h * self.half, (h + 1) * self.half)
        self.X[r, :H] = self.X[r, :H] + m[0] * self.Y2[r, :H] + m[3]
        self.XM[r, :H] = self._ln(self.X[r, :H]) * (1 + m[2]) + m[1]

    def qk(self, i):
        w = self.W[f'qkw_{i}']
        tab = self.W['rope']
        k = np.arange(self.M) % self.T
        y, x = k // self.side, k % self.side
        ang_c = np.concatenate([tab[y, :32], tab[x, :32]], 1)    # cos per pair
        ang_s = np.concatenate([tab[y, 32:], tab[x, 32:]], 1)
        for col, wt in ((0, w[:128]), (512, w[128:256])):
            for hd in range(3):
                c0 = col + hd * 128
                t = self.H1[:, c0:c0 + 128]
                t = t / np.sqrt((t ** 2).mean(-1, keepdims=True) + 1e-6) * wt
                a, b = t[:, 0::2], t[:, 1::2]
                out = np.empty_like(t)
                out[:, 0::2], out[:, 1::2] = a * ang_c - b * ang_s, a * ang_s + b * ang_c
                self.H1[:, c0:c0 + 128] = out

    def attn(self):
        T = self.T
        for s in range(2 * self.n):
            r = slice(s * T, (s + 1) * T)
            for hd in range(4):
                q = self.H1[r, hd * 128:(hd + 1) * 128]
                k = self.H1[r, 512 + hd * 128:512 + (hd + 1) * 128]
                v = self.H1[r, 1024 + hd * 128:1024 + (hd + 1) * 128]
                sc = q @ k.T / math.sqrt(128)
                p = np.exp(sc - sc.max(1, keepdims=True))
                self.L2IN[r, hd * 128:(hd + 1) * 128] = (p / p.sum(1, keepdims=True)) @ v

    def swiglu(self):
        g = self.H1[:, 1536:2688]
        self.L2IN[:, 384:1536] = g / (1 + np.exp(-g)) * self.H1[:, 2688:3840]


class NpuBackend(Model):
    def __init__(self, W, sprites, sets):
        super().__init__(W, sprites)
        from npu_host import Npu
        from pack import pack_b
        self.sets = json.loads((Path(sets) / 'sprite_sets.json').read_text())
        assert self.sets['M'] == self.M, (self.sets['M'], self.M)
        self.lda = self.sets['lda']
        npu = self.npu = Npu()
        ks = {k: npu.kernel_set(k, Path(sets) / k) for k in ('gemm', 'fa', 'ew')}
        self.st = {n: ks['gemm'].stream(n) for n in ('inp', 'lin1', 'lin2', 'outp')}
        self.st['attn'] = ks['fa'].stream('attn')
        for n in ('ln', 'res', 'qk', 'swiglu'):
            self.st[n] = ks['ew'].stream(n)
        M, b2 = self.M, 2
        self.AIN = npu.buf('AIN', (M * self.lda + 512) * b2)
        self.X = npu.buf('X', M * 1024 * b2)
        self.XM = npu.buf('XM', M * 512 * b2)
        self.H1 = npu.buf('H1', M * 4096 * b2)
        self.L2IN = npu.buf('L2IN', M * 1536 * b2)
        self.Y2 = npu.buf('Y2', M * 1024 * b2)
        self.VOUT = npu.buf('VOUT', M * 1024 * b2)
        self.D = npu.buf('D', 64 * EL * b2)
        for b in (self.AIN, self.X, self.XM, self.H1, self.L2IN, self.Y2, self.VOUT, self.D):
            b.zero()
        xm = np.zeros((M, 512), bfloat16); xm[:, H] = 1
        self.XM.write(xm)
        self.WB = {}
        for k in ['inp', 'outp'] + [f'{n}_{i}' for n in ('lin1', 'lin2') for i in range(DEPTH)]:
            pk = pack_b(W[k])
            self.WB[k] = npu.buf(f'W_{k}', pk.nbytes); self.WB[k].write(pk)
        # qk parameter runs: [slack, wq|wk, RoPE table (fp32 bits, 6 elements), slack] per block
        tab = np.zeros(6 * EL * 2, np.uint8)
        raw = W['rope'].astype(np.float32).tobytes(); tab[:len(raw)] = np.frombuffer(raw, np.uint8)
        qkp = np.zeros((DEPTH, 9, EL), np.uint16)
        for i in range(DEPTH):
            qkp[i, 1] = W[f'qkw_{i}'].astype(bfloat16).view(np.uint16)
            qkp[i, 2:8] = tab.view(np.uint16).reshape(6, EL)
        self.QKP = npu.buf('QKP', qkp.nbytes); self.QKP.write(qkp)
        self.MP = npu.buf('MP', 2 * (4 + DEPTH * 6) * EL * b2)   # one step's modulation runs, both halves
        self.dispatch_ms = []

    def _run(self, name, *bos):
        t0 = time.perf_counter()
        self.st[name].run(*bos)
        self.dispatch_ms.append((name, (time.perf_counter() - t0) * 1e3))

    def write_input(self, x):
        a = np.zeros((self.M, self.lda), np.float32)
        a[:, :self.P] = x.reshape(self.M, self.P); a[:, self.P] = 1
        self.AIN.write(a.astype(bfloat16))

    def read_output(self):
        v = self.VOUT.read(np.uint16).view(bfloat16).astype(np.float32).reshape(self.M, 1024)
        return v[:, :self.P].reshape(2 * self.n, self.T, self.P)

    def set_mods(self, ln0, res):
        mp = np.zeros((2, 4 + DEPTH * 6, EL), np.float32)
        for h in range(2):
            mp[h, 1:3] = ln0[h]
            for i in range(DEPTH):
                mp[h, 4 + 6 * i + 1:4 + 6 * i + 5] = res[h][i]
        self.MP.write(mp.astype(bfloat16))

    def eval(self, latent_patches, steps, k, cat, hook=None):
        ln0, res = self.mods(steps, k, cat)
        self.set_mods(ln0, res)
        return super().eval(latent_patches, steps, k, cat, hook)

    def _mp(self, h, run, n):
        off = ((h * (4 + DEPTH * 6)) + run) * EL * 2
        return self.MP.view(off, n * EL * 2)

    def gemm(self, name, i=None):
        w = self.WB[name if i is None else f'{name}_{i}'].bo
        a, c = {'inp': (self.AIN, self.X), 'lin1': (self.XM, self.H1), 'lin2': (self.L2IN, self.Y2),
                'outp': (self.XM, self.VOUT)}[name]
        self._run(name, a.bo, w, c.bo)

    def _rows(self, buf, h, ld):
        n = self.half * ld * 2
        return buf.view(h * n, n)

    def ln(self, h, m):
        self._run('ln', self._rows(self.X, h, 1024), self.D.bo, self._mp(h, 0, 4), self._rows(self.XM, h, 512), self.D.bo)

    def res(self, h, i, m):
        x = self._rows(self.X, h, 1024)
        self._run('res', x, self._rows(self.Y2, h, 1024), self._mp(h, 4 + 6 * i, 6),
                  self._rows(self.XM, h, 512), x)

    def qk(self, i):
        q = self.QKP.view(i * 9 * EL * 2, 9 * EL * 2)
        self._run('qk', self.H1.bo, self.H1.bo, q, self.H1.bo, self.H1.bo)

    def attn(self):
        self._run('attn', self.H1.bo, self.H1.bo, self.H1.bo, self.L2IN.bo)

    def swiglu(self):
        self._run('swiglu', self.H1.bo, self.H1.bo, self.D.bo, self.L2IN.bo, self.D.bo)

    def rows(self, name, cols):   # bf16 buffer -> float rows, for checks
        buf = getattr(self, name)
        ld = {'X': 1024, 'XM': 512, 'H1': 4096, 'L2IN': 1536, 'Y2': 1024}[name]
        return buf.read(np.uint16).view(bfloat16).astype(np.float32).reshape(self.M, ld)[:, cols]


def get_rows(be, name, cols):
    return be.rows(name, cols) if isinstance(be, NpuBackend) else getattr(be, name)[:, cols]


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def check(be, G):
    """Golden batch (n cond + n uncond at t) in rows [0, nT) and [half, half + nT)."""
    n, T = len(G['noise']), be.T
    t = float(G['t'])
    steps = 16
    ts = be.W[f'ts_s{steps}']
    k = int(np.argmin(np.abs(ts - t)))
    assert abs(ts[k] - t) < 1e-6, f'golden t {t} is not on the {steps}-step grid'
    cat = int(G['cat'][0])
    lat = np.zeros((be.n, T, be.P), np.float32)
    lat[:n] = patchify(G['noise'], be.p)
    rows = np.r_[0:n * T, be.half:be.half + n * T]

    def hook(stage, b):
        if stage == 'inp':
            got = get_rows(b, 'X', slice(0, H))[rows]
            print(f'  inp      rel {rel(got, G["inp_out"].reshape(-1, H)):.3e}')
        if stage == 'block0':
            got = get_rows(b, 'X', slice(0, H))[rows]
            print(f'  block0   rel {rel(got, G["b0_out"].reshape(-1, H)):.3e}')
            got = get_rows(b, 'L2IN', slice(0, 1536))[rows]
            print(f'  attn     rel {rel(got[:, :384], G["b0_attn"].reshape(-1, H)):.3e}   swiglu rel {rel(got[:, 384:], G["b0_lin2_in"].reshape(-1, 1536)[:, 384:]):.3e}')
        if stage == 'block9':
            got = get_rows(b, 'X', slice(0, H))[rows]
            print(f'  block9   rel {rel(got, G["last_out"].reshape(-1, H)):.3e}')

    v = be.eval(lat, steps, k, cat, hook)
    v = np.concatenate([v[:n], v[be.n:be.n + n]])
    ref = patchify(G['v'], be.p)
    cos = (v * ref).sum(-1) / (np.linalg.norm(v, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-12)
    r = rel(v, ref)
    print(f'  velocity rel {r:.3e}  min row cos {cos.min():.5f}  {"PASS" if r <= 3e-2 and cos.min() > 0.999 else "FAIL"}')
    return r


def sample(be, cat, steps, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((be.n, 4, 32, 32)).astype(np.float32)
    ts = be.W[f'ts_s{steps}']
    lat = patchify(x, be.p)
    t0 = time.perf_counter()
    for k in range(steps):
        v = be.eval(lat, steps, k, cat)
        vc, vu = v[:be.n], v[be.n:]
        lat = lat + (ts[k + 1] - ts[k]) * (vu + CFG * (vc - vu))
    return unpatchify(lat, be.p), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--sets')
    ap.add_argument('--np', action='store_true', help='numpy model of the dataflow instead of the NPU')
    ap.add_argument('--sprites', type=int, default=64)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check'); c.add_argument('--golden', required=True)
    s = sub.add_parser('sheet'); s.add_argument('--cat', default='food'); s.add_argument('--steps', type=int, default=16)
    s.add_argument('--seed', type=int, default=1); s.add_argument('--out', required=True)
    a = ap.parse_args()
    W = dict(np.load(a.weights))
    be = NumpyBackend(W, a.sprites) if a.np else NpuBackend(W, a.sprites, a.sets)
    if a.cmd == 'check':
        check(be, np.load(a.golden))
    else:
        x, secs = sample(be, CATS.index(a.cat), a.steps, a.seed)
        sys.path.insert(0, str(HERE))
        from snap import snap2
        from PIL import Image
        sheet = Image.new('RGBA', (8 * 34, 8 * 34), (0, 0, 0, 0))
        for i, sp in enumerate(x):
            sheet.paste(Image.fromarray(snap2(sp)), ((i % 8) * 34 + 1, (i // 8) * 34 + 1))
        sheet.save(a.out)
        np.save(a.out + '.npy', x)
        print(f'sheet {a.cat} {a.steps} steps: {secs:.2f} s ({be.n} sprites)')
        if isinstance(be, NpuBackend):
            from collections import defaultdict
            d = defaultdict(float)
            for n, ms in be.dispatch_ms:
                d[n] += ms
            print('  dispatch ms per sheet by op:', {k: round(v) for k, v in sorted(d.items(), key=lambda kv: -kv[1])})


if __name__ == '__main__':
    main()
