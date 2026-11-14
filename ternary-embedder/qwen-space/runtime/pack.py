"""Pack the frozen student + selected head into one self-describing .tde file, and build the
Python (torch) reference model of exactly that quantized format.

    python pack.py --head HEAD.npz --out model.tde [--source q8|float] [--emb int8|fp16] [--dim 1024]
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from common import CKPT, PROJECT, load_student, sha256
from tde import MAGIC, read_model

LIN = ('q', 'k', 'v', 'o', 'fc1', 'fc2')


def codes7(w):
    s = np.maximum(np.abs(w).max(1), 1e-12) / 63
    c = np.clip(np.round(w / s[:, None]), -64, 63).astype(np.int8)
    return c, s.astype(np.float32)


def tile(c):
    N, K = c.shape
    u = (c.astype(np.int16) + 64).astype(np.uint8)
    return np.ascontiguousarray(u.reshape(N // 32, 32, K // 4, 4).transpose(0, 2, 1, 3))


def untile(t, N, K):
    return t.transpose(0, 2, 1, 3).reshape(N, K).astype(np.int16) - 64


def fold_head(h):
    mu, sd = h['mu'].astype(np.float64), h['sd'].astype(np.float64)
    kind = str(h['kind'])
    if kind == 'ridge':
        W = h['W'].astype(np.float64)
        return {'head.skipW': W / sd[:, None], 'head.bias': h['b'] - (mu / sd) @ W}, 0
    W1 = h['fc1.weight'].T.astype(np.float64); S = h['skip.weight'].T.astype(np.float64)
    out = {'head.W1': W1 / sd[:, None], 'head.b1': h['fc1.bias'] - (mu / sd) @ W1,
           'head.W2': h['fc2.weight'].T, 'head.skipW': S / sd[:, None],
           'head.bias': h['fc2.bias'] + h['skip.bias'] - (mu / sd) @ S}
    return out, W1.shape[1]


def build(head_path, out, source='q8', emb='int8', dim=1024, extra_meta=None, mode=None):
    hz = np.load(head_path)
    hmeta = json.loads(str(hz['meta'])) if 'meta' in hz.files else {}
    mode = mode or hmeta.get('mode', 'windowed')
    native = mode == 'native'
    m = load_student(source, native512=native)
    st = {k: v.detach().float().numpy() for k, v in m.state_dict().items()}
    dims = m.dims
    d = dims['d']
    T = {}
    word = st['word.weight']
    if emb == 'int8':
        s = np.maximum(np.abs(word).max(1), 1e-12) / 127
        T['word'] = np.clip(np.round(word / s[:, None]), -127, 127).astype(np.int8)
        T['word_scale'] = s.astype(np.float32)
    else:
        T['word'] = word.astype(np.float16).view(np.uint16)
    T['pos'] = st['pos.weight']
    T['tok_type0'] = st['tok_type.weight'][0]
    T['ln_e_w'], T['ln_e_b'] = st['ln_e.weight'], st['ln_e.bias']
    for l in range(dims['layers']):
        for j in LIN:
            c, s = codes7(st[f'layers.{l}.{j}.weight'])
            T[f'l{l}.{j}.W'] = tile(c); T[f'l{l}.{j}.S'] = s; T[f'l{l}.{j}.B'] = st[f'layers.{l}.{j}.bias']
        for j in ('ln1', 'ln2'):
            T[f'l{l}.{j}_w'] = st[f'layers.{l}.{j}.weight']; T[f'l{l}.{j}_b'] = st[f'layers.{l}.{j}.bias']
    h = dict(hz)
    hm = json.loads(str(h.pop('meta'))) if 'meta' in h else {}
    hd, hidden = fold_head(h)
    T.update(hd)
    T = {k: np.ascontiguousarray(v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in T.items()}
    cfg = {'vocab': dims['vocab'], 'd': d, 'heads': dims['heads'], 'ffn': dims['ffn'], 'layers': dims['layers'],
           'max_pos': dims['max_pos'], 'window': 504 if native else 126, 'out': int(T['head.bias'].shape[0]),
           'head_hidden': hidden, 'emb_int8': int(emb == 'int8'), 'cls': 101, 'sep': 102,
           'max_windows': 1 if native else 4}
    meta = {'format': 'tde-v1', 'source_checkpoint': str(CKPT[source].relative_to(PROJECT)),
            'source_checkpoint_sha256': sha256(CKPT[source]), 'head_file_sha256': sha256(head_path),
            'head': hm, 'weight_format': '7-bit codes in [-64,63], one float scale per output row, stored +64, '
                                         'tiled [N/32][K/4][32][4]; int8 activations, power-of-two scale per '
                                         'token per 256 channels (whole row when K=384)',
            'embedding_format': 'int8 per-row scale' if emb == 'int8' else 'float16',
            'reading_mode': ('native: one sequence of up to 504 content tokens; positions 128-511 from '
                             'bge-micro-v2, untrained' if native else 'windowed: 4 x 126 content tokens'),
            'stored_dimension': dim, 'teacher': 'Qwen/Qwen3-Embedding-8B via DeepInfra',
            'document_contract': '504 WordPiece content tokens, 4 windows of 126, cut never inside a word'}
    meta.update(extra_meta or {})
    header = {'config': cfg, 'metadata': meta, 'tensors': {}}
    off = 0
    order = list(T)
    for k in order:
        header['tensors'][k] = {'dtype': T[k].dtype.str, 'shape': list(T[k].shape), 'offset': 0}
    # two passes: header size depends on offsets
    for _ in range(3):
        hb = json.dumps(header).encode()
        base = (4 + 8 + len(hb) + 63) // 64 * 64
        off = base
        for k in order:
            header['tensors'][k]['offset'] = off
            off = (off + T[k].nbytes + 63) // 64 * 64
    hb = json.dumps(header).encode()
    assert (4 + 8 + len(hb) + 63) // 64 * 64 == base
    out = Path(out)
    with open(out, 'wb') as f:
        f.write(MAGIC); f.write(len(hb).to_bytes(8, 'little')); f.write(hb)
        for k in order:
            f.seek(header['tensors'][k]['offset']); f.write(T[k].tobytes())
        f.truncate(off)
    return out


class Reference:
    """torch reference of a .tde file's exact quantized format (dequantized weights, quant_act activations)."""

    def __init__(self, path):
        from export_pair_precision import ReferenceLinear
        cfg, meta, A = read_model(path)
        self.cfg = cfg
        self.mode = 'native' if cfg['max_windows'] == 1 else 'windowed'
        m = load_student('q8', native512=self.mode == 'native')
        d, F = cfg['d'], cfg['ffn']
        with torch.no_grad():
            if cfg['emb_int8']:
                w = A['word'].astype(np.float32) * A['word_scale'][:, None]
            else:
                w = A['word'].view(np.float16).astype(np.float32)
            m.word.weight.copy_(torch.from_numpy(w))
            m.pos.weight.copy_(torch.from_numpy(np.array(A['pos'])))
            m.tok_type.weight[0].copy_(torch.from_numpy(np.array(A['tok_type0'])))
            m.ln_e.weight.copy_(torch.from_numpy(np.array(A['ln_e_w']))); m.ln_e.bias.copy_(torch.from_numpy(np.array(A['ln_e_b'])))
            KN = {'q': (d, d), 'k': (d, d), 'v': (d, d), 'o': (d, d), 'fc1': (d, F), 'fc2': (F, d)}
            for l, layer in enumerate(m.layers):
                for j in LIN:
                    K, N = KN[j]
                    c = untile(np.array(A[f'l{l}.{j}.W']), N, K).astype(np.float32)
                    W = torch.from_numpy(c * np.array(A[f'l{l}.{j}.S'])[:, None])
                    setattr(layer, j, ReferenceLinear(W, torch.from_numpy(np.array(A[f'l{l}.{j}.B']))))
                for j in ('ln1', 'ln2'):
                    getattr(layer, j).weight.copy_(torch.from_numpy(np.array(A[f'l{l}.{j}_w'])))
                    getattr(layer, j).bias.copy_(torch.from_numpy(np.array(A[f'l{l}.{j}_b'])))
        self.model = m.eval()
        self.A = A

    def head_raw(self, f):
        A = self.A
        out = f @ A['head.skipW'] + A['head.bias']
        if self.cfg['head_hidden']:
            h = torch.nn.functional.gelu(torch.from_numpy(f @ A['head.W1'] + A['head.b1']), approximate='tanh').numpy()
            out = out + h @ A['head.W2']
        return out

    def encode(self, texts_or_cut, mode=None):
        from common import features
        f = features(self.model, texts_or_cut, mode or self.mode)
        raw = self.head_raw(f)
        n = np.linalg.norm(raw, axis=1)
        return raw / n[:, None], n, f


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--head', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--source', default='q8'); ap.add_argument('--emb', default='int8')
    ap.add_argument('--dim', type=int, default=1024); ap.add_argument('--mode', default=None)
    a = ap.parse_args()
    p = build(a.head, a.out, a.source, a.emb, a.dim, mode=a.mode)
    print(p, sha256(p), p.stat().st_size)
