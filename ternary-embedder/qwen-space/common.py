"""Plan A helpers: student loading, feature extraction (single window, windowed,
native-512 probe), heads, metric wrappers and small IO utilities."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

QS = Path(__file__).resolve().parent
PROJECT = QS.parent
WORK = PROJECT / 'training-v2/work/qwen-space'
SHARED_WORK = PROJECT / 'training-v2/work/shared-qwen8b'
for p in ('tern-embed', 'training-v2', 'retrieval-diagnostics', 'shared-qwen8b'):
    sys.path.insert(0, str(PROJECT / p))

import shared  # noqa: E402
from model import Student  # noqa: E402

CKPT = {'q8': PROJECT / 'models/campaign-20261025/model-q8.pt',
        'float': PROJECT / 'models/campaign-20261025/model-float.pt',
        'q4': PROJECT / 'models/campaign-20261025/model-q4.pt'}
BGE_MICRO = next((PROJECT / 'cache/hub/models--TaylorAI--bge-micro-v2/snapshots').iterdir())
WINDOW = 126
CLS, SEP = 101, 102


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def load_student(precision: str = 'q8', native512: bool = False) -> Student:
    if precision in ('q8', 'q4'):
        from export_pair_precision import load_reference
        m = load_reference(CKPT[precision])
    else:
        m = Student.load(CKPT[precision]).eval()
    if native512:
        from safetensors.torch import load_file
        st = load_file(str(BGE_MICRO / 'model.safetensors'))
        key = next(k for k in st if k.endswith('position_embeddings.weight'))
        table = st[key]
        assert table.shape == (512, 384), table.shape
        new = torch.nn.Embedding(512, 384)
        with torch.no_grad():
            new.weight[:128] = m.pos.weight
            new.weight[128:] = table[128:512]
        m.pos = new
        m.dims = dict(m.dims, max_pos=512)
    return m.eval()


def tokenize(texts, content_tokens=504):
    """Content-token ids (no specials) of each text, first `content_tokens` kept.
    Callers pass texts already cut to the contract, so the cap is a guard."""
    tk = shared._tokenizer()
    out = []
    for i in range(0, len(texts), 4096):
        for e in tk.encode_batch(list(texts[i:i + 4096]), add_special_tokens=False):
            out.append(e.ids[:content_tokens])
    return out


@torch.inference_mode()
def _run(model, seqs, batch_tokens=32768):
    """Mean-pooled last states (384-d, before `out`) for each sequence of ids (with specials)."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = np.zeros((len(seqs), 384), np.float32)
    i = 0
    while i < len(order):
        L = max(len(seqs[order[i]]), 1)
        # grow batch while padded size fits
        j = i
        while j < len(order) and (j - i + 1) * len(seqs[order[j]]) <= batch_tokens:
            j += 1
        j = max(j, i + 1)
        idx = order[i:j]
        T = len(seqs[idx[-1]])
        ids = torch.zeros((len(idx), T), dtype=torch.long)
        mask = torch.zeros((len(idx), T), dtype=torch.bool)
        for r, k in enumerate(idx):
            s = seqs[k]
            ids[r, :len(s)] = torch.tensor(s)
            mask[r, :len(s)] = True
        states, _, _ = model(ids, mask)
        x = states[-1]
        m = mask.unsqueeze(-1).float()
        out[idx] = ((x * m).sum(1) / m.sum(1).clamp(min=1)).numpy()
        i = j
    return out


def features(model, texts, mode='windowed', content_tokens=504, return_counts=False):
    """Pooled 384-d features.

    mode 'single'  : one window of the first 126 content tokens (the 128 contract).
    mode 'windowed': windows of 126 content tokens, each with [CLS]/[SEP], no overlap;
                     token-count-weighted mean of window means (count incl. specials).
    mode 'native'  : one sequence of up to `content_tokens` + 2 tokens (needs max_pos >= 506).
    """
    toks = tokenize(texts, content_tokens)
    if mode == 'single':
        seqs = [[CLS] + t[:WINDOW] + [SEP] for t in toks]
        f = _run(model, seqs)
    elif mode == 'native':
        seqs = [[CLS] + t + [SEP] for t in toks]
        f = _run(model, seqs, batch_tokens=16384)
    elif mode == 'windowed':
        seqs, owner, weight = [], [], []
        for d, t in enumerate(toks):
            chunks = [t[i:i + WINDOW] for i in range(0, max(len(t), 1), WINDOW)] or [[]]
            for c in chunks:
                seqs.append([CLS] + c + [SEP]); owner.append(d); weight.append(len(c) + 2)
        w = _run(model, seqs)
        owner = np.asarray(owner); weight = np.asarray(weight, np.float32)
        f = np.zeros((len(toks), 384), np.float32)
        np.add.at(f, owner, w * weight[:, None])
        f /= np.bincount(owner, weights=weight, minlength=len(toks))[:, None].astype(np.float32)
    else:
        raise ValueError(mode)
    if return_counts:
        return f, np.array([len(t) for t in toks])
    return f


def student_vectors(model, feats):
    """SS384: the unchanged student's own output (out projection, unit norm)."""
    w = model.out.weight.detach().numpy()
    v = feats @ w.T
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def unit(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def prefix(x, k):
    return unit(np.asarray(x[..., :k], np.float32))


def evaluate(scores, d):
    from round2_metrics import evaluate as ev
    return ev(scores, d)


def paired(left_rows, right_rows):
    from diagnose import paired as pr
    return pr(left_rows, right_rows)


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.perf_counter(); return self

    def __exit__(self, *a):
        print(f'[{self.label}] {time.perf_counter() - self.t:.1f}s', flush=True)
