"""Plan B student (PLAN section 2, binding).

Tables: unigram u[t] (width d) with weight w[t] = softplus(theta[t]);
bigram/trigram residual rows r[g] (width 512, first 512 coords of the
representation), exact tables only, an n-gram belongs to the position of its
last token. For a position set S:
    h(S) = (sum_i w[t_i] u[t_i] + sum_{g ending in S} r[g]) / W(S)
Readout: normalize(Vz + mu + MLP(z)); MLP d->2048->4096 GELU, last layer zero
(rungs L2+). Chunks of c tokens (L3+), a K-layer pre-norm mixer of width 512
over chunk vectors with chunk-position embeddings, added through a
zero-initialized projection; pooling z = sum_j a_j h'_j,
a_j ~ W_j exp(s(h'_j)), s zero-initialized.

Table rows are nn.Embedding(sparse=True) for SparseAdam; everything else dense.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 4096
WNG = 512


def inv_softplus(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    return torch.where(x > 20, x, torch.log(torch.expm1(x.clamp_min(1e-12))))


class NgramTable(nn.Module):
    """Exact n-gram table: sorted uint64 keys (stored as int64) -> rows."""

    def __init__(self, keys_ranked: np.ndarray, width=WNG):
        super().__init__()
        keys = torch.as_tensor(keys_ranked.astype(np.int64))
        order = torch.argsort(keys)
        self.register_buffer("skeys", keys[order], persistent=True)
        self.register_buffer("srow", order, persistent=True)   # sorted pos -> row (rank)
        self.rows = nn.Embedding(len(keys), width, sparse=True)
        nn.init.zeros_(self.rows.weight)

    def lookup(self, k: torch.Tensor):
        """k int64 keys -> (row index, found mask)."""
        j = torch.searchsorted(self.skeys, k).clamp_max(len(self.skeys) - 1)
        found = self.skeys[j] == k
        return self.srow[j], found


class Student(nn.Module):
    def __init__(self, vocab=30522, d=1024, n2_keys=None, n3_keys=None, mlp=False,
                 chunk=0, K=0, max_chunks=33):
        super().__init__()
        self.d, self.vocab = d, vocab
        self.u = nn.Embedding(vocab, d, sparse=True)
        self.theta = nn.Embedding(vocab, 1, sparse=True)
        self.n2 = NgramTable(n2_keys) if n2_keys is not None else None
        self.n3 = NgramTable(n3_keys) if n3_keys is not None else None
        self.V = nn.Parameter(torch.zeros(D, d))
        self.mu = nn.Parameter(torch.zeros(D))
        self.mlp = None
        if mlp:
            self.add_mlp()
        self.chunk = chunk
        self.K = 0
        self.max_chunks = max_chunks
        self.pos = None
        self.mixer = None
        self.scorer = None
        if chunk:
            self.add_chunking(chunk)
        for _ in range(K):
            self.add_mixer_layer()

    # ---- ladder growth (each step leaves the function unchanged) ----
    def add_mlp(self):
        self.mlp = nn.Sequential(nn.Linear(self.d, 2048), nn.GELU(), nn.Linear(2048, D))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        dev = self.V.device
        self.mlp.to(dev)

    def add_chunking(self, c):
        """L3: chunks, chunk positions, learned pooling (K = 0). Positions enter
        only through the mixer, so with K = 0 they are inert; the scorer is zero."""
        self.chunk = c
        dev = self.V.device
        self.scorer = nn.Linear(self.d, 1).to(dev)
        nn.init.zeros_(self.scorer.weight)
        nn.init.zeros_(self.scorer.bias)
        self.pos = nn.Parameter(torch.zeros(self.max_chunks, 512, device=dev))
        nn.init.normal_(self.pos, std=0.02)
        self.down = nn.Linear(self.d, 512, bias=False).to(dev) if self.d != 512 else nn.Identity()
        self.up = nn.Linear(512, self.d, bias=False).to(dev)        # zero-init: mixer adds nothing
        nn.init.zeros_(self.up.weight)
        if self.d != 512:
            with torch.no_grad():
                self.down.weight.zero_()
                self.down.weight[:, :512] = torch.eye(512, device=dev)
        self.mixer = nn.ModuleList()

    def add_mixer_layer(self):
        layer = nn.TransformerEncoderLayer(512, 8, 2048, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True).to(self.V.device)
        # zero output projections: a new pre-norm layer is the identity, so the
        # rung computes the same function at its first step
        for lin in (layer.self_attn.out_proj, layer.linear2):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.mixer.append(layer)
        self.K = len(self.mixer)

    # ---- forward ----
    def weights(self, ids):
        return F.softplus(self.theta(ids)).squeeze(-1)

    def sums(self, ids, seg, nseg, drop=None, n2k=None, n3k=None):
        """Per-segment numerator (nseg, d) and W (nseg,).
        ids: (P,) token ids; seg: (P,) segment index of each position;
        n2k/n3k: (P,) int64 keys of the n-gram ending at that position, -1 if none.
        drop: (P,) bool, dropped positions (unigram and any n-gram covering them)."""
        w = self.weights(ids)
        keep = torch.ones_like(w, dtype=torch.bool) if drop is None else ~drop
        wk = w * keep
        num = torch.zeros(nseg, self.d, device=ids.device, dtype=torch.float32)
        num.index_add_(0, seg, wk[:, None] * self.u(ids))
        W = torch.zeros(nseg, device=ids.device).index_add_(0, seg, wk)
        for tab, keys, n in ((self.n2, n2k, 2), (self.n3, n3k, 3)):
            if tab is None or keys is None:
                continue
            valid = keys >= 0
            if drop is not None:
                cov = drop.clone()
                for s in range(1, n):
                    cov[s:] |= drop[:-s]
                valid &= ~cov
            p = torch.nonzero(valid).squeeze(1)
            if len(p) == 0:
                continue
            row, found = tab.lookup(keys[p])
            p, row = p[found], row[found]
            r = tab.rows(row)
            num[:, :WNG].index_add_(0, seg[p], r)
        return num, W

    def readout(self, z):
        y = z @ self.V.T + self.mu
        if self.mlp is not None:
            y = y + self.mlp(z)
        return y

    def forward(self, batch, drop=None):
        """batch: dict with ids, doc (P,), pos_in_doc (P,), ndoc, n2k, n3k, [offset]
        Returns unnormalized 4096-d outputs (ndoc, 4096) and z."""
        ids, doc, ndoc = batch["ids"], batch["doc"], batch["ndoc"]
        if not self.chunk:
            num, W = self.sums(ids, doc, ndoc, drop, batch.get("n2k"), batch.get("n3k"))
            z = num / W.clamp_min(1e-6)[:, None]
            return self.readout(z), z
        c = self.chunk
        off = batch.get("offset")
        p = batch["pos_in_doc"] + (off[doc] if off is not None else 0)
        cj = torch.div(p, c, rounding_mode="floor")       # chunk index (offset grid)
        nch = self.max_chunks          # 504 tokens, c >= 16, offset < c -> at most 33 chunks
        cj = cj.clamp_max(nch - 1)
        seg = doc * nch + cj
        num, W = self.sums(ids, seg, ndoc * nch, drop, batch.get("n2k"), batch.get("n3k"))
        num = num.view(ndoc, nch, self.d)
        W = W.view(ndoc, nch)
        present = W > 0
        h = num / W.clamp_min(1e-6)[..., None]
        if self.K:
            x = self.down(h)
            x = x + self.pos.unsqueeze(0)
            pad = ~present
            for layer in self.mixer:
                x = layer(x, src_key_padding_mask=pad)
            h = h + self.up(x)
        s = self.scorer(h).squeeze(-1)
        logit = torch.where(present, torch.log(W.clamp_min(1e-12)) + s, torch.full_like(s, -1e30))
        a = torch.softmax(logit, dim=1)
        z = (a[..., None] * h).sum(1)
        return self.readout(z), z


def prefix_cos_loss(pred, target, prefixes=(256, 512, 1024, 4096), per_item=False):
    """Mean over prefixes of 1 - cos(prefix(pred), prefix(target))."""
    ls = []
    for k in prefixes:
        ls.append(1 - F.cosine_similarity(pred[:, :k], target[:, :k], dim=-1))
    l = torch.stack(ls).mean(0)
    return l if per_item else l.mean()


# ---- batching ----------------------------------------------------------------

def make_batch(seqs, device, offsets=None):
    """seqs: list of int arrays (token ids). Returns the dict forward() needs."""
    lens = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
    flat = np.concatenate(seqs).astype(np.int64) if len(seqs) else np.zeros(0, np.int64)
    doc = np.repeat(np.arange(len(seqs)), lens)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    pos = np.arange(len(flat)) - np.repeat(starts, lens)
    n2k = np.full(len(flat), -1, np.int64)
    n3k = np.full(len(flat), -1, np.int64)
    m2 = pos >= 1
    m3 = pos >= 2
    i = np.nonzero(m2)[0]
    n2k[i] = (flat[i - 1] << 15) | flat[i]
    i = np.nonzero(m3)[0]
    n3k[i] = (flat[i - 2] << 30) | (flat[i - 1] << 15) | flat[i]
    t = lambda a: torch.from_numpy(a).to(device, non_blocking=True)
    b = {"ids": t(flat), "doc": t(doc), "pos_in_doc": t(pos), "ndoc": len(seqs),
         "n2k": t(n2k), "n3k": t(n3k)}
    if offsets is not None:
        b["offset"] = t(np.asarray(offsets, np.int64))
    return b
