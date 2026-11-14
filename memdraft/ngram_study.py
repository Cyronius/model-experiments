#!/usr/bin/env python3
"""Offline study: what would a free n-gram table drafter get on a model's own output?

No model runs here. The input is text a model already wrote, with the model's own top choice
stored at every position (manifold-compression's heal shards: Qwen3.5-9B answers with its top-20
log-probs). From the training shards we count, for the last 1..K tokens, which token the model
chose next. The table drafts from those counts; the held-out shard says how often the model would
have accepted the draft.

Two drafters are scored, alone and together:
  static   the table built ahead of time from other requests (the thing MemDraft never had)
  context  the same lookup against the current request only (prompt lookup with short keys)

Two ways to score a draft token:
  greedy   it equals the model's top choice at that position (temperature 0)
  sampled  it equals the token the sampler actually wrote (the shards were sampled at T=1.0)

Speed is not measured here. A decode step that carries n draft tokens is charged 1 + beta*n plain
steps (plus, with --cost, a charge for a rejected draft and for a changed batch shape); the lookup
is charged nothing. The charges have to come from timing the real target (live_bench.py with
--spec-table-const). The printed speedups are what that cost model predicts.

    python ngram_study.py --shards DIR0 DIR1 DIR2 --out results/study.json
"""
import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

U = np.uint64
GOLD, M1, M2 = U(0x9E3779B97F4A7C15), U(0xBF58476D1CE4E5B9), U(0x94D049BB133111EB)
LBITS = U(18)                       # label bits packed under the key hash (vocab < 2**18)
LMASK = U((1 << 18) - 1)
PBITS = U(13)                       # position bits in the in-request index (sequences < 8192)
PMASK = U((1 << 13) - 1)
CT_EDGES = np.array([1, 2, 3, 4, 6, 10, 20, 50, 200, 1000])   # total-count buckets
N_RATIO = 12                        # ratio buckets: 0..9 tenths, 10 = [0.95, 1), 11 = exactly 1
CC_MAX = 6                          # in-request prior-occurrence count is capped here
TAUS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BETAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def mix(h, t):
    """One more token folded into a context hash (splitmix64 finalizer)."""
    x = h + (t.astype(np.uint64) + U(1)) * GOLD
    x ^= x >> U(30)
    x *= M1
    x ^= x >> U(27)
    x *= M2
    x ^= x >> U(31)
    return x


# ---------------------------------------------------------------- data

class Data:
    """Token streams of many requests, laid end to end.

    tok[j] is the token at j, pos[j] its index inside its request, seq[j] the request. written[j]
    marks answer tokens; amax[j] is the model's top choice for position j, or -1 when not stored.
    """

    def __init__(self, rows):
        n = sum(len(r['prompt_ids']) + len(r['answer_ids']) for r in rows)
        self.tok = np.empty(n, np.int32)
        self.pos = np.empty(n, np.int32)
        self.seq = np.empty(n, np.int32)
        self.written = np.zeros(n, bool)
        self.amax = np.full(n, -1, np.int32)
        self.start = np.empty(len(rows), np.int64)
        self.ans = np.empty(len(rows), np.int64)
        self.end = np.empty(len(rows), np.int64)
        self.domain = []
        j = 0
        for s, r in enumerate(rows):
            p, a = r['prompt_ids'], r['answer_ids']
            m = len(p) + len(a)
            self.tok[j:j + len(p)] = p
            self.tok[j + len(p):j + m] = a
            self.pos[j:j + m] = np.arange(m)
            self.seq[j:j + m] = s
            self.written[j + len(p):j + m] = True
            if r.get('amax') is not None:
                self.amax[j + len(p):j + m] = r['amax']
            self.start[s], self.ans[s], self.end[s] = j, j + len(p), j + m
            self.domain.append(f"{r['source']}/{'think' if r['thinking'] else 'plain'}")
            j += m

    def context(self, idx, K):
        """The K tokens before each position in idx, most recent first, and how many are real."""
        C = np.stack([self.tok[idx - i] for i in range(1, K + 1)], axis=1)
        return C, np.minimum(self.pos[idx], K).astype(np.int32)


def load_shard(folder, limit=None, cache=None):
    """Rows of one shard; finished rows carry 'amax', the teacher's top choice per answer token.

    The top choices sit in large compressed arrays; with a cache folder they are unpacked once."""
    folder = Path(folder)
    rows = [json.loads(line) for line in (folder / 'samples.jsonl').open()]
    for i, r in enumerate(rows):
        r['row'] = i
    if limit:
        rows = rows[:limit]
    saved = Path(cache) / f'{folder.resolve().parent.parent.name}-{folder.name}-top1.npz' if cache and not limit else None
    if saved and saved.exists():
        z = np.load(saved)
        tops = dict(zip(z['rows'].tolist(), np.split(z['top'], z['ends'][:-1])))
    else:
        tops, chunks = {}, {}
        for r in rows:
            if not r['finished']:
                continue
            if r['chunk'] not in chunks:
                chunks[r['chunk']] = np.load(folder / f"teacher_{r['chunk']}.npz")
            tops[r['row']] = chunks[r['chunk']][f"ids_{r['row']}"][:, 0]
        if saved:
            saved.parent.mkdir(parents=True, exist_ok=True)
            keys = sorted(tops)
            np.savez(saved, rows=np.array(keys), ends=np.cumsum([len(tops[k]) for k in keys]),
                     top=np.concatenate([tops[k] for k in keys]))
    for r in rows:
        r['amax'] = tops.get(r['row'])
        if r['amax'] is not None and len(r['amax']) != len(r['answer_ids']):
            raise RuntimeError(f"row {r['row']}: {len(r['amax'])} teacher rows, {len(r['answer_ids'])} tokens")
    return rows


# ---------------------------------------------------------------- the static table

class Static:
    """For each order k: sorted context hashes with the most frequent next token and its counts."""

    min_count = 1                   # entries seen fewer times than this are not proposed

    def __init__(self, K):
        self.K = K
        self.keys, self.best, self.cbest, self.ctot = [], [], [], []

    @classmethod
    def build(cls, data, idx, labels, K):
        """Count labels[i] after the context of position idx[i], for every order 1..K."""
        self = cls(K)
        h = np.zeros(len(idx), np.uint64)
        for k in range(1, K + 1):
            h = mix(h, data.tok[idx - k])
            ok = data.pos[idx] >= k
            packed = ((h[ok] >> LBITS) << LBITS) | labels[ok].astype(np.uint64)
            packed.sort()
            first = np.flatnonzero(np.r_[True, packed[1:] != packed[:-1]])
            count = np.diff(np.r_[first, len(packed)])
            pair_key, pair_lab = packed[first] >> LBITS, (packed[first] & LMASK).astype(np.int32)
            head = np.flatnonzero(np.r_[True, pair_key[1:] != pair_key[:-1]])
            total = np.add.reduceat(count, head)
            top = np.maximum.reduceat(count, head)
            group = np.repeat(np.arange(len(head)), np.diff(np.r_[head, len(pair_key)]))
            winners = np.flatnonzero(count == top[group])
            winners = winners[np.r_[True, group[winners][1:] != group[winners][:-1]]]
            self.keys.append(pair_key[head])
            self.best.append(pair_lab[winners])
            self.cbest.append(top.astype(np.int64))
            self.ctot.append(total.astype(np.int64))
        return self

    def lookup(self, k, h):
        """(hit, token, best count, total count) for context hashes h at order k."""
        keys = self.keys[k - 1]
        q = h >> LBITS
        i = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
        hit = keys[i] == q
        return hit, self.best[k - 1][i], self.cbest[k - 1][i], self.ctot[k - 1][i]

    def propose(self, k, h):
        """(hit, token, calibration cell) for context hashes h at order k."""
        hit, token, cbest, ctot = self.lookup(k, h)
        return hit & (ctot >= self.min_count), token, static_cell(cbest, ctot)


def static_cell(cbest, ctot):
    """Calibration cell of a table entry: a total-count bucket and a ratio bucket."""
    ratio = cbest / np.maximum(ctot, 1)
    rb = np.minimum((ratio * 10).astype(np.int64), 9)
    rb = np.where(ratio >= 0.95, 10, rb)
    rb = np.where(cbest == ctot, 11, rb)
    return np.searchsorted(CT_EDGES, ctot, side='right') - 1, rb


# ---------------------------------------------------------------- the shipped table (memdraft.h)

MAGIC, STREAM_MAGIC = b'MDTABLE1', b'MDSTREAM'
BUCKET = 8                          # slots per bucket: one cache line, so one miss per lookup


def export_table(static, calib, path, floor, load=0.5):
    """Write the table the C++ drafter reads: per context, the token and its confidence in 255ths.

    Only entries the drafter could use are kept (confidence at least floor). A bucket holds 8;
    when more land in one, the least useful (confidence x times seen) are dropped. Returns
    (entries kept, entries dropped for space)."""
    key, token, conf, weight = [], [], [], []
    for k in range(1, static.K + 1):
        cbest, ctot = static.cbest[k - 1], static.ctot[k - 1]
        c = calib.confidence('static', k, static_cell(cbest, ctot))
        q = np.round(c * 255).astype(np.uint64)
        keep = (c >= floor) & (q > 0) & (ctot >= static.min_count)
        key.append(mix(static.keys[k - 1][keep], np.full(int(keep.sum()), k, np.uint64)))
        token.append(static.best[k - 1][keep].astype(np.uint64))
        conf.append(q[keep])
        weight.append(c[keep] * ctot[keep])
    key, token, conf, weight = (np.concatenate(x) for x in (key, token, conf, weight))
    n_buckets = 1 << max(4, int(np.ceil(np.log2(max(1, len(key)) / (BUCKET * load)))))
    bucket = (key & U(n_buckets - 1)).astype(np.int64)
    order = np.lexsort((-weight, bucket))
    sorted_bucket = bucket[order]
    first = np.flatnonzero(np.r_[True, sorted_bucket[1:] != sorted_bucket[:-1]])
    rank = np.arange(len(key)) - np.repeat(first, np.diff(np.r_[first, len(key)]))
    fits = rank < BUCKET
    chosen = order[fits]
    slots = np.zeros(n_buckets * BUCKET, np.uint64)
    slots[sorted_bucket[fits] * BUCKET + rank[fits]] = ((key[chosen] >> U(32)) << U(32)) | (token[chosen] << U(8)) | conf[chosen]
    seen = np.array([[calib.confidence('context', k, (np.array([c]),))[0] for c in range(CC_MAX + 1)]
                     for k in range(1, static.K + 1)])
    seen = np.round(seen * 255).astype(np.uint8).tobytes()
    with open(path, 'wb') as f:
        f.write(struct.pack('<8sIIIIQQ24x', MAGIC, 1, static.K, CC_MAX + 1, BUCKET, n_buckets, len(chosen)))
        f.write(seen + b'\0' * (-len(seen) % 8))
        f.write(slots.tobytes())
    return len(chosen), int(len(key) - len(chosen))


def export_table2(static, calib, path, load=0.5):
    """Version 2 of the table file, for the reranker: per context the token and its calibration
    cell (count bucket x ratio bucket), and per order the confidence of every cell as a float.

    Every entry seen at least static.min_count times is kept: the reranker uses weak entries too.
    A bucket holds 8; when more land in one, the least useful (confidence x times seen) are
    dropped. Returns (entries kept, entries dropped for space)."""
    n_ct = len(CT_EDGES)
    grid = (np.repeat(np.arange(n_ct), N_RATIO), np.tile(np.arange(N_RATIO), n_ct))
    key, token, code, weight = [], [], [], []
    for k in range(1, static.K + 1):
        cbest, ctot = static.cbest[k - 1], static.ctot[k - 1]
        ct, rb = static_cell(cbest, ctot)
        c = calib.confidence('static', k, (ct, rb))
        keep = ctot >= static.min_count
        key.append(mix(static.keys[k - 1][keep], np.full(int(keep.sum()), k, np.uint64)))
        token.append(static.best[k - 1][keep].astype(np.uint64))
        code.append((ct * N_RATIO + rb)[keep].astype(np.uint64))
        weight.append(c[keep] * ctot[keep])
    key, token, code, weight = (np.concatenate(x) for x in (key, token, code, weight))
    n_buckets = 1 << max(4, int(np.ceil(np.log2(max(1, len(key)) / (BUCKET * load)))))
    bucket = (key & U(n_buckets - 1)).astype(np.int64)
    order = np.lexsort((-weight, bucket))
    sorted_bucket = bucket[order]
    first = np.flatnonzero(np.r_[True, sorted_bucket[1:] != sorted_bucket[:-1]])
    rank = np.arange(len(key)) - np.repeat(first, np.diff(np.r_[first, len(key)]))
    fits = rank < BUCKET
    chosen = order[fits]
    slots = np.zeros(n_buckets * BUCKET, np.uint64)
    slots[sorted_bucket[fits] * BUCKET + rank[fits]] = ((key[chosen] >> U(32)) << U(32)) | (token[chosen] << U(8)) | code[chosen]
    seen = np.array([[calib.confidence('context', k, (np.array([c]),))[0] for c in range(CC_MAX + 1)]
                     for k in range(1, static.K + 1)], np.float32)
    cells = np.array([calib.confidence('static', k, grid) for k in range(1, static.K + 1)], np.float32)
    head = seen.tobytes() + cells.tobytes()
    with open(path, 'wb') as f:
        f.write(struct.pack('<8sIIIIQQII16x', MAGIC, 2, static.K, CC_MAX + 1, BUCKET, n_buckets, len(chosen), n_ct * N_RATIO, N_RATIO))
        f.write(head + b'\0' * (-len(head) % 8))
        f.write(slots.tobytes())
    return len(chosen), int(len(key) - len(chosen))


class Exported2:
    """A version 2 table read back: proposes what memdraft.h proposes, with the calibration cell
    of each entry, and stands in for the calibration (the confidences are in the file)."""

    def __init__(self, path):
        raw = np.fromfile(path, np.uint8)
        magic, version, self.K, n_seen, bucket, self.n_buckets, self.n_entries, n_cells, n_ratio = \
            struct.unpack('<8sIIIIQQII', raw[:48].tobytes())
        if magic != MAGIC or version != 2 or bucket != BUCKET or n_ratio != N_RATIO:
            raise ValueError(f'{path} is not a version 2 MemDraft table')
        at = 64
        self.seen = raw[at:at + 4 * self.K * n_seen].view(np.float32).reshape(self.K, n_seen)
        at += 4 * self.K * n_seen
        self.cells = raw[at:at + 4 * self.K * n_cells].view(np.float32).reshape(self.K, n_cells)
        at += 4 * self.K * n_cells
        at += -at % 8
        self.slots = raw[at:].view(np.uint64).reshape(self.n_buckets, BUCKET)

    def propose(self, k, h):
        g = mix(h >> LBITS, np.full(len(h), k, np.uint64))
        rows = self.slots[(g & U(self.n_buckets - 1)).astype(np.int64)]
        match = (rows != 0) & ((rows >> U(32)) == (g >> U(32))[:, None])
        slot = rows[np.arange(len(g)), match.argmax(1)]
        code = (slot & U(0xFF)).astype(np.int64)
        return match.any(1), ((slot >> U(8)) & U(0xFFFFFF)).astype(np.int32), (code // N_RATIO, code % N_RATIO)

    def confidence(self, kind, k, cell):
        if kind == 'static':
            return self.cells[k - 1][cell[0] * N_RATIO + cell[1]]
        return self.seen[k - 1][cell[0]]


class Exported:
    """A shipped table read back. It proposes exactly what memdraft.h proposes, and it stands in
    for the calibration too, because the confidences are already in the file."""

    def __init__(self, path):
        raw = np.fromfile(path, np.uint8)
        magic, version, self.K, n_seen, bucket, self.n_buckets, self.n_entries = struct.unpack('<8sIIIIQQ', raw[:40].tobytes())
        if magic != MAGIC or version != 1 or bucket != BUCKET:
            raise ValueError(f'{path} is not a MemDraft table')
        at = 64 + (self.K * n_seen + 7) // 8 * 8
        self.seen = raw[64:64 + self.K * n_seen].reshape(self.K, n_seen)
        self.slots = raw[at:].view(np.uint64).reshape(self.n_buckets, BUCKET)

    def propose(self, k, h):
        g = mix(h >> LBITS, np.full(len(h), k, np.uint64))
        rows = self.slots[(g & U(self.n_buckets - 1)).astype(np.int64)]
        match = (rows != 0) & ((rows >> U(32)) == (g >> U(32))[:, None])
        slot = rows[np.arange(len(g)), match.argmax(1)]
        return match.any(1), ((slot >> U(8)) & U(0xFFFFFF)).astype(np.int32), ((slot & U(0xFF)).astype(np.int64),)

    def confidence(self, kind, k, cell):
        q = cell[0] if kind == 'static' else self.seen[k - 1][cell[0]]
        return q.astype(np.float32) / np.float32(255)


def write_stream(data, path):
    """The requests of `data` for replay.cpp: per request the prompt and answer tokens, then the
    model's top choice for each answer token."""
    with open(path, 'wb') as f:
        f.write(struct.pack('<8sII', STREAM_MAGIC, len(data.start), 0))
        for s in range(len(data.start)):
            a, b, c = data.start[s], data.ans[s], data.end[s]
            f.write(struct.pack('<II', b - a, c - b))
            f.write(data.tok[a:c].astype('<i4').tobytes())
            f.write(data.amax[b:c].astype('<i4').tobytes())


# ---------------------------------------------------------------- the in-request index

class InRequest:
    """For each order k: every (request, context hash) with the position it was seen at, sorted,
    so the latest earlier occurrence of a context inside the same request is one binary search."""

    def __init__(self, data, K):
        self.data, self.K, self.index = data, K, []
        everything = np.arange(len(data.tok))
        h = np.zeros(len(everything), np.uint64)
        for k in range(1, K + 1):
            h = mix(h, data.tok[everything - k])
            ok = data.pos >= k
            packed = self.pack(h[ok], data.seq[ok], data.pos[ok])
            packed.sort()
            self.index.append(packed)

    @staticmethod
    def pack(h, seq, pos):
        return ((mix(h, seq) >> PBITS) << PBITS) | pos.astype(np.uint64)

    def lookup(self, k, h, seq, pos):
        """(hit, token that followed, number of earlier occurrences) strictly before pos."""
        index = self.index[k - 1]
        x = self.pack(h, seq, pos)
        i = np.searchsorted(index, x) - 1
        j = np.maximum(i, 0)
        hit = (i >= 0) & ((index[j] >> PBITS) == (x >> PBITS))
        where = self.data.start[seq] + (index[j] & PMASK).astype(np.int64)
        first = np.searchsorted(index, (x >> PBITS) << PBITS)
        return hit, self.data.tok[np.where(hit, where, 0)], np.minimum(i - first + 1, CC_MAX)


# ---------------------------------------------------------------- candidates and calibration

def candidates(C, avail, seq, pos, static, inreq, K):
    """Every proposal for a batch of contexts: a list of (kind, k, hit, token, cell)."""
    out = []
    h = np.zeros(len(C), np.uint64)
    for k in range(1, K + 1):
        h = mix(h, C[:, k - 1])
        real = avail >= k
        if static is not None:
            hit, token, cell = static.propose(k, h)
            out.append(('static', k, hit & real, token, cell))
        if inreq is not None:
            hit, token, seen = inreq.lookup(k, h, seq, pos)
            out.append(('context', k, hit & real, token, (seen,)))
    return out


class Calibration:
    """How often a proposal of a given kind, order and cell was right on held-out requests."""

    def __init__(self, K):
        self.hit = {('static', k): np.zeros((len(CT_EDGES), N_RATIO)) for k in range(1, K + 1)}
        self.hit.update({('context', k): np.zeros(CC_MAX + 1) for k in range(1, K + 1)})
        self.n = {key: np.zeros_like(v) for key, v in self.hit.items()}

    def add(self, cands, truth):
        for kind, k, hit, token, cell in cands:
            where = tuple(c[hit] for c in cell)
            np.add.at(self.n[kind, k], where, 1)
            np.add.at(self.hit[kind, k], where, token[hit] == truth[hit])

    def confidence(self, kind, k, cell):
        """Smoothed hit rate of the cell; a cell seen under 30 times is trusted less."""
        n, hit = self.n[kind, k][cell], self.hit[kind, k][cell]
        return ((hit + 0.5) / (n + 2.0) * np.minimum(n, 30) / 30).astype(np.float32)


def choose(cands, calib, min_conf=0.0):
    """The most trusted proposal per context: (token, confidence); confidence 0 means none."""
    n = len(cands[0][2])
    token, conf = np.full(n, -1, np.int32), np.zeros(n, np.float32)
    for kind, k, hit, tok, cell in cands:
        c = np.where(hit, calib.confidence(kind, k, cell), 0)
        better = c >= np.maximum(conf, 1e-9)        # ties go to the longer context
        token, conf = np.where(better, tok, token), np.where(better, c, conf)
    conf[conf < min_conf] = 0
    return token, conf


# ---------------------------------------------------------------- drafting and scoring

def draft(data, idx, static, inreq, calib, K, n_max, floor, pad_to=0):
    """A draft chain for every position: tokens [n, n_max] and the running chance all of the
    chain so far is accepted. A chain ends when that chance falls under floor.

    The first pad_to tokens are drafted whatever their chance: the best guess, or a repeat of the
    last token when there is no guess. That is memdraft.h's `pad`, which keeps the batch shape
    fixed; the chance recorded for such tokens is below floor, often 0."""
    n = len(idx)
    tokens = np.full((n, n_max), -1, np.int32)
    chance = np.zeros((n, n_max), np.float32)
    C, avail = data.context(idx, K)
    seq, pos = data.seq[idx], data.pos[idx]
    alive, running = np.arange(n), np.ones(n, np.float32)
    for d in range(n_max):
        cands = candidates(C[alive], avail[alive], seq[alive], pos[alive], static, inreq, K)
        token, conf = choose(cands, calib)
        if d < pad_to:
            token = np.where(conf > 0, token, C[alive, 0])
        now = running[alive] * conf
        keep = (now >= floor) | (d < pad_to)
        alive, token, now = alive[keep], token[keep], now[keep]
        if not len(alive):
            break
        tokens[alive, d], chance[alive, d], running[alive] = token, now, now
        C[alive, 1:] = C[alive, :-1]
        C[alive, 0] = token
        avail[alive] = np.minimum(avail[alive] + 1, K)
    return tokens, chance


def score(data, idx, tokens, chance, mode):
    """Per position: how many leading draft tokens the target accepts, and how many can be judged.

    greedy: a draft token is right if it is the model's top choice. The stored text was sampled,
    so once the written token differs from the top choice the rest of the chain cannot be judged
    and is cut there (it counts as neither drafted nor accepted).
    """
    n, n_max = tokens.shape
    truth = data.amax if mode == 'greedy' else data.tok
    right = np.zeros((n, n_max), bool)
    judged = np.zeros((n, n_max), bool)
    on_path = np.ones(n, bool)
    for d in range(n_max):
        at = idx + d
        inside = at < data.end[data.seq[idx]]
        at = np.where(inside, at, idx)
        known = inside & (truth[at] >= 0) & on_path
        judged[:, d] = known
        right[:, d] = known & (tokens[:, d] == truth[at])
        if mode == 'greedy':
            on_path = known & (data.tok[at] == data.amax[at])
        else:
            on_path = known
    accepted = np.cumprod(right, axis=1).sum(1)
    return accepted.astype(np.int32), judged.sum(1).astype(np.int32)


COUNTS = ('tokens', 'steps', 'drafted', 'accepted', 'rejected', 'reshaped')


def walk(data, idx, accepted, drafted):
    """Decode every request step by step: a step drafts drafted[j] tokens at j, the target keeps
    accepted[j] of them and adds its own next token.

    Returns per-request counts. 'rejected' is the steps that lost at least one draft token (they
    pay for a rollback); 'reshaped' is the steps whose draft length differs from the step before
    (the target cannot reuse its previous graph)."""
    seqs = np.unique(data.seq[idx])
    where = np.full(len(data.tok), -1, np.int64)
    where[idx] = np.arange(len(idx))
    first = np.array([idx[np.searchsorted(idx, data.ans[s])] for s in seqs])
    last = np.array([idx[np.searchsorted(idx, data.end[s]) - 1] for s in seqs])
    out = {k: np.zeros(len(seqs), np.int64) for k in COUNTS}
    out['seqs'], out['tokens'] = seqs, last - first + 1
    cur, shape = first.copy(), np.full(len(seqs), -1, np.int64)
    going = np.arange(len(seqs))
    while len(going):
        i = where[cur[going]]
        d, a = drafted[i], accepted[i]
        out['steps'][going] += 1
        out['drafted'][going] += d
        out['accepted'][going] += a
        out['rejected'][going] += a < d
        out['reshaped'][going] += d != shape[going]
        shape[going] = d
        cur[going] += a + 1
        going = going[cur[going] <= last[going]]
    return out


def sweep(data, idx, tokens, chance, mode, pad_to=0):
    """The counts of walk() for every drafting rule tried:
    ('tau', t)  draft while the chain's chance of full acceptance is at least t;
    ('pad', n)  always draft exactly n tokens (needs chains drafted with pad_to >= n)."""
    accepted_all, judged = score(data, idx, tokens, chance, mode)
    out = {}
    for tau in TAUS:
        drafted = np.minimum((chance >= tau).sum(1).astype(np.int32), judged)
        out['tau', tau] = walk(data, idx, np.minimum(accepted_all, drafted), drafted)
    for n in range(1, pad_to + 1):
        drafted = np.minimum(np.int32(n), judged)
        out['pad', n] = walk(data, idx, np.minimum(accepted_all, drafted), drafted)
    return out


def cost_name(cost):
    extra = f"+rej{cost['reject']}+shape{cost['reshape']}" if cost['reject'] or cost['reshape'] else ''
    return cost.get('name') or f"b{cost['beta']}{extra}"


def summarize(data, runs, pick, costs):
    """Speedup tables. A step is charged 1 + `beta` per draft token + `accept` per accepted draft
    token (it is sampled and sent like any other) + `reject` if the step lost a draft token +
    `reshape` if its draft length changed; plain decoding is charged 1 per token. For each cost,
    the rule is chosen on `pick` (the calibration requests) and scored on `runs`."""
    rows = {}
    for cost in costs:
        def speed(run, mask=None):
            t = {k: float(run[k][slice(None) if mask is None else mask].sum()) for k in COUNTS}
            spent = (t['steps'] + cost['beta'] * t['drafted'] + cost.get('accept', 0.0) * t['accepted']
                     + cost['reject'] * t['rejected'] + cost['reshape'] * t['reshaped'])
            return {'speedup': t['tokens'] / spent, 'tokens_per_step': t['tokens'] / t['steps'],
                    'drafted_per_step': t['drafted'] / t['steps'], 'acceptance': t['accepted'] / max(1.0, t['drafted']),
                    'rejected_per_step': t['rejected'] / t['steps'], 'reshaped_per_step': t['reshaped'] / t['steps'],
                    'tokens': int(t['tokens'])}
        rule = max(pick, key=lambda r: speed(pick[r])['speedup'])
        run = runs[rule]
        domains = np.array([data.domain[s] for s in run['seqs']])
        rows[cost_name(cost)] = {'rule': list(rule), 'cost': cost, 'all': speed(run),
                                 'domains': {d: speed(run, domains == d) for d in sorted(set(domains))}}
    return rows


# ---------------------------------------------------------------- the study

def labelled(data):
    """Answer positions whose model top choice is stored."""
    return np.flatnonzero(data.written & (data.amax >= 0))


def table_size(static, calib, floor):
    """Entries a shipped table would keep (those the drafter could ever use at this floor)."""
    total = 0
    for k in range(1, static.K + 1):
        cell = static_cell(static.cbest[k - 1], static.ctot[k - 1])
        total += int(((calib.confidence('static', k, cell) >= floor) & (static.ctot[k - 1] >= static.min_count)).sum())
    return total


def evaluate(static, cal, test, mode, policy, K, n_max, costs, pad_to=0):
    """Calibrate on `cal`, draft on both, choose the rule on `cal`: the held-out rows of summarize()."""
    runs, calib = {}, Calibration(K)
    for data in (cal, test):
        idx = labelled(data)
        inreq = InRequest(data, K) if policy in ('context', 'both') else None
        if data is cal:
            C, avail = data.context(idx, K)
            truth = (data.amax if mode == 'greedy' else data.tok)[idx]
            calib.add(candidates(C, avail, data.seq[idx], data.pos[idx], static, inreq, K), truth)
        tokens, chance = draft(data, idx, static, inreq, calib, K, n_max, TAUS[0], pad_to)
        runs[data] = sweep(data, idx, tokens, chance, mode, pad_to)
    return summarize(test, runs[test], runs[cal], costs), calib


def run(args):
    t0 = time.time()
    shards = [load_shard(s, args.limit, args.cache) for s in args.shards]
    log(f'loaded {[len(s) for s in shards]} rows in {time.time() - t0:.0f} s')
    train_rows = [r for s in shards[:-1] for r in s]
    if args.train_sources:
        train_rows = [r for r in train_rows if r['source'] in args.train_sources]
    held = [r for r in shards[-1] if r['finished']]
    if args.test_sources:
        held = [r for r in held if r['source'] in args.test_sources]
    # A quarter of the held-out problems set the confidences and pick the drafting rule; the rest are the test.
    order = {p: i for i, p in enumerate(dict.fromkeys(r['id'] for r in held))}
    cal = Data([r for r in held if order[r['id']] % 4 == 0])
    test = Data([r for r in held if order[r['id']] % 4 != 0])
    costs = [{'beta': b, 'reject': 0.0, 'reshape': 0.0} for b in BETAS]
    for spec in args.cost or []:
        name, _, numbers = spec.partition('=')
        beta, reject, reshape, accept = ([float(x) for x in numbers.split(',')] + [0.0])[:4]
        costs.append({'name': name, 'beta': beta, 'reject': reject, 'reshape': reshape, 'accept': accept})
    result = {'args': vars(args), 'configs': {}}

    for fraction in args.fractions:
        keep = max(1, round(1 / fraction))
        ids = {p: i for i, p in enumerate(dict.fromkeys(r['id'] for r in train_rows))}
        part = [r for r in train_rows if ids[r['id']] % keep == 0]
        for labels in args.labels:
            rows = [r for r in part if r['finished']] if labels == 'top' else part
            train = Data(rows)
            idx = labelled(train) if labels == 'top' else np.flatnonzero(train.written)
            target = train.amax[idx] if labels == 'top' else train.tok[idx]
            t1 = time.time()
            static = Static.build(train, idx, target, args.K)
            name = f'{labels}@{len(idx) / 1e6:.1f}M'
            log(f'{name}: table from {len(idx):,} tokens in {time.time() - t1:.0f} s, entries per order '
                f'{[len(k) for k in static.keys]}')
            del train
            for mode, policy, min_count in ((m, p, c) for m in args.modes for p in args.policies for c in args.min_counts):
                first = labels == args.labels[0] and fraction == args.fractions[0] and min_count == args.min_counts[0]
                if policy == 'context' and not first:
                    continue                # the in-request drafter does not depend on the table
                static.min_count = min_count
                use_static = static if policy in ('static', 'both') else None
                rows_out, calib = evaluate(use_static, cal, test, mode, policy, args.K, args.n_max, costs, args.pad_to)
                key = f'{name} {policy} {mode}' + (f' seen>={min_count}' if min_count > 1 else '')
                result['configs'][key] = {
                    'train_tokens': int(len(idx)), 'labels': labels, 'policy': policy, 'mode': mode, 'min_count': min_count,
                    'entries_total': int(sum(len(k) for k in static.keys)) if use_static else 0,
                    'entries_kept': {str(t): table_size(static, calib, t) for t in (0.1, 0.3, 0.5)} if use_static else {},
                    'by_cost': rows_out}
                shown = [n for n in rows_out if n in ('b0.05', 'b0.1', 'b0.2', 'b0.3') or not n.startswith('b0')]
                line = '  '.join(f"{n}: {rows_out[n]['all']['speedup']:.3f}x ({rows_out[n]['rule'][0]} {rows_out[n]['rule'][1]}, "
                                 f"{rows_out[n]['all']['tokens_per_step']:.2f} tok/step, acc {rows_out[n]['all']['acceptance']:.0%})"
                                 for n in shown)
                log(f'{key}: {line}')
                if args.out:
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.out).write_text(json.dumps(result, indent=1))
    log(f'done in {time.time() - t0:.0f} s')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--shards', nargs='+', required=True, help='shard folders; the last one is held out')
    ap.add_argument('--out')
    ap.add_argument('--K', type=int, default=6, help='longest context, in tokens')
    ap.add_argument('--n-max', type=int, default=8, help='longest draft')
    ap.add_argument('--pad-to', type=int, default=0, help='also score always drafting exactly 1..N tokens')
    ap.add_argument('--cost', action='append', metavar='NAME=BETA,REJECT,RESHAPE[,ACCEPT]',
                    help='a measured cost of one step, in plain steps: per draft token, per step that loses a '
                         'draft token, per step whose draft length changed, and per accepted draft token')
    ap.add_argument('--labels', nargs='+', default=['top', 'written'], choices=['top', 'written'],
                    help="what the table counts: the model's top choice, or the token it wrote")
    ap.add_argument('--modes', nargs='+', default=['greedy', 'sampled'], choices=['greedy', 'sampled'])
    ap.add_argument('--policies', nargs='+', default=['static', 'context', 'both'],
                    choices=['static', 'context', 'both'])
    ap.add_argument('--fractions', nargs='+', type=float, default=[1.0], help='shares of the training problems')
    ap.add_argument('--min-counts', nargs='+', type=int, default=[1], help='keep only contexts seen at least this often')
    ap.add_argument('--train-sources', nargs='+')
    ap.add_argument('--test-sources', nargs='+')
    ap.add_argument('--limit', type=int, help='rows per shard, for a quick check')
    ap.add_argument('--cache', help="folder for the shards' unpacked top choices")
    run(ap.parse_args())


if __name__ == '__main__':
    main()
