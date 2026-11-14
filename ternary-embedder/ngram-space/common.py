"""Plan B common helpers: tokenization without special tokens, n-gram key packing,
and access to the shared code in `shared-qwen8b/shared.py`.

Never imports anything from `qwen-space/`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # ternary-embedder/
PLAN = ROOT / "ngram-space"
WORK = ROOT / "training-v2" / "work" / "ngram-space"
SHARED_WORK = ROOT / "training-v2" / "work" / "shared-qwen8b"
TOKENIZER_JSON = ROOT / "tokenizer" / "tokenizer.json"

VOCAB = 30522
PAD, UNK, CLS, SEP, MASK = 0, 100, 101, 102, 103
SPECIAL_IDS = (PAD, UNK, CLS, SEP, MASK)
BITS = 15                       # every id < 2**15
MASK15 = (1 << BITS) - 1

_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(str(TOKENIZER_JSON))
        _TOK.no_truncation()
        _TOK.no_padding()
    return _TOK


def encode(texts, offsets=False):
    """List of id arrays (uint16) without [CLS]/[SEP]; optionally char offsets."""
    enc = tokenizer().encode_batch(list(texts), add_special_tokens=False)
    ids = [np.asarray(e.ids, dtype=np.uint16) for e in enc]
    if offsets:
        return ids, [e.offsets for e in enc]
    return ids


# ---- n-gram keys -----------------------------------------------------------
# bigram  (a, b)    -> a << 15 | b                (30 bits)
# trigram (a, b, c) -> a << 30 | b << 15 | c      (45 bits)
# An n-gram belongs to the position of its last token.

def bigram_keys(ids: np.ndarray) -> np.ndarray:
    a = ids.astype(np.uint64)
    return (a[:-1] << np.uint64(BITS)) | a[1:]


def trigram_keys(ids: np.ndarray) -> np.ndarray:
    a = ids.astype(np.uint64)
    return (a[:-2] << np.uint64(2 * BITS)) | (a[1:-1] << np.uint64(BITS)) | a[2:]


def unpack(key: int, n: int):
    return tuple((int(key) >> (BITS * (n - 1 - k))) & MASK15 for k in range(n))


def doc_keys(flat: np.ndarray, starts: np.ndarray, n: int):
    """Keys of all within-document n-grams of a flat id stream with document
    start offsets (len = ndocs + 1). Returns (keys, end_positions)."""
    a = flat.astype(np.uint64)
    L = len(a)
    if L < n:
        return np.zeros(0, np.uint64), np.zeros(0, np.int64)
    if n == 2:
        k = (a[:-1] << np.uint64(BITS)) | a[1:]
    else:
        k = (a[:-2] << np.uint64(2 * BITS)) | (a[1:-1] << np.uint64(BITS)) | a[2:]
    end = np.arange(n - 1, L, dtype=np.int64)
    # valid iff the n-gram's first token is in the same doc as its last
    doc_of = np.searchsorted(starts, end, side="right") - 1
    ok = end - (n - 1) >= starts[doc_of]
    return k[ok], end[ok]


# ---- shared code -------------------------------------------------------------

def shared():
    """Import `shared-qwen8b/shared.py` (built under the contract's `code` claim)."""
    p = str(ROOT / "shared-qwen8b")
    if p not in sys.path:
        sys.path.insert(0, p)
    import shared as _s  # noqa: E402
    return _s


def embed(texts, instruction=""):
    """Hosted teacher vectors through the shared cache, counted against Plan B."""
    return shared().embed(texts, instruction=instruction, plan="B")


def unit(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)
