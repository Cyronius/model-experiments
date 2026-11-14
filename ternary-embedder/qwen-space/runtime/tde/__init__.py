"""tde: CPU document embedder whose vectors live in the space of hosted Qwen3-Embedding-8B.

Runtime dependencies: numpy, tokenizers, and the bundled libtde.so (C ABI, no OpenMP).
Never imports torch or transformers.

    from tde import Embedder
    emb = Embedder("/path/to/release")          # folder with model.tde, tokenizer.json, libtde.so
    vecs = emb.encode(["a document", ...])       # float32 [n, dim], unit norm
    vecs, conf = emb.encode(docs, return_confidence=True)

Documents are cut to the document contract (first 504 WordPiece content tokens, never inside a word);
`contract_text(text)` returns exactly the string a later refinement job should send to the hosted model.
English only (BERT uncased vocabulary).
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

__all__ = ["Embedder", "read_model", "MAGIC"]
MAGIC = b"TDE1"
_CFG_KEYS = ["vocab", "d", "heads", "ffn", "layers", "max_pos", "window", "out", "head_hidden", "emb_int8",
             "cls", "sep", "max_windows"]


def read_model(path):
    """-> (config dict, metadata dict, {name: numpy array}) from a .tde file (memory-mapped)."""
    path = Path(path)
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"{path} is not a tde model file")
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    arrays = {}
    for name, t in header["tensors"].items():
        dt = np.dtype(t["dtype"])
        count = int(np.prod(t["shape"])) if t["shape"] else 1
        arrays[name] = raw[t["offset"]:t["offset"] + count * dt.itemsize].view(dt).reshape(t["shape"])
    return header["config"], header.get("metadata", {}), arrays


def _ptr_names(cfg):
    names = ["word", "word_scale", "pos", "tok_type0", "ln_e_w", "ln_e_b"]
    for l in range(cfg["layers"]):
        for j in ("q", "k", "v", "o", "fc1", "fc2"):
            names += [f"l{l}.{j}.W", f"l{l}.{j}.S", f"l{l}.{j}.B"]
        names += [f"l{l}.ln1_w", f"l{l}.ln1_b", f"l{l}.ln2_w", f"l{l}.ln2_b"]
    names += ["head.skipW", "head.bias", "head.W1", "head.b1", "head.W2"]
    return names


class Embedder:
    def __init__(self, folder=None, model=None, tokenizer=None, library=None, workers=2, dim=None):
        folder = Path(folder) if folder else Path(__file__).resolve().parent.parent
        model = Path(model) if model else folder / "model.tde"
        tokenizer = Path(tokenizer) if tokenizer else folder / "tokenizer.json"
        library = Path(library) if library else _find_lib(folder)
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(tokenizer))
        self.tok.no_truncation()
        self.tok.no_padding()
        self.config, self.metadata, self._arrays = read_model(model)
        self.content_tokens = self.config["window"] * self.config["max_windows"]
        self.dim = int(dim or self.metadata.get("stored_dimension", self.config["out"]))
        self.workers = max(1, int(workers))
        lib = ctypes.CDLL(str(library))  # CDLL: calls release the GIL
        lib.tde_load.restype = ctypes.c_void_p
        lib.tde_load.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_void_p)]
        lib.tde_encode.restype = ctypes.c_int
        lib.tde_encode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.tde_isa.argtypes = [ctypes.c_void_p]
        lib.tde_free.argtypes = [ctypes.c_void_p]
        self._lib = lib
        self._cfg = np.array([self.config[k] for k in _CFG_KEYS], dtype=np.int32)
        ptrs = []
        for name in _ptr_names(self.config):
            a = self._arrays.get(name)
            ptrs.append(None if a is None else a.ctypes.data)
        self._ptrs = (ctypes.c_void_p * len(ptrs))(*ptrs)
        h = lib.tde_load(self._cfg.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), self._ptrs)
        if not h:
            raise RuntimeError("tde_load failed (CPU without AVX2/FMA/F16C, or bad model file)")
        self._h = h
        self._pool = ThreadPoolExecutor(self.workers) if self.workers > 1 else None

    # ------------------------------------------------------------------ contract
    @property
    def isa(self):
        return {2: "avx512-vnni", 1: "avx2"}[self._lib.tde_isa(self._h)]

    def _cut(self, enc):
        ids = enc.ids
        n = self.content_tokens
        if len(ids) <= n:
            return ids, None
        words = enc.word_ids
        while n > 0 and words[n] is not None and words[n] == words[n - 1]:
            n -= 1
        return ids[:n], enc.offsets[n][0]

    def tokenize(self, texts):
        """Content-token ids under the document contract (no [CLS]/[SEP])."""
        return [self._cut(e)[0] for e in self.tok.encode_batch(list(texts), add_special_tokens=False)]

    def contract_text(self, text):
        """The exact string covered by the document contract (send this to the hosted model)."""
        _, cut = self._cut(self.tok.encode(text, add_special_tokens=False))
        return text if cut is None else text[:cut].rstrip()

    # ------------------------------------------------------------------ encode
    def _encode_ids(self, ids, out, conf, feats):
        offsets = np.zeros(len(ids) + 1, np.int64)
        offsets[1:] = np.cumsum([len(x) for x in ids])
        flat = np.fromiter((t for x in ids for t in x), dtype=np.int32, count=int(offsets[-1]))
        if flat.size == 0:
            flat = np.zeros(1, np.int32)
        rc = self._lib.tde_encode(self._h, flat.ctypes.data, offsets.ctypes.data, len(ids), out.ctypes.data,
                                  conf.ctypes.data, None if feats is None else feats.ctypes.data)
        if rc != 0:
            raise RuntimeError(f"tde_encode failed ({rc})")

    def encode(self, texts, dim=None, return_confidence=False, return_tokens=False, return_features=False,
               batch=16):
        """Unit vectors [n, dim] (prefix of the 4096-d space, re-normalised).

        return_confidence: also the norm of the head output before normalisation.
        return_tokens: also the number of content tokens used per document.
        """
        texts = list(texts)
        ids = self.tokenize(texts)
        n, O = len(ids), self.config["out"]
        out = np.empty((n, O), np.float32)
        conf = np.empty(n, np.float32)
        feats = np.empty((n, self.config["d"]), np.float32) if return_features else None
        chunks = [(i, min(i + batch, n)) for i in range(0, n, batch)]
        run = lambda c: self._encode_ids(ids[c[0]:c[1]], out[c[0]:c[1]], conf[c[0]:c[1]],
                                         None if feats is None else feats[c[0]:c[1]])
        if self._pool and len(chunks) > 1:
            list(self._pool.map(run, chunks))
        else:
            for c in chunks:
                run(c)
        k = int(dim or self.dim)
        if k < O:
            out = out[:, :k] / np.linalg.norm(out[:, :k], axis=1, keepdims=True)
        res = [out]
        if return_confidence:
            res.append(conf)
        if return_tokens:
            res.append(np.array([len(x) for x in ids]))
        if return_features:
            res.append(feats)
        return res[0] if len(res) == 1 else tuple(res)

    def close(self):
        if self._pool:
            self._pool.shutdown()
        if getattr(self, "_h", None):
            self._lib.tde_free(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _find_lib(folder):
    for p in (os.environ.get("TDE_LIBRARY"), folder / "libtde.so", Path(__file__).resolve().parent / "libtde.so",
              Path(__file__).resolve().parent.parent / "build/libtde.so"):
        if p and Path(p).exists():
            return Path(p)
    raise FileNotFoundError("libtde.so not found; set TDE_LIBRARY")

