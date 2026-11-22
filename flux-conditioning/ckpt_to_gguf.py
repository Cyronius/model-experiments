"""Write healed Klein block weights (heal2.py ckpt.pt) into a copy of the Q8_0 DiT GGUF for the NPU worker (2026-11-20)."""
from __future__ import annotations

import argparse
import mmap
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"C:\code\phlegm\open-kernels\open_kernels")
from klein_gguf import BLOCK, Gguf, resolve  # noqa: E402


def q8_0(w: np.ndarray) -> np.ndarray:
    """llama.cpp Q8_0: blocks of 32 along the input dim, f16 scale = amax / 127."""
    b = w.astype(np.float32).reshape(-1, 32)
    d = np.abs(b).max(axis=1, keepdims=True) / 127.0
    q = np.round(b / np.where(d == 0, 1, d)).clip(-127, 127).astype(np.int8)
    out = np.empty((len(b), 34), np.uint8)
    out[:, 0:2] = d.astype("<f2").view(np.uint8)
    out[:, 2:] = q.view(np.uint8)
    return out.reshape(-1)


def encode(w: np.ndarray, typ: str) -> np.ndarray:
    if typ == "Q8_0":
        return q8_0(w)
    if typ == "F32":
        return w.astype("<f4").reshape(-1).view(np.uint8)
    if typ == "F16":
        return w.astype("<f2").reshape(-1).view(np.uint8)
    if typ == "BF16":
        return (w.astype(np.float32).reshape(-1).view(np.uint32) >> 16).astype("<u2").view(np.uint8)
    raise ValueError(typ)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    shutil.copyfile(a.src, a.out)
    g = Gguf(a.out)
    with open(a.out, "r+b") as f, mmap.mmap(f.fileno(), 0) as mm:
        n, skipped, worst = 0, [], 0.0
        for k, sd in sorted(ck["blocks"].items()):
            prefix = f"transformer_blocks.{k}." if k < 5 else f"single_transformer_blocks.{k - 5}."
            for tail, v in sd.items():
                r = resolve(prefix + tail)
                if r is None:
                    skipped.append(prefix + tail)
                    continue
                tensor, part = r
                typ, dims, off = g.tensors[tensor]
                w = v.float().numpy()
                rows, cols = dims[::-1] if len(dims) == 2 else (1, dims[0])
                per, width = BLOCK[typ]
                row_bytes = cols // per * width
                r0 = 0 if part is None else part * w.shape[0]
                start = g.data_base + off + r0 * row_bytes
                enc = encode(w.reshape(-1, cols), typ)
                assert enc.size == w.reshape(-1, cols).shape[0] * row_bytes, (tensor, enc.size)
                old = g.array(tensor)[r0:r0 + w.reshape(-1, cols).shape[0]] if len(dims) == 2 else g.array(tensor)
                mm[start:start + enc.size] = enc.tobytes()
                worst = max(worst, float(np.abs(w.reshape(old.shape) - old).max()))
                n += 1
        mm.flush()
    print(f"patched {n} tensors from {len(ck['blocks'])} blocks; max change {worst:.4f}; no GGUF mapping for {skipped}")
    g2 = Gguf(a.out)
    k = max(ck["blocks"])
    name = f"single_transformer_blocks.{k - 5}.attn.to_out.weight"
    t, _ = resolve(name)
    ref = ck["blocks"][k]["attn.to_out.weight"].float().numpy()
    got = g2.array(t)
    print(f"check {t}: rel err after Q8_0 {np.linalg.norm(got - ref) / np.linalg.norm(ref):.4f}")


if __name__ == "__main__":
    main()
