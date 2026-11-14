"""Native projection ABI and independent NumPy arithmetic for the 384 student.

No Hugging Face downloads or accelerator runtime are needed by this module.
Weights are [N/32, K/16, 128] bytes; activations [K/16, 4, 16] int8.
An output block is [scale groups, 2 output halves, 4 rows, 16] int32,
and contains FOUR times the signed dot product (the int4 operand is 4*w).
"""
from __future__ import annotations

import numpy as np


def group_width(k: int) -> int:
    return 256 if k > 256 and k % 256 == 0 else k


def quantize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    k = x.shape[-1]
    grouped = x.reshape(*x.shape[:-1], k // group_width(k), group_width(k))
    maximum = np.maximum(np.abs(grouped).max(-1, keepdims=True), np.float32(1e-5))
    mantissa, exponent = np.frexp(maximum)
    scales = np.ldexp(np.ones_like(maximum), exponent - 7 + (2 * mantissa > 127 / 64))
    codes = np.clip(np.rint(grouped / scales), -128, 127).astype(np.int8)
    return codes.reshape(x.shape), scales[..., 0]


def pack_weights(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes)
    n, k = codes.shape
    if n % 32 or k % 16 or not np.isin(codes, [-1, 0, 1]).all():
        raise ValueError("Weights require N%32=K%16=0 and codes in {-1,0,1}")
    c = (codes.astype(np.int16) & 3).astype(np.uint8)
    kb, lane = np.arange(k // 16)[:, None], np.arange(128)[None, :]
    ki, ni = kb * 16 + lane // 8, 2 * (lane % 8)
    blocks = []
    for start in range(0, n, 32):
        w = c[start:start + 32]
        blocks.append(w[ni, ki] | (w[ni + 16, ki] << 2)
                      | (w[ni + 1, ki] << 4) | (w[ni + 17, ki] << 6))
    return np.stack(blocks)


def unpack_weights(packed: np.ndarray) -> np.ndarray:
    """Decode by output/input coordinates, independently of the packer's indexing."""
    blocks, kb, _ = packed.shape
    out = np.empty((blocks * 32, kb * 16), dtype=np.int8)
    for n in range(blocks * 32):
        for k in range(kb * 16):
            lane = (k % 16) * 8 + (n % 16) // 2
            shift = (2 if n % 32 >= 16 else 0) + (4 if n % 2 else 0)
            code = (int(packed[n // 32, k // 16, lane]) >> shift) & 3
            if code == 2:
                raise ValueError("Reserved 2-bit ternary code 2")
            out[n, k] = code if code < 2 else code - 4
    return out


def pack_activations(codes: np.ndarray) -> np.ndarray:
    if codes.ndim != 2 or codes.shape[0] != 4 or codes.shape[1] % 16:
        raise ValueError("The native projection primitive takes exactly four rows")
    return codes.reshape(4, -1, 16).transpose(1, 0, 2).copy()


def integer_products(codes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the complete hardware ABI, using int64 independent dot products."""
    n, k = weights.shape
    width = group_width(k)
    out = np.empty((n // 32, k // width, 2, 4, 16), dtype=np.int32)
    for group, start in enumerate(range(0, k, width)):
        dot = 4 * (codes[:, start:start + width].astype(np.int64)
                   @ weights[:, start:start + width].astype(np.int64).T)
        if np.abs(dot).max() > np.iinfo(np.int32).max:
            raise OverflowError("int32 product overflow")
        for block in range(n // 32):
            out[block, group] = dot[:, block * 32:(block + 1) * 32].reshape(4, 2, 16).transpose(1, 0, 2)
    return out


def dequantize(products, activation_scales, weight_scales, bias):
    blocks, groups, _, rows, _ = products.shape
    dot = products.transpose(3, 1, 0, 2, 4).reshape(rows, groups, blocks * 32).astype(np.float32) / 4
    scaled = (dot * activation_scales[..., None]).sum(1)
    return scaled * weight_scales.reshape(1, -1) + bias.reshape(1, -1)
