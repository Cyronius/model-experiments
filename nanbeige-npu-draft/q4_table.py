"""Row-scaled signed Q4 table format for the 16-dimensional token factors.

Each row stores 16 signed nibbles (low nibble first) and one FP16 scale.
The format is deliberately simple enough to implement in an XDNA2 kernel.
"""

from __future__ import annotations

import numpy as np


def packed_bytes(vocab: int, rank: int) -> int:
    if rank % 2:
        raise ValueError("rank must be even")
    return vocab * (rank // 2 + 2)


def pack_q4(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if table.ndim != 2 or table.shape[1] % 2:
        raise ValueError("expected a 2D table with even rank")
    values = np.asarray(table, dtype=np.float32)
    scale = np.max(np.abs(values), axis=1) / 7.0
    scale = np.maximum(scale, np.finfo(np.float16).tiny).astype(np.float16)
    quant = np.clip(np.rint(values / scale[:, None]), -8, 7).astype(np.int8)
    bits = quant.astype(np.uint8) & 15
    packed = bits[:, 0::2] | (bits[:, 1::2] << 4)
    return packed, scale


def unpack_q4(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if packed.ndim != 2 or scale.shape != (packed.shape[0],):
        raise ValueError("incompatible packed table and scales")
    bits = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    bits[:, 0::2] = packed & 15
    bits[:, 1::2] = packed >> 4
    signed = bits.astype(np.int8)
    signed[signed >= 8] -= 16
    return signed.astype(np.float32) * scale.astype(np.float32)[:, None]


def streaming_argmax(vectors: np.ndarray, packed: np.ndarray, scale: np.ndarray,
                     rows_per_chunk: int = 2048) -> np.ndarray:
    """Reference for a bounded-memory vocabulary scan, returning only token IDs.

    The NPU implementation should shard these row chunks across memory tiles
    and reduce one maximum per drafted position. This reference checks the
    exact result without allocating a [positions, vocabulary] logit matrix.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != packed.shape[1] * 2:
        raise ValueError("expected [positions, rank] vectors")
    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be positive")
    best_score = np.full(len(vectors), -np.inf, dtype=np.float32)
    best_id = np.zeros(len(vectors), dtype=np.int32)
    for start in range(0, len(scale), rows_per_chunk):
        stop = min(start + rows_per_chunk, len(scale))
        logits = vectors @ unpack_q4(packed[start:stop], scale[start:stop]).T
        local_id = np.argmax(logits, axis=1)
        local_score = logits[np.arange(len(vectors)), local_id]
        update = local_score > best_score
        best_score[update] = local_score[update]
        best_id[update] = start + local_id[update]
    return best_id
