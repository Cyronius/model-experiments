"""Optional low-rank Q4-head warm start from Nanbeige's Q8 output matrix.

This streams the GGUF output head twice and keeps only a small randomized SVD
sketch in RAM. The resulting factors initialize the tiny drafter's tied token
table; acceptance still has to be measured after drafter training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


GGUF_PY = Path("/home/cyrus/code/phlegm/target/npu-dev/xing/src/llama.cpp-7d7f0287530bc8981690f361d8c1627dbae1c99a/gguf-py")
sys.path.insert(0, str(GGUF_PY))
from gguf import GGMLQuantizationType, GGUFReader  # noqa: E402


def head_rows(raw: np.ndarray, start: int, stop: int, hidden: int) -> np.ndarray:
    blocks = raw[start:stop]
    scales = np.ascontiguousarray(blocks[..., :2]).view("<f2").reshape(stop - start, hidden // 32).astype(np.float32)
    values = blocks[..., 2:].view(np.int8).astype(np.float32)
    return (values * scales[..., None]).reshape(stop - start, hidden)


def run(args):
    reader = GGUFReader(str(args.model), mode="r")
    tensor = next((t for t in reader.tensors if t.name == "output.weight"), None)
    if tensor is None or tensor.tensor_type != GGMLQuantizationType.Q8_0:
        raise ValueError("expected a Q8_0 output.weight in the Nanbeige GGUF")
    hidden, vocab = map(int, tensor.shape)
    if (hidden, vocab) != (7168, 166912):
        raise ValueError(f"unexpected output head shape: {(hidden, vocab)}")
    blocks_per_row = hidden // 32
    raw = tensor.data.reshape(vocab, blocks_per_row, 34)
    rng = np.random.default_rng(args.seed)
    omega, _ = np.linalg.qr(rng.standard_normal((hidden, args.sketch)).astype(np.float32))
    projected = np.empty((vocab, args.sketch), np.float32)
    head_hash = hashlib.sha256()
    energy = 0.0
    for start in range(0, vocab, args.rows):
        stop = min(start + args.rows, vocab)
        head_hash.update(raw[start:stop].tobytes())
        weights = head_rows(raw, start, stop, hidden)
        projected[start:stop] = weights @ omega
        energy += float(np.sum(weights.astype(np.float64) ** 2))
        if start % (args.rows * 64) == 0:
            print(f"sketch {stop}/{vocab}", flush=True)
    for iteration in range(args.power):
        q, _ = np.linalg.qr(projected, mode="reduced")
        right = np.zeros((hidden, args.sketch), np.float32)
        for start in range(0, vocab, args.rows):
            stop = min(start + args.rows, vocab)
            right += head_rows(raw, start, stop, hidden).T @ q[start:stop]
        right, _ = np.linalg.qr(right, mode="reduced")
        for start in range(0, vocab, args.rows):
            stop = min(start + args.rows, vocab)
            projected[start:stop] = head_rows(raw, start, stop, hidden) @ right
        print(f"power iteration {iteration + 1}/{args.power}", flush=True)
    basis, _ = np.linalg.qr(projected, mode="reduced")
    small = np.zeros((args.sketch, hidden), np.float32)
    for start in range(0, vocab, args.rows):
        stop = min(start + args.rows, vocab)
        small += basis[start:stop].T @ head_rows(raw, start, stop, hidden)
        if start % (args.rows * 64) == 0:
            print(f"factor {stop}/{vocab}", flush=True)
    left, singular, right_t = np.linalg.svd(small, full_matrices=False)
    factors = (basis @ left[:, :args.rank]) * singular[:args.rank]
    rms = float(np.sqrt(np.mean(factors.astype(np.float64) ** 2)))
    factor_multiplier = args.target_rms / rms
    factors = (factors * factor_multiplier).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, factors)
    np.save(args.output.with_name(args.output.stem + "-right.npy"), right_t[:args.rank])
    report = {"source": str(args.model), "source_bytes": args.model.stat().st_size,
              "output_head_sha256": head_hash.hexdigest(), "vocab": vocab,
              "hidden": hidden, "rank": args.rank, "sketch": args.sketch,
              "power_iterations": args.power,
              "seed": args.seed, "target_rms": args.target_rms,
              "factor_multiplier": factor_multiplier,
              "estimated_rank_energy_fraction": float(np.sum(singular[:args.rank] ** 2) / energy),
              "singular_values": singular[:args.rank].tolist(),
              "note": "Randomized SVD without power iterations; energy is a lower bound for the optimal rank at this sketch size."}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("rank", "estimated_rank_energy_fraction",
                                           "output_head_sha256")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path,
                        default=Path("/home/cyrus/models/Nanbeige4.3-33B-Q4pure-NPU2/model.gguf"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--sketch", type=int, default=32)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--power", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target-rms", type=float, default=0.25)
    args = parser.parse_args()
    if not 1 <= args.rank <= args.sketch <= 128 or args.rows < 1 or not 0 <= args.power <= 4:
        parser.error("require 1 <= rank <= sketch <= 128, positive rows, and 0 <= power <= 4")
    run(args)


if __name__ == "__main__":
    main()
