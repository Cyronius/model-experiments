"""Check whether captured hidden vectors reproduce greedy Nanbeige head choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from init_head import GGMLQuantizationType, GGUFReader, head_rows


def run(args):
    with np.load(args.data) as data:
        hidden = data["hidden"][:args.samples].astype(np.float32)
        seed = data["seed"][:args.samples].astype(np.int32)
    if len(seed) == 0:
        raise ValueError("no captured blocks")
    reader = GGUFReader(str(args.model), mode="r")
    tensor = next(t for t in reader.tensors if t.name == "output.weight")
    if tensor.tensor_type != GGMLQuantizationType.Q8_0:
        raise ValueError("expected Q8_0 target head")
    width, vocab = map(int, tensor.shape)
    raw = tensor.data.reshape(vocab, width // 32, 34)
    best_score = np.full(len(seed), -np.inf, np.float32)
    best_id = np.zeros(len(seed), np.int32)
    for start in range(0, vocab, args.rows):
        stop = min(start + args.rows, vocab)
        logits = head_rows(raw, start, stop, width) @ hidden.T
        local = logits.argmax(axis=0)
        score = logits[local, np.arange(len(seed))]
        improve = score > best_score
        best_score[improve] = score[improve]
        best_id[improve] = start + local[improve]
    factors = np.load(args.factors)
    right = np.load(args.factors.with_name(args.factors.stem + "-right.npy"))
    rank = factors.shape[1]
    approximate = (factors @ (right @ hidden.T)).argmax(axis=0)
    report = {"samples": len(seed), "exact_q8_head_matches_greedy_seed": int(np.sum(best_id == seed)),
              "rank": rank, "low_rank_head_matches_greedy_seed": int(np.sum(approximate == seed)),
              "exact_head_token_ids": best_id.tolist(), "greedy_seed_ids": seed.tolist(),
              "rank16_token_ids": approximate.tolist(),
              "note": "Captured hidden vectors are FP16; near-tie argmax may differ from the native target."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("samples", "rank", "exact_q8_head_matches_greedy_seed",
                                           "low_rank_head_matches_greedy_seed")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path,
                        default=Path("/home/cyrus/models/Nanbeige4.3-33B-Q4pure-NPU2/model.gguf"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--rows", type=int, default=256)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
