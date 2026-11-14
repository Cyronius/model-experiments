"""Measure one-step MTP acceptance against the exact Q4 GGUF output head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from init_head import GGMLQuantizationType, GGUFReader, head_rows


def probe(args: argparse.Namespace) -> None:
    with np.load(args.data) as data:
        states = data["hidden"][:args.samples].astype(np.float32)
        actual = data["future"][:args.samples, 0].astype(np.int32)
        owners = data["owner"][:args.samples].astype(np.int32)
    if len(states) == 0 or not np.isfinite(states).all():
        raise ValueError("empty or nonfinite MTP states")
    reader = GGUFReader(str(args.model), mode="r")
    tensor = next(t for t in reader.tensors if t.name == "output.weight")
    if tensor.tensor_type != GGMLQuantizationType.Q8_0:
        raise ValueError("expected Q8_0 target head")
    hidden, vocab = map(int, tensor.shape)
    if states.shape[1] != hidden or args.topk > args.rows:
        raise ValueError("state width or top-k chunk size is invalid")
    raw = tensor.data.reshape(vocab, hidden // 32, 34)
    scores = np.full((args.topk, len(states)), -np.inf, np.float32)
    ids = np.zeros((args.topk, len(states)), np.int32)
    columns = np.arange(len(states))[None, :]
    for start in range(0, vocab, args.rows):
        stop = min(start + args.rows, vocab)
        logits = head_rows(raw, start, stop, hidden) @ states.T
        local = np.argpartition(logits, -args.topk, axis=0)[-args.topk:]
        local_scores = logits[local, columns]
        joined_scores = np.concatenate((scores, local_scores))
        joined_ids = np.concatenate((ids, (start + local).astype(np.int32)))
        selected = np.argpartition(joined_scores, -args.topk, axis=0)[-args.topk:]
        scores = joined_scores[selected, columns]
        ids = joined_ids[selected, columns]
        if start % (args.rows * 64) == 0:
            print(f"output head {stop}/{vocab}", flush=True)
    order = np.argsort(-scores, axis=0)
    scores = scores[order, columns].T.copy()
    ids = ids[order, columns].T.copy()
    top1 = ids[:, 0] == actual
    topk = (ids == actual[:, None]).any(axis=1)
    per_prompt = {str(owner): {"blocks": int(np.sum(owners == owner)),
                               "top1": float(np.mean(top1[owners == owner])),
                               "topk": float(np.mean(topk[owners == owner]))}
                  for owner in np.unique(owners)}
    domains = {}
    if args.generated is not None:
        rows = [json.loads(line) for line in args.generated.read_text().splitlines() if line]
        if int(owners.max()) >= len(rows):
            raise ValueError("generated prompt mapping does not cover every owner")
        labels = np.asarray([rows[owner]["id"].split("-", 1)[0] for owner in owners])
        domains = {str(label): {"blocks": int(np.sum(labels == label)),
                                "top1": float(np.mean(top1[labels == label])),
                                "topk": float(np.mean(topk[labels == label]))}
                   for label in np.unique(labels)}
    report = {"blocks": len(actual), "top1_acceptance": float(np.mean(top1)),
              "topk_coverage": float(np.mean(topk)), "k": args.topk,
              "prompt_count": len(per_prompt), "per_prompt": per_prompt, "domains": domains,
              "target": str(args.model), "features": str(args.data),
              "note": "Greedy, offline first speculative token; Q4 target output head on saved BF16 MTP states."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez(args.output.with_suffix(".npz"), top_ids=ids, top_scores=scores,
             actual=actual, owner=owners)
    print(json.dumps({key: report[key] for key in ("blocks", "top1_acceptance", "topk_coverage", "k")}),
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=Path("/home/cyrus/models/Nanbeige4.3-33B-Q4pure-NPU2/model.gguf"))
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()
    if args.topk < 1 or args.rows < 1:
        parser.error("topk and rows must be positive")
    probe(args)


if __name__ == "__main__":
    main()
