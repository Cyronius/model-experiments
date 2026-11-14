"""Capture greedy target continuations and per-token Q4 target features.

Run `generate` against a normal llama-server, then restart it with
`--embedding --pooling none --embd-normalize -1` and run `embed`.
Both servers must load the same GGUF. Keep generation and embedding files
separate for training and held-out evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np


def post(url: str, path: str, payload: dict) -> dict | list:
    req = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def prompts(path: Path):
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
        rows = rows["prompts"] if isinstance(rows, dict) else rows
    else:
        rows = (json.loads(line) for line in path.read_text().splitlines() if line)
    for row in rows:
        prompt = row.get("prompt") or row.get("question")
        if prompt:
            yield {"id": str(row.get("id", row.get("key", ""))), "prompt": prompt}


def generate(args):
    rows = list(prompts(args.prompts))[args.start:args.start + args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = []
    if args.output.exists():
        done = [json.loads(line) for line in args.output.read_text().splitlines() if line]
        if len(done) > len(rows) or any(a["id"] != b["id"] or a["prompt"] != b["prompt"]
                                        for a, b in zip(done, rows)):
            raise ValueError("existing output is not a prefix of this prompt selection")
    with args.output.open("a") as out:
        for i, row in enumerate(rows[len(done):], start=len(done)):
            result = post(args.url, "/completion", {
                "prompt": row["prompt"], "n_predict": args.tokens,
                "temperature": 0, "seed": 1, "cache_prompt": False,
                "return_tokens": True,
            })
            if not isinstance(result.get("content"), str) or not isinstance(result.get("tokens"), list):
                raise ValueError(f"completion missing text or token IDs at row {i}")
            record = {**row, "completion": result["content"],
                      "generated_tokens": result["tokens"],
                      "tokens_predicted": result.get("tokens_predicted"),
                      "stop_type": result.get("stop_type")}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{i + 1}/{len(rows)} {row['id']} tokens={record['tokens_predicted']}", flush=True)


def embed(args):
    rows = [json.loads(line) for line in args.generated.read_text().splitlines() if line]
    features, seeds, futures, owners = [], [], [], []
    for i, row in enumerate(rows):
        prompt_ids = post(args.url, "/tokenize", {"content": row["prompt"], "add_special": True})["tokens"]
        ids = prompt_ids + row["generated_tokens"]
        result = post(args.url, "/embeddings", {"content": ids})
        hidden = np.asarray(result[0]["embedding"], dtype=np.float32)
        if hidden.shape != (len(ids), 7168):
            raise ValueError(f"feature alignment failed at row {i}: {hidden.shape}, {len(ids)} tokens")
        # Only draft within the generated continuation. Overlapping anchors
        # provide more supervised training cases; evaluation uses stride 7.
        for anchor in range(len(prompt_ids) - 1, len(ids) - 7, args.stride):
            features.append(hidden[anchor].astype(np.float16))
            seeds.append(ids[anchor + 1])
            futures.append(ids[anchor + 2:anchor + 8])
            owners.append(i)
        print(f"{i + 1}/{len(rows)} {row['id']} blocks={len(features)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, hidden=np.asarray(features, dtype=np.float16),
             seed=np.asarray(seeds, dtype=np.int32),
             future=np.asarray(futures, dtype=np.int32),
             owner=np.asarray(owners, dtype=np.int32))
    manifest = {"generated": str(args.generated),
                "generated_sha256": hashlib.sha256(args.generated.read_bytes()).hexdigest(),
                "examples": len(rows), "blocks": len(features), "target_hidden": 7168,
                "continuation_only": True, "anchor_stride": args.stride}
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["generate", "embed"])
    parser.add_argument("--url", default="http://127.0.0.1:8400")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--stride", type=int, default=7)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("stride must be positive")
    if args.mode == "generate":
        if args.prompts is None:
            parser.error("generate requires --prompts")
        generate(args)
    else:
        if args.generated is None:
            parser.error("embed requires --generated")
        embed(args)


if __name__ == "__main__":
    main()
