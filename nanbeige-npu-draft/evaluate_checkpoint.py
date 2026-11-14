"""Prompt-level uncertainty and newline baseline for a saved Q4 drafter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from drafter import Config, Drafter


def score(hits: np.ndarray, owner: np.ndarray, boots: int = 10000) -> dict:
    prefix = np.cumprod(hits.astype(np.int32), axis=1)
    accepted = prefix.sum(axis=1)
    groups = np.unique(owner)
    per_prompt = np.asarray([accepted[owner == item].mean() for item in groups])
    rng = np.random.default_rng(37)
    draws = rng.integers(len(groups), size=(boots, len(groups)))
    bootstrap = per_prompt[draws].mean(axis=1)
    return {"blocks": len(hits), "prompts": len(groups),
            "accepted_future_per_block": float(accepted.mean()),
            "accepted_future_per_prompt": float(per_prompt.mean()),
            "prompt_bootstrap_95ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "six_future_exact_blocks": int(prefix[:, -1].sum()),
            "first_future_accuracy": float(hits[:, 0].mean()),
            "per_prompt_accepted_future": per_prompt.tolist()}


@torch.no_grad()
def run(args):
    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Drafter(Config(**ck["config"]["config"])).to(device).eval()
    model.load_state_dict(ck["model"])
    with np.load(args.data) as archive:
        hidden = torch.from_numpy(archive["hidden"].astype(np.float32)).to(device)
        seed = torch.from_numpy(archive["seed"].astype(np.int64)).to(device)
        future = archive["future"].astype(np.int64)
        owner = archive["owner"].astype(np.int32)
    reports = {}
    for passes in (1, 2):
        predictions = []
        for start in range(0, len(seed), args.batch):
            result = model.draft(hidden[start:start + args.batch], seed[start:start + args.batch],
                                 passes=passes)
            predictions.append(result[:, 1:].cpu().numpy())
        hits = np.concatenate(predictions) == future
        reports[f"{passes}_pass"] = score(hits, owner)
    if args.train:
        with np.load(args.train) as archive:
            training = archive["future"].astype(np.int64)
        majority = np.asarray([np.bincount(training[:, j]).argmax() for j in range(6)])
        reports["constant_majority"] = {"tokens": majority.tolist(),
                                        **score(future == majority, owner)}
    report = {"checkpoint": str(args.checkpoint), "checkpoint_step": ck["step"],
              "data": str(args.data), "reports": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: {"accepted_future_per_block": v["accepted_future_per_block"],
                          "prompt_bootstrap_95ci": v["prompt_bootstrap_95ci"]}
                      for k, v in reports.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=32)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
