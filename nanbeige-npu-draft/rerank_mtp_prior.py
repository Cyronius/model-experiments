"""Test a sparse MTP top-k prior on the tiny drafter's first draft slot.

Select two scalar prior weights on validation, then evaluate them once on the
disjoint test prompts. No new drafter parameters or vocabulary rows are added.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from drafter import Config, Drafter


@torch.no_grad()
def summaries(model: Drafter, data_path: Path, prior_path: Path, device, batch: int):
    with np.load(data_path) as data, np.load(prior_path) as prior:
        hidden = data["hidden"].astype(np.float32)
        seed = data["seed"].astype(np.int64)
        future = data["future"].astype(np.int64)
        owner = data["owner"].astype(np.int32)
        ids = prior["top_ids"].astype(np.int64)
        scores = prior["top_scores"].astype(np.float32)
        if not np.array_equal(prior["actual"], future[:, 0]) or not np.array_equal(prior["owner"], owner):
            raise ValueError("MTP top-k rows do not align with tiny-drafter features")
    if ids.shape[0] != len(seed) or scores.shape != ids.shape:
        raise ValueError("invalid MTP top-k shape")
    outside_scores, outside_ids, candidate_scores, later_ids = [], [], [], []
    for start in range(0, len(seed), batch):
        stop = min(start + batch, len(seed))
        h = torch.from_numpy(hidden[start:stop]).to(device)
        s = torch.from_numpy(seed[start:stop]).to(device)
        tokens = torch.zeros((len(s), model.config.block), dtype=torch.long, device=device)
        tokens[:, 0] = s
        visible = torch.zeros_like(tokens, dtype=torch.bool)
        visible[:, 0] = True
        logits = model(h, tokens, visible)
        top_ids = torch.from_numpy(ids[start:stop]).to(device)
        first = logits[:, 0]
        candidate_scores.append(first.gather(1, top_ids).float().cpu().numpy())
        first.scatter_(1, top_ids, float("-inf"))
        best = first.max(dim=1)
        outside_scores.append(best.values.float().cpu().numpy())
        outside_ids.append(best.indices.cpu().numpy())
        later_ids.append(logits[:, 1:].argmax(dim=-1).cpu().numpy())
    return {"outside_scores": np.concatenate(outside_scores),
            "outside_ids": np.concatenate(outside_ids),
            "candidate_scores": np.concatenate(candidate_scores),
            "later_ids": np.concatenate(later_ids),
            "top_ids": ids, "top_scores": scores, "future": future, "owner": owner}


def score(rows: dict, boost: float, slope: float) -> dict:
    candidate = rows["candidate_scores"] + boost + slope * (rows["top_scores"] - rows["top_scores"][:, :1])
    chosen = candidate.argmax(axis=1)
    first = np.where(candidate[np.arange(len(chosen)), chosen] > rows["outside_scores"],
                     rows["top_ids"][np.arange(len(chosen)), chosen], rows["outside_ids"])
    draft = np.column_stack((first, rows["later_ids"]))
    prefix = np.cumprod(draft == rows["future"], axis=1)
    accepted = prefix.sum(axis=1)
    owners = np.unique(rows["owner"])
    by_prompt = np.asarray([accepted[rows["owner"] == owner].mean() for owner in owners])
    return {"accepted_future_per_block": float(accepted.mean()),
            "accepted_future_per_prompt": float(by_prompt.mean()),
            "first_future_accuracy": float(np.mean(first == rows["future"][:, 0])),
            "per_prompt": by_prompt.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--validation-prior", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--test-prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Drafter(Config(**checkpoint["config"]["config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    validation = summaries(model, args.validation_data, args.validation_prior, device, args.batch)
    grid = []
    for boost in (0, 1, 2, 4, 8, 16, 32):
        for slope in (0, 0.25, 0.5, 1, 2, 4, 8):
            result = score(validation, boost, slope)
            grid.append({"boost": boost, "slope": slope, **result})
    chosen = max(grid, key=lambda r: (r["accepted_future_per_prompt"], -r["boost"], -r["slope"]))
    test = summaries(model, args.test_data, args.test_prior, device, args.batch)
    report = {"checkpoint": str(args.checkpoint), "selection": "validation accepted future per prompt",
              "validation_blocks": len(validation["future"]), "test_blocks": len(test["future"]),
              "chosen_boost": chosen["boost"], "chosen_slope": chosen["slope"],
              "validation": chosen, "test": score(test, chosen["boost"], chosen["slope"]),
              "test_no_prior": score(test, 0, 0), "validation_grid": grid,
              "note": "Offline greedy first-slot sparse prior; no NPU or end-to-end latency measurement."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"chosen_boost": chosen["boost"], "chosen_slope": chosen["slope"],
                      "test": report["test"], "test_no_prior": report["test_no_prior"]}), flush=True)


if __name__ == "__main__":
    main()
