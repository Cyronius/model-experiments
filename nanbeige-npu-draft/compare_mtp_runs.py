"""Paired prompt-level comparison of base-hidden and MTP-hidden tiny drafters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--mtp-data", type=Path, required=True)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--mtp-report", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.base_data) as left, np.load(args.mtp_data) as right:
        for key in ("seed", "future", "owner", "anchor"):
            if not np.array_equal(left[key], right[key]):
                raise ValueError(f"mismatched comparison data: {key}")
        owners = left["owner"].copy()
    base = json.loads(args.base_report.read_text())["reports"]["1_pass"]
    mtp = json.loads(args.mtp_report.read_text())["reports"]["1_pass"]
    groups = np.unique(owners)
    rows = [json.loads(line) for line in args.generated.read_text().splitlines() if line]
    if len(groups) != len(base["per_prompt_accepted_future"]) or len(groups) != len(mtp["per_prompt_accepted_future"]):
        raise ValueError("prompt-level scores do not match the holdout")
    left = np.asarray(base["per_prompt_accepted_future"], np.float64)
    right = np.asarray(mtp["per_prompt_accepted_future"], np.float64)
    delta = right - left
    rng = np.random.default_rng(1101)
    draws = rng.integers(len(groups), size=(10000, len(groups)))
    ci = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist()
    domains = {}
    for label in sorted({rows[int(g)]["id"].split("-", 1)[0] for g in groups}):
        mask = np.asarray([rows[int(g)]["id"].split("-", 1)[0] == label for g in groups])
        domains[label] = {"prompts": int(mask.sum()), "base": float(left[mask].mean()),
                          "mtp": float(right[mask].mean()), "delta": float(delta[mask].mean())}
    result = {"prompts": len(groups), "blocks": len(owners),
              "base_accepted_future_per_block": base["accepted_future_per_block"],
              "mtp_accepted_future_per_block": mtp["accepted_future_per_block"],
              "paired_prompt_mean_delta": float(delta.mean()), "paired_prompt_95ci": ci,
              "domains": domains,
              "note": "Offline greedy prefix match, not a GPU+NPU latency measurement."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
