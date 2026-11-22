"""Build full [512, 3072] conditioning from enricher samples for image checks (2026-11-10)."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

import enricher_smoke as e
import harness

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--targets", type=Path, required=True)
parser.add_argument("--score", type=Path, required=True)
parser.add_argument("--subjects", nargs="+", required=True, help="group--base_id keys in samples.safetensors")
parser.add_argument("--per-subject", type=int, default=3)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

cases = {c["id"]: c for c in json.loads((args.targets / "manifest.json").read_text(encoding="utf-8"))["cases"]}
pairs = e.load_pairs(args.targets)
by = {}
for p in pairs:
    by.setdefault(p["base_id"], []).append(p)
samples = load_file(str(args.score / "samples.safetensors"))
report = json.loads((args.score / "score.json").read_text(encoding="utf-8"))
args.output.mkdir(parents=True, exist_ok=False)
index = []
for key in args.subjects:
    group, base_id = key.split("--", 1)
    items = by[base_id]
    k, base = items[0]["k"], items[0]["base_full"]
    teach = torch.stack([p["window"] for p in items])
    name = base_id[2:]
    base_file = args.targets / cases[base_id]["file"]
    # Native files carry the token IDs/mask of the prompt passed as --prompt: the base prompt for samples.
    harness.save_prediction(base_file, base.to(torch.bfloat16), args.output / f"{name}-0base.safetensors")
    for i in range(args.per_subject):
        win = samples[key][i]
        j = int(((win[None] - teach) ** 2).flatten(1).sum(1).argmin())
        full = base.clone()
        # Padding rows after the window stay as the base prompt's rows; stage 1 showed they carry little.
        full[k:k + e.WINDOW] = win
        harness.save_prediction(base_file, full.to(torch.bfloat16), args.output / f"{name}-s{i}-sample.safetensors")
        harness.save_prediction(args.targets / items[j]["file"], items[j]["full"].to(torch.bfloat16),
                                args.output / f"{name}-s{i}-teacher.safetensors")
        index.append({"subject": base_id, "group": group, "sample": i, "nearest_phrase": e.PHRASES[items[j]["phrase"]],
                      "nearest_split": items[j]["split"], "padding": "base",
                      "sample_prompt": cases[base_id]["prompt"], "teacher_prompt": cases[items[j]["id"]]["prompt"]})
(args.output / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
for row in index:
    print(row["subject"], row["sample"], row["nearest_phrase"], row["nearest_split"])
