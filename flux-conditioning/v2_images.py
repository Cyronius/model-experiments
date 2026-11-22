"""Build native conditioning files for enricher v2 image checks: base, sample on base padding, nearest teacher (2026-11-14)."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

import harness
from export_pairs import TAIL

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--score", type=Path, required=True, help="Folder with image-index-test.json and image-samples-test.safetensors")
parser.add_argument("--teacher", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--guidance", type=float, nargs="+", help="Keep only samples with these guidance strengths")
parser.add_argument("--no-teacher", action="store_true", help="Skip the nearest-teacher files")
parser.add_argument("--targets", type=Path, help="Reuse an earlier export folder that holds the base prompts")
parser.add_argument("--threads", type=int, default=12)
args = parser.parse_args()
torch.set_num_threads(args.threads)

index = json.loads((args.score / "image-index-test.json").read_text(encoding="utf-8"))
if args.guidance:
    index = [r for r in index if r.get("guidance", 1.0) in args.guidance]
samples = load_file(str(args.score / "image-samples-test.safetensors"))
args.output.mkdir(parents=True, exist_ok=False)
targets = args.targets or args.output / "targets"
if not args.targets:
    cases, prompts = [], set()
    for r in index:
        wanted = [(f"b-{r['group']}", r["base"])]
        if not args.no_teacher:
            wanted.append((f"t-{r['key']}", f"{r['base']}, {r['nearest_detail']}"))
        for cid, prompt in wanted:
            if prompt not in prompts:
                prompts.add(prompt)
                cases.append({"id": cid, "prompt": prompt, "split": "test", "seed": 17})
    (args.output / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    harness.export(argparse.Namespace(cases=args.output / "cases.jsonl", output=targets, model=args.teacher, dtype="fp32", device="cpu"))
manifest = {c["prompt"]: c for c in json.loads((targets / "manifest.json").read_text(encoding="utf-8"))["cases"]}
out = args.output / "cond"
for r in index:
    base_case = manifest[r["base"]]
    # The shard's first changed row must match a fresh tokenization of the base prompt.
    assert base_case["untruncated_tokens"] - TAIL == r["k"], (r["key"], base_case["untruncated_tokens"], r["k"])
    base_file = targets / base_case["file"]
    base = load_file(str(base_file))["projected_conditioning"].float()
    if not (out / f"{r['group']}-0base.safetensors").exists():
        harness.save_prediction(base_file, base.to(torch.bfloat16), out / f"{r['group']}-0base.safetensors")
    full = base.clone()
    # Padding rows after the window stay as the base prompt's rows (the v2 padding policy).
    full[r["k"]:r["k"] + r["rows"]] = samples[r["key"]].float()
    harness.save_prediction(base_file, full.to(torch.bfloat16), out / f"{r['key']}-sample.safetensors")
    if args.no_teacher:
        continue
    t_case = manifest[f"{r['base']}, {r['nearest_detail']}"]
    teacher = load_file(str(targets / t_case["file"]))["projected_conditioning"].to(torch.bfloat16)
    harness.save_prediction(targets / t_case["file"], teacher, out / f"{r['key']}-teacher.safetensors")
print(f"{len(index)} samples -> {out}")
