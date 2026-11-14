"""Check unpadded, batched teacher export against padded 512-row targets on real rows (2026-11-11)."""
import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

import harness

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--targets", type=Path, required=True)
parser.add_argument("--limit", type=int, default=128)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(args.threads)

from transformers import AutoTokenizer, Qwen3Model

manifest = json.loads((args.targets / "manifest.json").read_text(encoding="utf-8"))
cases = manifest["cases"][: args.limit]
tok = AutoTokenizer.from_pretrained(args.model / "tokenizer", local_files_only=True)
model = Qwen3Model.from_pretrained(args.model / "text_encoder", local_files_only=True,
                                   torch_dtype=torch.bfloat16, attn_implementation="eager").eval()
weights, _ = harness.projection_files(args.model)
weight = weights["context_embedder.weight"].to(torch.bfloat16)
model(torch.tensor([[1, 2, 3]]))

rows, t0 = [], time.perf_counter()
for i in range(0, len(cases), args.batch):
    chunk = cases[i:i + args.batch]
    texts = [harness.chat_text(c["prompt"]) for c in chunk]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    with torch.inference_mode():
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        cond = torch.nn.functional.linear(torch.cat([hs[k] for k in harness.TAPS], -1), weight)
    for j, c in enumerate(chunk):
        n = int(enc["attention_mask"][j].sum())
        ref = load_file(str(args.targets / c["file"]))
        if not torch.equal(ref["input_ids"][:n], enc["input_ids"][j, :n]):
            raise ValueError(f"token mismatch {c['id']}")
        a, b = cond[j, :n].float(), ref["projected_conditioning"][:n].float()
        rows.append({"id": c["id"], "rows": n, "bit_identical": bool(torch.equal(cond[j, :n], ref["projected_conditioning"][:n])),
                     "max_abs": float((a - b).abs().max()), "rel_l2": float((a - b).norm() / b.norm()),
                     "rel_l2_excl_row0": float((a[1:] - b[1:]).norm() / b[1:].norm())})
elapsed = time.perf_counter() - t0
summary = {"cases": len(rows), "batch": args.batch, "threads": args.threads, "seconds": elapsed,
           "seconds_per_prompt": elapsed / len(rows),
           "bit_identical": sum(r["bit_identical"] for r in rows),
           "worst_rel_l2_excl_row0": max(r["rel_l2_excl_row0"] for r in rows),
           "median_rel_l2_excl_row0": sorted(r["rel_l2_excl_row0"] for r in rows)[len(rows) // 2]}
args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=1))
