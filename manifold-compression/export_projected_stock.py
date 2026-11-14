"""Export a matrix-adapter checkpoint as an ordinary, untied-head HF model.

This explicitly trades storage for compatibility with stock conversion tools.
The input checkpoint is never overwritten. FP32 materialization is checked
before optional BF16 storage rounding; no GGUF/runtime parity is implied.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from audit_rotation import compare
from projection_models import load_causal_model, materialize_stock_output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        ap.error("output directory must be empty")
    torch.set_num_threads(args.threads)
    model = load_causal_model(args.model, dtype=torch.float32).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer("Explain why the seasons change during the year.", return_tensors="pt").input_ids
    with torch.no_grad():
        before = model(ids, use_cache=False).logits
        params_before = sum(p.numel() for p in model.parameters())
        materialize_stock_output(model)
        after = model(ids, use_cache=False).logits
        report = {"source": args.model, "stored_dtype": args.dtype, "fp32_fold_check": compare([before], [after]),
                  "parameters_before": params_before, "parameters_after": sum(p.numel() for p in model.parameters())}
        if not report["fp32_fold_check"]["numerical_pass"]:
            raise RuntimeError(f"Output materialization changed the model: {report}")
        model.to(getattr(torch, args.dtype)).save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    (args.out / "materialization_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
