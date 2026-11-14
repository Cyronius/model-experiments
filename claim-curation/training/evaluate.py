#!/usr/bin/env python3
"""Paired generation evaluation of base and adapted NuExtract3 on held-out docs."""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path


def compare(expected, actual, document):
    from teacher import mechanically_validate
    errors = mechanically_validate(actual, document)
    claims = actual.get("claims", []) if isinstance(actual, dict) else []
    if errors:
        return {"schema_and_citations_valid": False, "errors": errors,
                "reference_quotes_covered": 0, "reference_qualifiers_covered": 0,
                "reference_claims": len(expected["claims"])}
    quoted, scoped = 0, 0
    for reference in expected["claims"]:
        matches = [c for c in claims if (reference["quote"] in c["quote"] or c["quote"] in reference["quote"])
                   and len(c["quote"].split()) >= 3]
        quoted += bool(matches)
        conditions = reference["conditions"]
        scoped += any(all(condition.lower() in (c["statement"] + " " + " ".join(c["conditions"])).lower()
                          for condition in conditions) for c in matches)
    return {"schema_and_citations_valid": True, "errors": [],
            "reference_quotes_covered": quoted, "reference_qualifiers_covered": scoped,
            "reference_claims": len(expected["claims"]),
            "generated_claims": len(claims),
            "warning": "Quotation/qualifier coverage is diagnostic, not semantic entailment or fully adjudicated accuracy."}


def run(args):
    import torch
    from runtime import configure
    runtime_settings = configure(torch)
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from run_models import render
    from train_lora import load_rows, verify_checkpoint
    verify_checkpoint(args.checkpoint)
    torch.set_num_threads(args.threads)
    torch.manual_seed(3271)
    rows = [r for r in load_rows(args.labels) if r["split"] in args.splits]
    if not rows:
        raise ValueError("No accepted held-out examples")
    args.out.mkdir(parents=True, exist_ok=True)
    prompt = json.loads((project / "prompt.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, padding_side="left")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to(device).eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    results = []
    for row in rows:
        text = render(tokenizer, {"task": "extraction", "text": row["document"]}, prompt, "nuextract3")
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_tokens,
                                    eos_token_id=[248044, 248046], pad_token_id=tokenizer.pad_token_id)
        ids = output[0, inputs.input_ids.shape[1]:].tolist()
        stopped = any(i in (248044, 248046) for i in ids)
        raw = tokenizer.decode(ids, skip_special_tokens=True).strip()
        try:
            actual = json.loads(raw)
            metrics = compare(row["target"], actual, row["document"])
        except (ValueError, TypeError) as exc:
            actual, metrics = None, {"schema_and_citations_valid": False, "errors": [str(exc)]}
        record = {"id": row["id"], "source_key": row["source_key"], "source_sha256": row["source_sha256"],
                  "split": row["split"], "adapter": str(args.adapter) if args.adapter else None,
                  "raw_output": raw, "prediction": actual, "metrics": metrics,
                  "truncated": not stopped, "teacher_reference": row["target"]}
        record["reference_annotation_provenance"] = row.get("provenance", "Unspecified; inspect retained labels")
        results.append(record)
        (args.out / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
        print(row["id"], json.dumps(metrics), flush=True)
    summary = {"cases": len(results), "schema_and_citations_valid": sum(r["metrics"]["schema_and_citations_valid"] for r in results),
               "truncated": sum(r["truncated"] for r in results), "training_examples_evaluated": 0,
               "reference_quotes_covered": sum(r["metrics"].get("reference_quotes_covered", 0) for r in results),
               "reference_qualifiers_covered": sum(r["metrics"].get("reference_qualifiers_covered", 0) for r in results),
               "reference_claims": sum(len(r["teacher_reference"]["claims"]) for r in results),
               "adapter": str(args.adapter) if args.adapter else None, "runtime": runtime_settings,
               "interpretation": "Reference provenance is retained per case; this pilot's held-out references are agent-authored, not independent human gold. Quotation overlap and exact qualifier matching are diagnostics, not semantic accuracy. Compare raw outputs against original sources.",
               "paid_compute_cost_usd": 0}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--splits", nargs="+", choices=["validation", "test"], default=["test"])
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=3072)
    run(p.parse_args())
