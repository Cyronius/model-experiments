#!/usr/bin/env python3
"""Bounded NuExtract3 text-only LoRA pilot with frozen vision/base weights."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

STUDENT_REVISION = "c99dc8f5641b866aa0192b6ea78f84bf9f3535f1"


def fp(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    eligible = [r for r in rows if r.get("training_eligible")]
    groups, hashes = {}, {}
    for row in rows:
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError("Unknown data split")
        if row["source_key"] in groups and groups[row["source_key"]] != row["split"]:
            raise ValueError("Source document leaks across splits")
        groups[row["source_key"]] = row["split"]
        sha = row.get("source_sha256")
        if sha:
            if sha in hashes and hashes[sha] != row["split"]:
                raise ValueError("Duplicate document content leaks across splits")
            hashes[sha] = row["split"]
    return eligible


def verify_checkpoint(path):
    meta = path / ".cache/huggingface/download/config.json.metadata"
    if not meta.exists() or meta.read_text().splitlines()[0] != STUDENT_REVISION:
        raise ValueError("Student checkpoint does not match pinned download revision")


def encode_example(tokenizer, row, prompt_config, max_length):
    # Imported only after installation in claim-curation/training/.
    from run_models import render
    rendered = render(tokenizer, {"task": "extraction", "text": row["document"]}, prompt_config, "nuextract3")
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    target = tokenizer.encode(json.dumps(row["target"], ensure_ascii=False), add_special_tokens=False)
    ids = prefix + target + [tokenizer.eos_token_id]
    if len(ids) > max_length:
        raise ValueError(f'{row["id"]}: {len(ids)} tokens exceed {max_length}; split the source, never truncate evidence')
    return {"input_ids": ids, "labels": [-100] * len(prefix) + target + [tokenizer.eos_token_id],
            "id": row["id"], "split": row["split"], "source_key": row["source_key"]}


def target_names(model, last_layers):
    import torch
    count = model.config.text_config.num_hidden_layers
    first = count - last_layers
    selected = []
    pattern = re.compile(r"model\.language_model\.layers\.(\d+)\.(mlp\.(gate_proj|up_proj|down_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|out_proj))$")
    for name, module in model.named_modules():
        match = pattern.fullmatch(name)
        if match and int(match[1]) >= first and isinstance(module, torch.nn.Linear):
            selected.append(name)
    if not selected or any("visual" in name for name in selected):
        raise ValueError("No valid text-only adapter targets found")
    return selected


def run(args):
    import torch
    from runtime import configure
    runtime_settings = configure(torch)
    import transformers
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from teacher import mechanically_validate
    verify_checkpoint(args.checkpoint)
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    prompt = json.loads((project / "prompt.json").read_text())
    rows = load_rows(args.labels)
    for row in rows:
        errors = mechanically_validate(row["target"], row["document"])
        if errors:
            raise ValueError(f'{row["id"]}: invalid supervision: {errors}')
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, padding_side="left")
    # Held-out test labels never enter optimization, validation, or length selection.
    encoded = [encode_example(tokenizer, r, prompt, args.max_length) for r in rows if r["split"] != "test"]
    train = [r for r in encoded if r["split"] == "train"]
    validation = [r for r in encoded if r["split"] == "validation"]
    if not train or not validation:
        raise ValueError("Need accepted training and document-disjoint validation examples")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.bfloat16
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.checkpoint, dtype=dtype, attn_implementation="sdpa", local_files_only=True).to(device)
    base.config.use_cache = False
    for p in base.parameters():
        p.requires_grad_(False)
    names = target_names(base, args.last_layers)
    if args.resume:
        model = PeftModel.from_pretrained(base, args.resume, is_trainable=True)
    else:
        config = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0,
                            target_modules=names, bias="none", task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(base, config)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n or "visual" in n for n, _ in trainable):
        raise AssertionError("Unexpected trainable base/vision parameter")
    args.out.mkdir(parents=True, exist_ok=True)
    signature = fp({"labels": [r for r in rows if r["split"] != "test"], "prompt": prompt, "targets": names,
                    "rank": args.rank, "lr": args.lr, "grad_accum": args.grad_accum,
                    "seed": args.seed, "max_length": args.max_length})
    manifest = {"student_revision": STUDENT_REVISION, "signature": signature,
                "train_examples": len(train), "validation_examples": len(validation),
                "test_examples_excluded": sum(r["split"] == "test" for r in rows),
                "trainable_parameters": sum(p.numel() for _, p in trainable),
                "target_modules": names, "vision_frozen": True, "base_frozen": True,
                "device": device, "torch": torch.__version__, "transformers": transformers.__version__,
                "runtime": runtime_settings,
                "steps_requested": args.steps, "paid_compute_cost_usd": 0}
    (args.out / "training-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0)
    start = 0
    if args.resume:
        state = torch.load(args.resume / "optimizer.pt", map_location=device, weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Resume data/config signature differs")
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]

    def batch(row):
        return {k: torch.tensor([row[k]], dtype=torch.long, device=device) for k in ("input_ids", "labels")}

    def validate():
        model.eval()
        losses = []
        with torch.no_grad():
            for row in validation:
                losses.append(float(model(**batch(row), use_cache=False).loss))
        model.train()
        return sum(losses) / len(losses)

    before = validate()
    history = [{"step": start, "validation_loss": before}]
    print("VALIDATION_BEFORE", before, flush=True)
    model.train()
    initial = {n: p.detach().cpu().clone() for n, p in trainable}
    for step in range(start, args.steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(args.grad_accum):
            index = step * args.grad_accum + micro
            epoch, position = divmod(index, len(train))
            order = list(range(len(train)))
            random.Random(args.seed + epoch).shuffle(order)
            loss = model(**batch(train[order[position]]), use_cache=False).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            (loss / args.grad_accum).backward()
            losses.append(float(loss.detach()))
        norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0))
        if not math.isfinite(norm):
            raise FloatingPointError("Non-finite adapter gradients")
        optimizer.step()
        record = {"step": step + 1, "loss": sum(losses) / len(losses), "gradient_norm": norm}
        history.append(record)
        print(json.dumps(record), flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            checkpoint = args.out / f"step-{step + 1}"
            model.save_pretrained(checkpoint, safe_serialization=True)
            torch.save({"optimizer": optimizer.state_dict(), "step": step + 1, "signature": signature}, checkpoint / "optimizer.pt")
            (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    after = validate()
    changed = [n for n, p in trainable if not torch.equal(initial[n], p.detach().cpu())]
    if args.steps > start and not changed:
        raise AssertionError("No adapter tensors changed")
    model.save_pretrained(args.out / "final", safe_serialization=True)
    tokenizer.save_pretrained(args.out / "final")
    result = {**manifest, "steps_completed": args.steps, "validation_loss_before": before,
              "validation_loss_after": after, "changed_adapter_tensors": len(changed),
              "interpretation": "Teacher-forced validation loss is not semantic extraction accuracy; generation evaluation is required."}
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--last-layers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-length", type=int, default=5120)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=3271)
    p.add_argument("--save-every", type=int, default=8)
    run(p.parse_args())
