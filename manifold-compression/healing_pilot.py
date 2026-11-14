"""Matched, bounded whole-model healing with complete assistant responses."""
import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer

from audit_rotation import compare, save
from healing_data import chat_ids, encode_complete, check_answer, generation_metrics, scoring_spec
from projection_models import load_causal_model

SOURCES = {"qwen35": "models/manifold-audit/Qwen3.5-2B",
           "smollm2": "models/manifold-audit/SmolLM2-1.7B-Instruct"}
REPORTS = {"qwen35": "qwen35-2b-1792.json", "smollm2": "smollm2-1.7b-1792.json"}


def source_student(arch):
    return f"models/manifold-compression/output-fix-{arch}-1792"


def healed_path(arch, arm):
    return Path(f"models/manifold-compression/healing-pilot-{arch}-{arm}")


def checkpoint_signature(path):
    path = Path(path)
    files = sorted(path.glob("*.safetensors")) + [path / "config.json"]
    stamp = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    return hashlib.sha256(json.dumps(stamp).encode()).hexdigest()


def load_checked(path):
    model, info = load_causal_model(path, dtype=torch.float32, local_files_only=True,
                                   attn_implementation="eager", output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Unmatched checkpoint keys: {info}")
    return model.eval().requires_grad_(False)


@torch.no_grad()
def legacy_initialization(model, rms_scale):
    """Same trainable adapter capacity; change only its starting function.

    The saved corrected matrix is c*M, so diag(saved)/c**2 reproduces
    diag(M)/c. Both arms may learn off-diagonal entries during healing.
    """
    norm = model.model.norm
    if not torch.all(norm.weight + norm.offset == 1):
        raise ValueError("Legacy reconstruction requires an untouched initialization")
    norm.matrix.copy_(torch.diag(norm.matrix.diag().clone() / rms_scale**2))


def student_for_arm(arch, arm):
    model = load_checked(source_student(arch))
    if arm == "legacy":
        report = json.loads((Path("manifold-compression/results/output-fix-20261024") / REPORTS[arch]).read_text())
        legacy_initialization(model, report["rms_scales"][-1])
    return model


def encoded_rows(data, tokenizer, split):
    output = []
    for row in data[split]:
        encoded = encode_complete(tokenizer, row["messages"], data["max_length"])
        if encoded is None:
            raise ValueError("Prepared row unexpectedly exceeds the complete-response length limit")
        output.append({"id": row["id"], **encoded})
    return output


def make_batch(rows, pad):
    width = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    attention = torch.zeros_like(ids)
    positions, targets, batch_indices = [], [], []
    for b, row in enumerate(rows):
        n = len(row["input_ids"])
        ids[b, :n] = torch.tensor(row["input_ids"])
        attention[b, :n] = 1
        positions.extend(row["positions"])
        targets.extend(row["targets"])
        batch_indices.extend([b] * len(row["positions"]))
    return ids, attention, torch.tensor(batch_indices), torch.tensor(positions), torch.tensor(targets)


def cache_teacher(args, data, tokenizer, cache_path):
    model = load_checked(SOURCES[args.arch])
    result = {"source": SOURCES[args.arch], "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "head": model.lm_head.weight.detach(), "rows": {}}
    start = time.perf_counter()
    with torch.no_grad():
        for split in ("train", "validation"):
            for i, row in enumerate(encoded_rows(data, tokenizer, split)):
                ids = torch.tensor([row["input_ids"]])
                hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
                result["rows"][row["id"]] = hidden[0, row["positions"]].float().contiguous()
                if (i + 1) % 8 == 0:
                    print(f"cache {split} {i + 1}: {time.perf_counter() - start:.1f}s", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    torch.save(result, temporary)
    temporary.replace(cache_path)
    save({"source": result["source"], "data_sha256": result["data_sha256"],
          "rows": len(result["rows"]), "elapsed_seconds": time.perf_counter() - start},
         args.out / f"{args.arch}-teacher-cache.json")


def load_cache(args, path):
    cache = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if cache["data_sha256"] != hashlib.sha256(args.data.read_bytes()).hexdigest():
        raise ValueError("Teacher cache was built from a different dataset")
    if cache["source"] != SOURCES[args.arch]:
        raise ValueError("Teacher cache belongs to a different architecture")
    return cache


def loss_parts(student_logits, teacher_logits, targets):
    slp = student_logits.float().log_softmax(-1)
    with torch.no_grad():
        tlp = teacher_logits.float().log_softmax(-1)
    kl = (tlp.exp() * (tlp - slp)).sum()
    ce = F.nll_loss(slp, targets, reduction="sum")
    return kl, ce


@torch.no_grad()
def validation(model, rows, cache):
    model.eval()
    kl_sum = ce_sum = eos_nll_sum = 0.
    n = same = 0
    for row in rows:
        ids = torch.tensor([row["input_ids"]])
        hs = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, row["positions"]]
        ht = cache["rows"][row["id"]]
        targets = torch.tensor(row["targets"])
        for i in range(0, len(targets), 32):
            student = model.lm_head(hs[i:i + 32]).float()
            teacher = F.linear(ht[i:i + 32], cache["head"]).float()
            if not torch.isfinite(student).all():
                raise RuntimeError("Non-finite validation logits")
            kl, ce = loss_parts(student, teacher, targets[i:i + 32])
            kl_sum += kl.item()
            ce_sum += ce.item()
            same += (student.argmax(-1) == teacher.argmax(-1)).sum().item()
        last = model.lm_head(hs[-1:]).float().log_softmax(-1)
        eos_nll_sum += -last[0, row["eos_token_id"]].item()
        n += len(targets)
    return {"assistant_tokens": n, "kl": kl_sum / n, "nll": ce_sum / n,
            "eos_nll": eos_nll_sum / len(rows), "teacher_top1_agreement": same / n}


def train(args, data, tokenizer, cache):
    model = student_for_arm(args.arch, args.arm)
    train_rows = encoded_rows(data, tokenizer, "train")
    val_rows = encoded_rows(data, tokenizer, "validation")
    if args.steps * args.rows_per_step > len(train_rows):
        raise ValueError("Pilot exceeds its one-pass data budget")
    history = {"arch": args.arch, "arm": args.arm, "data_sha256": cache["data_sha256"],
               "settings": {"lr": args.lr, "steps": args.steps, "rows_per_step": args.rows_per_step,
                            "ce_weight": args.ce_weight, "precision": "FP32 parameters; CPU BF16 autocast training; FP32 evaluation",
                            "seed": 137, "head_chunk": args.chunk, "trainable": "all parameters, including matrix in both arms"},
               "steps": [], "spend_usd": 0}
    report_path = args.out / f"{args.arch}-{args.arm}{'-benchmark' if args.benchmark else ''}-training.json"
    if not args.benchmark:
        history["before"] = validation(model, val_rows, cache)
        print("before", history["before"], flush=True)
    model.requires_grad_(True).train()
    training_head = cache["head"].to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0., betas=(.9, .95), foreach=True)
    start = time.perf_counter()
    for step in range(args.steps):
        tick = time.perf_counter()
        rows = train_rows[step * args.rows_per_step:(step + 1) * args.rows_per_step]
        ids, attention, bi, pos, targets = make_batch(rows, tokenizer.pad_token_id or tokenizer.eos_token_id)
        ht = torch.cat([cache["rows"][r["id"]] for r in rows])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            hs = model.model(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state[bi, pos]
            def chunk_loss(s, t, labels):
                with torch.no_grad():
                    teacher = F.linear(t, training_head)
                student = model.lm_head(s)
                kl, ce = loss_parts(student, teacher, labels)
                return kl + args.ce_weight * ce
            loss = sum(checkpoint(chunk_loss, hs[i:i + args.chunk], ht[i:i + args.chunk], targets[i:i + args.chunk],
                                  use_reentrant=False) for i in range(0, len(targets), args.chunk)) / len(targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * min(1., (step + 1) / 2)
        optimizer.step()
        row = {"step": step + 1, "loss": loss.item(), "grad_norm": grad_norm.item(),
               "input_tokens": sum(len(r["input_ids"]) for r in rows), "assistant_tokens": len(targets),
               "ending_tokens": len(rows), "elapsed_seconds": time.perf_counter() - tick}
        history["steps"].append(row)
        save(history, report_path)
        print(json.dumps(row), flush=True)
    history["training_seconds"] = time.perf_counter() - start
    if not args.benchmark:
        del optimizer, hs, ht, loss
        gc.collect()
        model.eval().requires_grad_(False)
        history["after"] = validation(model, val_rows, cache)
        print("after", history["after"], flush=True)
        destination = healed_path(args.arch, args.arm)
        if destination.exists():
            raise ValueError(f"Refusing to overwrite an existing healed checkpoint: {destination}")
        temporary = destination.with_name(destination.name + ".incomplete")
        model.save_pretrained(temporary)
        tokenizer.save_pretrained(temporary)
        ids = torch.tensor([val_rows[0]["input_ids"][:32]])
        with torch.no_grad():
            expected = model(ids, use_cache=False).logits
        del model
        gc.collect()
        restored = load_checked(temporary)
        with torch.no_grad():
            history["roundtrip"] = compare([expected], [restored(ids, use_cache=False).logits])
        if not history["roundtrip"]["numerical_pass"]:
            raise RuntimeError("Healed checkpoint failed reload verification")
        temporary.replace(destination)
        history["checkpoint"] = str(destination)
    save(history, report_path)


@torch.no_grad()
def evaluate(args, data, tokenizer):
    out = args.out / f"{args.arch}-{args.label}-generation.json"
    data_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if args.label == "stock":
        model_path = SOURCES[args.arch]
    elif args.label == "recovered-500m":
        model_path = "models/manifold-compression/distill-08b/continuation-500m/ckpt_500M"
    elif args.label.endswith("before"):
        model_path = source_student(args.arch)
    else:
        model_path = healed_path(args.arch, args.label.split("-")[0])
    signature = checkpoint_signature(model_path)
    if out.exists() and not args.force_generation:
        previous = json.loads(out.read_text())
        if (previous.get("summary", {}).get("cases") == len(data["generation_eval"])
                and previous.get("checkpoint_signature") == signature
                and previous.get("data_sha256") == data_hash
                and previous.get("max_new_tokens") == args.max_new
                and previous.get("generation_batch") == args.generation_batch):
            print(f"Reusing completed evaluation: {out}", flush=True)
            return
    if args.label == "stock":
        model = load_checked(SOURCES[args.arch])
    elif args.label == "recovered-500m":
        if args.arch != "qwen35":
            raise ValueError("The recovered checkpoint is Qwen-specific")
        model = load_checked("models/manifold-compression/distill-08b/continuation-500m/ckpt_500M")
    elif args.label.endswith("before"):
        model = student_for_arm(args.arch, args.label.split("-")[0])
    else:
        model = load_checked(healed_path(args.arch, args.label.split("-")[0]))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    results = {"arch": args.arch, "label": args.label, "max_new_tokens": args.max_new,
               "decoding": "greedy, thinking disabled, no repetition penalty", "precision": "float32",
               "data_sha256": data_hash, "checkpoint_signature": signature,
               "generation_batch": args.generation_batch, "threads": args.threads, "cases": []}
    start = time.perf_counter()
    # Group anticipated lengths to avoid decoding many padded, already-ended
    # rows alongside long answers. This order is fixed before seeing outputs.
    order = ["json", "copy", "multiply", "sort", "units", "logic", "multi-memory", "multi-edit",
             "five-lines", "poem", "summary", "alias", "ice", "interview", "dog", "hash-table"]
    cases = sorted(data["generation_eval"], key=lambda c: order.index(c["id"]))
    for i in range(0, len(cases), args.generation_batch):
        batch_cases = cases[i:i + args.generation_batch]
        texts = [tokenizer.apply_chat_template(c["messages"], tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False) for c in batch_cases]
        batch = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt")
        generated = model.generate(**batch, max_new_tokens=args.max_new, do_sample=False, use_cache=True,
                                   eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        for c, seq in zip(batch_cases, generated[:, batch.input_ids.shape[1]:], strict=True):
            ids = seq.tolist()
            if tokenizer.eos_token_id in ids:
                ids = ids[:ids.index(tokenizer.eos_token_id) + 1]
            answer = tokenizer.decode(ids, skip_special_tokens=True)
            spec = scoring_spec(c)
            results["cases"].append({"id": c["id"], "answer": answer, "check": spec,
                                     "check_pass": check_answer(answer, spec),
                                     **generation_metrics(answer, ids, tokenizer.eos_token_id)})
        print(f"{args.arch} {args.label}: generated {len(results['cases'])}/{len(cases)}", flush=True)
        save(results, out)
    n = len(cases)
    results["summary"] = {"cases": n, "check_passes": sum(c["check_pass"] for c in results["cases"]),
                          "completed": sum(c["ended"] for c in results["cases"]),
                          "loop_flags": sum(c["loop_flag"] for c in results["cases"]),
                          "generated_tokens": sum(c["generated_tokens"] for c in results["cases"]),
                          "elapsed_seconds": time.perf_counter() - start}
    if args.label == "recovered-500m":
        cache = load_cache(args, args.cache_dir / f"{args.arch}.pt")
        results["validation"] = validation(model, encoded_rows(data, tokenizer, "validation"), cache)
    save(results, out)
    print(results["summary"], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["cache", "train", "evaluate"])
    ap.add_argument("--arch", choices=list(SOURCES), required=True)
    ap.add_argument("--arm", choices=["corrected", "legacy"], default="corrected")
    ap.add_argument("--label", choices=["stock", "corrected-before", "legacy-before", "corrected-after", "legacy-after", "recovered-500m"], default="stock")
    ap.add_argument("--data", type=Path, default=Path("manifold-compression/results/healing-pilot-20261024/data.json"))
    ap.add_argument("--out", type=Path, default=Path("manifold-compression/results/healing-pilot-20261024"))
    ap.add_argument("--cache-dir", type=Path, default=Path("target/manifold-audit/healing-cache"))
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--rows-per-step", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--ce-weight", type=float, default=.1)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--generation-batch", type=int, default=4)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--force-generation", action="store_true", help="Regenerate even if an identical completed evaluation exists")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(137)
    random.seed(137)
    args.out.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.data.read_text())
    tok = AutoTokenizer.from_pretrained(SOURCES[args.arch], local_files_only=True)
    cache_path = args.cache_dir / f"{args.arch}.pt"
    if args.stage == "cache":
        cache_teacher(args, data, tok, cache_path)
    elif args.stage == "train":
        train(args, data, tok, load_cache(args, cache_path))
    else:
        evaluate(args, data, tok)


if __name__ == "__main__":
    main()
