"""Run the encoded-memory smoke test without editing the frozen recipient."""
import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import BM25, ROOT, digest, exact_match, numeric_match, read_jsonl, write_jsonl
from memory import EncoderConfig, MemoryEncoder, answer_loss, greedy_answer, make_batch

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_recipient(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    # Resolve a local snapshot first. Some Transformers adapter-discovery paths
    # otherwise issue HEAD requests even with local_files_only=True.
    source = args.model
    if args.local_files_only and not Path(source).is_dir():
        source = snapshot_download(source, cache_dir=args.cache_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=args.cache_dir,
                                               local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(source, cache_dir=args.cache_dir, dtype=dtype,
                                                local_files_only=args.local_files_only,
                                                attn_implementation="sdpa").to(device).eval()
    if model.config.model_type != "llama":
        raise ValueError(f"This experiment expects a conventional Llama recipient, got {model.config.model_type}")
    model.requires_grad_(False)
    model._smoke_model_revision = (getattr(model.config, "_commit_hash", None) or
                                   (Path(source).name if len(Path(source).name) == 40 else None))
    model._smoke_fingerprint = model_fingerprint(model)
    return model, tokenizer


def model_fingerprint(model):
    """Complete hash before/after training; parameters stay frozen, including embeddings."""
    result = hashlib.sha256()
    for name, param in model.named_parameters():
        result.update(name.encode())
        result.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def document_features(model, tokenizer, documents, args):
    """Frozen contextual features, keyed by content and model revision. No questions."""
    device = model.get_input_embeddings().weight.device
    signature = {"model": args.model, "recipient_sha256": model._smoke_fingerprint,
                 "layer": args.feature_layer, "max_tokens": args.max_doc_tokens,
                 "config": digest(model.config.to_json_string())}
    cache = Path(args.work) / "features" / digest(json.dumps(signature, sort_keys=True))[:16]
    cache.mkdir(parents=True, exist_ok=True)
    features, missing = {}, []
    for doc in documents:
        path = cache / (digest(doc["text"]) + ".pt")
        if path.exists():
            features[doc["id"]] = torch.load(path, map_location="cpu", weights_only=True)
        else:
            missing.append((doc, path))
    start = time.monotonic()
    for offset in range(0, len(missing), args.encode_batch_size):
        group = missing[offset:offset + args.encode_batch_size]
        tokenized = tokenizer([doc["text"] for doc, _ in group], return_tensors="pt", padding=True,
                              truncation=True, max_length=args.max_doc_tokens).to(device)
        # The decoder body avoids allocating the large vocabulary logits for documents.
        with torch.no_grad():
            output = model.model(**tokenized, output_hidden_states=True, use_cache=False)
        hidden = output.hidden_states[args.feature_layer]
        for i, (doc, path) in enumerate(group):
            length = int(tokenized.attention_mask[i].sum())
            # Store compact CPU tensors; the trainable encoder itself stays float32.
            value = {"features": hidden[i, :length].to(dtype=torch.float16, device="cpu").clone(),
                     "tokens": tokenized.input_ids[i, :length].cpu().clone(),
                     "source_tokens": len(tokenizer(doc["text"]).input_ids),
                     "text_sha256": digest(doc["text"])}
            torch.save(value, path)
            features[doc["id"]] = value
        print(f"encoded {min(offset + len(group), len(missing))}/{len(missing)} documents "
              f"({time.monotonic() - start:.1f}s)", flush=True)
    return features


def feature_batch(features, doc_ids, device):
    rows = [features[doc_id]["features"].float().to(device) for doc_id in doc_ids]
    masks = [torch.ones(len(row), dtype=torch.bool, device=device) for row in rows]
    return (torch.nn.utils.rnn.pad_sequence(rows, batch_first=True),
            torch.nn.utils.rnn.pad_sequence(masks, batch_first=True))


def train_encoder(model, tokenizer, documents, questions, args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    fit_docs = [doc for doc in documents if doc["split"] == "train"]
    train_q = [q for q in questions if q["split"] == "train"]
    if not fit_docs or not train_q:
        raise ValueError("No training documents/questions")
    if any(doc["domain"] != "synthetic" for doc in fit_docs):
        raise ValueError("Initial protocol keeps all real XDNA facts outside training")
    fit_ids = {doc["id"] for doc in fit_docs}
    if any(q["doc_id"] not in fit_ids for q in train_q):
        raise ValueError("Training question references a held-out document")
    features = document_features(model, tokenizer, fit_docs, args)
    embedding = model.get_input_embeddings().weight
    # Read a deterministic sample, rather than copying the entire vocabulary matrix.
    rms = embedding[:min(4096, len(embedding))].float().square().mean().sqrt().item()
    config = EncoderConfig(hidden_size=model.config.hidden_size, width=args.width, slots=args.slots,
                           max_tokens=args.max_doc_tokens, embedding_rms=rms)
    encoder = MemoryEncoder(config).to(embedding.device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=0.01)
    before_hash = model._smoke_fingerprint
    rng = random.Random(args.seed)
    log = []
    start = time.monotonic()
    for step in range(1, args.steps + 1):
        selected = rng.choices(train_q, k=args.batch_size)
        x, mask = feature_batch(features, [q["doc_id"] for q in selected], embedding.device)
        memories = encoder(x, mask)
        batch = make_batch(model, tokenizer, selected, memories, memory_site=args.memory_site)
        loss = answer_loss(model, batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        row = {"step": step, "loss": loss.item(), "grad_norm": float(grad_norm),
               "elapsed_seconds": time.monotonic() - start}
        log.append(row)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps(row), flush=True)
            write_jsonl(Path(args.out) / "training.jsonl", log)
    after_hash = model_fingerprint(model)
    if before_hash != after_hash or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Frozen recipient changed or accumulated gradients")
    checkpoint = {"config": encoder.specification(), "state_dict": encoder.cpu().state_dict(),
                  "model": args.model, "model_commit": model._smoke_model_revision,
                  "feature_layer": args.feature_layer, "recipient_sha256": after_hash,
                  "train_doc_ids": sorted(fit_ids),
                  "train_text_sha256": {d["id"]: digest(d["text"]) for d in fit_docs},
                  "dataset_sha256": digest(json.dumps(documents, sort_keys=True)),
                  "steps": args.steps, "seed": args.seed}
    checkpoint["memory_site"] = args.memory_site
    torch.save(checkpoint, Path(args.out) / "encoder.pt")
    save_json(Path(args.out) / "training-summary.json", {
        "model": args.model, "model_commit": checkpoint["model_commit"], "encoder": encoder.specification(),
        "encoder_parameters": sum(p.numel() for p in encoder.parameters()), "steps": args.steps,
        "first_loss": log[0]["loss"], "last_loss": log[-1]["loss"], "elapsed_seconds": time.monotonic() - start,
        "recipient_unchanged": before_hash == after_hash, "recipient_sha256": after_hash,
        "train_documents": len(fit_docs), "train_questions": len(train_q), "training_domain": "synthetic",
        "torch": torch.__version__, "transformers": transformers.__version__, "device": str(embedding.device),
        "memory_site": args.memory_site})


def evaluate(model, tokenizer, documents, questions, args):
    device = model.get_input_embeddings().weight.device
    checkpoint = torch.load(Path(args.out) / "encoder.pt", map_location="cpu", weights_only=True)
    if checkpoint["model"] != args.model or checkpoint["feature_layer"] != args.feature_layer:
        raise ValueError("Checkpoint model/feature layer differs from this evaluation")
    if checkpoint["config"]["max_tokens"] != args.max_doc_tokens:
        raise ValueError("Checkpoint document token cap differs from this evaluation")
    if checkpoint["recipient_sha256"] != model._smoke_fingerprint:
        raise ValueError("Recipient weights differ from the trained interface's recipient")
    encoder = MemoryEncoder(EncoderConfig(**checkpoint["config"])).to(device)
    encoder.load_state_dict(checkpoint["state_dict"])
    encoder.eval()
    by_id = {doc["id"]: doc for doc in documents}
    selected = [q for q in questions if q["split"] == args.eval_split]
    if args.eval_per_domain:
        kept = []
        for domain in sorted({q["domain"] for q in selected}):
            # Deterministically shuffle documents, then take one question per document
            # before taking second questions. This avoids a tiny prefix of one family.
            groups = defaultdict(list)
            for q in selected:
                if q["domain"] == domain:
                    groups[q["doc_id"]].append(q)
            keys = sorted(groups)
            random.Random(args.seed).shuffle(keys)
            interleaved = [groups[key][i] for i in range(max(map(len, groups.values())))
                           for key in keys if i < len(groups[key])]
            kept += interleaved[:args.eval_per_domain]
        selected = kept
    if not selected:
        raise ValueError("No evaluation questions")
    train_ids = set(checkpoint["train_doc_ids"])
    for q in selected:
        if q["doc_id"] in train_ids or digest(by_id[q["doc_id"]]["text"]) in set(checkpoint["train_text_sha256"].values()):
            raise ValueError("Evaluation fact overlaps encoder training")
    # Actual retrieval uses all documents (including distractors), not just probe records.
    retriever = BM25(documents)
    retrieval = {q["id"]: retriever.retrieve(q["question"])[0] for q in selected}
    wrong = {}
    for q in selected:
        candidates = sorted(d["id"] for d in documents if d["id"] != q["doc_id"] and d["domain"] == q["domain"])
        if q["domain"] == "synthetic":
            candidates = [key for key in candidates if by_id[key]["values"][q["property"]] != by_id[q["doc_id"]]["values"][q["property"]]]
        if not candidates:
            raise ValueError("Wrong-memory control needs a distinct document")
        wrong[q["id"]] = candidates[int(digest(q["id"])[:8], 16) % len(candidates)]
    # A small gate can encode only the records its controls need. The encoder is
    # question-independent in either mode. A complete offline bank uses "all".
    needed = (set(by_id) if args.bank_scope == "all" else
              set(q["doc_id"] for q in selected) | set(retrieval.values()) | set(wrong.values()))
    features = document_features(model, tokenizer, [by_id[key] for key in sorted(needed)], args)
    bank = {}
    # Encoding of held-out documents occurs with the learned interface frozen.
    for key in sorted(needed):
        x, mask = feature_batch(features, [key], device)
        with torch.no_grad():
            bank[key] = encoder(x, mask)[0].to(dtype=torch.float16, device="cpu")
    torch.save({"memories": bank, "model": args.model,
                "encoder_sha256": hashlib.sha256((Path(args.out) / "encoder.pt").read_bytes()).hexdigest(),
                "text_sha256": {key: digest(by_id[key]["text"]) for key in bank}}, Path(args.out) / "memory-bank.pt")
    # Validate source support after truncation: never score a claim the encoder couldn't see.
    support = {}
    for q in selected:
        value = features[q["doc_id"]]
        visible = tokenizer.decode(value["tokens"], skip_special_tokens=True)
        span = q.get("evidence_span", q["answers"][0])
        support[q["id"]] = span in visible
        if not support[q["id"]]:
            raise ValueError(f"Evidence for {q['id']} was truncated or altered: {span}")
    conditions = args.conditions.split(",")
    allowed = {"none", "text", "oracle", "retrieved", "wrong"}
    if not set(conditions) <= allowed:
        raise ValueError(f"Unknown condition; choose {sorted(allowed)}")
    rows = []
    result_path = Path(args.out) / f"predictions-{args.eval_split}.jsonl"
    for i, q in enumerate(selected):
        for condition in conditions:
            memory, context, used_id = None, None, None
            if condition == "text":
                # Exactly the same truncated source seen by the encoder.
                context = tokenizer.decode(features[q["doc_id"]]["tokens"], skip_special_tokens=True)
                used_id = q["doc_id"]
            elif condition in {"oracle", "retrieved", "wrong"}:
                used_id = {"oracle": q["doc_id"], "retrieved": retrieval[q["id"]], "wrong": wrong[q["id"]]}[condition]
                memory = bank[used_id].to(device)
            start = time.monotonic()
            prediction = greedy_answer(model, tokenizer, q, memory=memory, context=context,
                                       max_new_tokens=args.max_new_tokens,
                                       memory_site=checkpoint.get("memory_site", "prefix"))
            rows.append({"id": q["id"], "doc_id": q["doc_id"], "domain": q["domain"], "condition": condition,
                         "question": q["question"], "answers": q["answers"], "prediction": prediction,
                         "exact_match": exact_match(prediction, q["answers"]),
                         "numeric_match": numeric_match(prediction, q["answers"]),
                         "memory_doc_id": used_id, "retrieval_hit": retrieval[q["id"]] == q["doc_id"],
                         "record_sha256": digest(by_id[q["doc_id"]]["text"]),
                         "source": by_id[q["doc_id"]].get("source"),
                         "source_sha256": by_id[q["doc_id"]].get("source_sha256"),
                         "elapsed_seconds": time.monotonic() - start})
        write_jsonl(result_path, rows)
        print(f"evaluated {i + 1}/{len(selected)}: " + json.dumps({row["condition"]: row["prediction"] for row in rows[-len(conditions):]}), flush=True)
    summary = summarize(rows)
    summary.update({"model": args.model, "split": args.eval_split, "steps": checkpoint["steps"],
                    "slots": checkpoint["config"]["slots"], "bank_documents_encoded": len(bank),
                    "memory_site": checkpoint.get("memory_site", "prefix"),
                    "retrieval_index_documents": len(documents), "heldout_facts": True,
                    "training_domain": "synthetic", "source_support_after_truncation": all(support.values()),
                    "memory_bank_bytes": sum(value.numel() * value.element_size() for value in bank.values()),
                    "limitations": ["Small evaluation; no multi-hop claims", "Untrained BM25 text index",
                                    "Frozen recipient features are generated offline before compression",
                                    "All XDNA questions are numeric span probes; not a domain reasoning benchmark"]})
    save_json(Path(args.out) / f"summary-{args.eval_split}.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["condition"])].append(row)
    metrics = {}
    for (domain, condition), group in sorted(grouped.items()):
        metrics.setdefault(domain, {})[condition] = {
            "n": len(group), "exact_match": sum(r["exact_match"] for r in group) / len(group),
            "numeric_match": sum(r["numeric_match"] for r in group) / len(group),
            "mean_seconds": sum(r["elapsed_seconds"] for r in group) / len(group)}
    hits = {row["id"]: row["retrieval_hit"] for row in rows}
    # The paired recovery metric measures facts the plain model missed but text exposed.
    by_question = defaultdict(dict)
    for row in rows:
        by_question[(row["domain"], row["id"])][row["condition"]] = row
    recovery = {}
    retrieval_by_domain = {}
    for domain in metrics:
        domain_cases = [group for (d, _), group in by_question.items() if d == domain]
        retrieval_by_domain[domain] = {
            "hits": sum(next(iter(group.values()))["retrieval_hit"] for group in domain_cases),
            "n": len(domain_cases)}
        eligible = [group for (d, _), group in by_question.items() if d == domain and
                    "none" in group and "text" in group and not group["none"]["numeric_match"] and group["text"]["numeric_match"]]
        recovery[domain] = {"eligible": len(eligible)}
        for condition in ["oracle", "retrieved"]:
            measured = [group for group in eligible if condition in group]
            recovery[domain][condition] = (sum(g[condition]["numeric_match"] for g in measured) / len(measured)) if measured else None
    return {"metrics": metrics, "paired_recovery": recovery,
            "retrieval_recall_at_1": sum(hits.values()) / len(hits),
            "retrieval_definition": "Exact source-document ID; another finding may also contain valid evidence",
            "retrieval_by_domain": retrieval_by_domain}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "evaluate", "all"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "llama-1b")
    parser.add_argument("--work", type=Path, default=ROOT / "work")
    parser.add_argument("--cache-dir", type=Path, default=ROOT.parent / "models" / "encoded-memory-hf")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--max-doc-tokens", type=int, default=256)
    parser.add_argument("--feature-layer", type=int, default=-1)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-split", choices=["dev", "test"], default="dev")
    parser.add_argument("--eval-per-domain", type=int, default=0, help="0 evaluates every question")
    parser.add_argument("--conditions", default="none,text,oracle,retrieved,wrong")
    parser.add_argument("--bank-scope", choices=["needed", "all"], default="needed",
                        help="Small gate: needed records only; full offline bank: all records")
    parser.add_argument("--memory-site", choices=["prefix", "reference"], default="reference",
                        help="Training insertion site; evaluation uses the checkpoint's stored site")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.encode_batch_size < 1:
        parser.error("steps and batch sizes must be positive")
    if args.out.exists() and (args.out / "encoder.pt").exists() and args.stage in {"fit", "all"}:
        parser.error("Output already contains an encoder; choose a new --out to preserve the earlier experiment")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    documents = read_jsonl(args.data / "documents.jsonl")
    questions = read_jsonl(args.data / "questions.jsonl")
    if len({doc["id"] for doc in documents}) != len(documents):
        raise ValueError("Duplicate document IDs")
    argument_name = f"arguments-{args.stage}" + (f"-{args.eval_split}" if args.stage == "evaluate" else "")
    save_json(args.out / f"{argument_name}.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    model, tokenizer = load_recipient(args)
    print(f"recipient={args.model}; parameters={sum(p.numel() for p in model.parameters())}; frozen=True", flush=True)
    if args.stage in {"fit", "all"}:
        train_encoder(model, tokenizer, documents, questions, args)
    if args.stage in {"evaluate", "all"}:
        evaluate(model, tokenizer, documents, questions, args)


if __name__ == "__main__":
    main()
