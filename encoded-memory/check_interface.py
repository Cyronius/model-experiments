"""Can exact source embeddings be read at the SAME site as compressed memories?"""
import argparse
from pathlib import Path

import torch

from data import ROOT, exact_match, numeric_match, read_jsonl, write_jsonl
from memory import greedy_answer
from run import load_recipient, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--site", choices=["prefix", "reference"], default="prefix")
    args = parser.parse_args()
    checkpoint = torch.load(args.run / "encoder.pt", map_location="cpu", weights_only=True)
    args.model = checkpoint["model"]
    args.device, args.dtype = None, "float32"
    args.cache_dir = ROOT.parent / "models" / "encoded-memory-hf"
    args.local_files_only = True
    torch.set_num_threads(args.threads)
    model, tokenizer = load_recipient(args)
    if model._smoke_fingerprint != checkpoint["recipient_sha256"]:
        raise ValueError("Control must use the exact frozen recipient")
    by_id = {doc["id"]: doc for doc in read_jsonl(args.data / "documents.jsonl")}
    prior = read_jsonl(args.run / f"predictions-{args.split}.jsonl")
    tested_ids = {row["id"] for row in prior}
    questions = [q for q in read_jsonl(args.data / "questions.jsonl") if q["id"] in tested_ids]
    rows = []
    for i, q in enumerate(questions):
        ids = tokenizer(by_id[q["doc_id"]]["text"], return_tensors="pt", truncation=True,
                        max_length=checkpoint["config"]["max_tokens"]).input_ids.to(model.device)
        with torch.no_grad():
            memory = model.get_input_embeddings()(ids)[0]
        # These are exact original token embeddings, not contextual hidden states
        # and not learned compressed memories. The insertion point is identical.
        prediction = greedy_answer(model, tokenizer, q, memory=memory, max_new_tokens=12, memory_site=args.site)
        rows.append({"id": q["id"], "doc_id": q["doc_id"], "question": q["question"],
                     "answers": q["answers"], "prediction": prediction,
                     "exact_match": exact_match(prediction, q["answers"]),
                     "numeric_match": numeric_match(prediction, q["answers"]),
                     "source_tokens": ids.shape[1]})
        suffix = "" if args.site == "prefix" else "-reference"
        write_jsonl(args.run / f"interface{suffix}-control.jsonl", rows)
        print(f"interface {i + 1}/{len(questions)}: {prediction!r}", flush=True)
    summary = {"n": len(rows), "numeric_correct": sum(r["numeric_match"] for r in rows),
               "exact_correct": sum(r["exact_match"] for r in rows), "split": args.split,
               "model": args.model, "recipient_sha256": model._smoke_fingerprint,
               "site": args.site, "control": "Exact source token embeddings; no compression"}
    save_json(args.run / f"interface{suffix}-summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
