#!/usr/bin/env python3
"""Local Qwen 4B recall pilot: documents -> saved latents -> attention -> answers."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import re
import subprocess
import time
from pathlib import Path

from memory import CODE_REVISION, MODEL_REVISION, sha256

FILES = [
    "findings/F-0006-program-memory-16kb.md",
    "findings/F-0009-worker-stack-size.md",
    "findings/F-0014-aie-api-pitfalls.md",
    "findings/F-0038-elf-vs-classic-load.md",
    "findings/F-0066-masked-softmax-lowest.md",
    "findings/F-0101-ctx-switch-decode-sized.md",
]
CANARIES = {"original": ["velvet-heron-7316", "mosaic-lantern-4829"],
            "counterfactual": ["copper-badger-9527", "amber-compass-1638"]}


def prepare(args):
    root, out = args.notes.resolve(), args.data
    out.mkdir(parents=True, exist_ok=True)
    corpus, sources = [], []
    snapshot = out / "sources"
    snapshot.mkdir(exist_ok=True)
    for name in FILES:
        data = (root / name).read_bytes()
        (snapshot / Path(name).name).write_bytes(data)
        sources.append({"path": name, "sha256": sha256(data)})
        corpus.append(f"\n===== {name} =====\n{data.decode().strip()}\n")
    combined = "\n".join(corpus)
    for name, values in CANARIES.items():
        addendum = (
            "\n# Synthetic recall controls (not hardware findings)\n"
            f"The code for the memory recall pilot is {values[0]}.\n"
            f"The recovery phrase for the pilot memory bank is {values[1]}.\n"
            "These arbitrary strings are test fixtures, not XDNA2 recommendations.\n"
        )
        (out / f"{name}.txt").write_text(combined + addendum)
    rev = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    (out / "sources.json").write_text(json.dumps({"notes_commit": rev, "sources": sources}, indent=2) + "\n")
    cases = json.loads((Path(__file__).parent / "cases.json").read_text())
    # Separate query-only input makes answer leakage easy to audit.
    questions = [{k: c[k] for k in ("id", "question", "category", "source")} for c in cases]
    (out / "questions.json").write_text(json.dumps(questions, indent=2) + "\n")
    (out / "gold.json").write_text(json.dumps(cases, indent=2) + "\n")
    print("PREPARED", len(FILES), "notes;", len(cases), "questions", flush=True)


def encode(args):
    from memory import encode_banks, setup
    setup(args.threads)
    actual = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != CODE_REVISION:
        raise ValueError(f"Expected pinned LCLM code {CODE_REVISION}, got {actual}")
    banks = {name: (args.data / f"{name}.txt").read_text() for name in CANARIES}
    encode_banks(args.checkpoint, args.upstream, banks, args.memories)


def evaluate(args):
    import torch
    import transformers
    from safetensors.torch import load_file
    from memory import answer, load_decoder, make_prefix, prefill, setup
    setup(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    questions = json.loads((args.data / "questions.json").read_text())
    if args.limit:
        questions = questions[:args.limit]
    manifest = json.loads((args.memories / "manifest.json").read_text())
    if manifest["model_revision"] != MODEL_REVISION:
        raise ValueError("Memory/decoder revision mismatch")
    (args.output / "environment.json").write_text(json.dumps({
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "device": "cpu", "dtype": "bfloat16",
        "threads": args.threads, "model_revision": MODEL_REVISION,
        "code_revision": CODE_REVISION, "max_new_tokens": args.max_tokens,
        "greedy": True, "training_steps": 0, "paid_compute_cost_usd": 0,
    }, indent=2) + "\n")
    model, tokenizer = load_decoder(args.checkpoint)
    stats = {}
    with (args.output / "answers.jsonl").open("w") as stream:
        for condition in args.conditions:
            selected = [q for q in questions if condition != "counterfactual" or q["category"] == "canary"]
            if not selected:
                continue
            memory, document = None, None
            if condition == "full_text":
                document = (args.data / "original.txt").read_text()
            elif condition in ("latent", "zero_memory", "counterfactual"):
                bank = "counterfactual" if condition == "counterfactual" else "original"
                path = args.memories / f"{bank}.safetensors"
                if sha256(path.read_bytes()) != manifest["banks"][bank]["file_sha256"]:
                    raise ValueError("Memory file failed its checksum")
                memory = load_file(path)["memory"]
                if condition == "zero_memory":
                    memory = torch.zeros_like(memory)
            start = time.perf_counter()
            with torch.inference_mode():
                prefix = make_prefix(model, tokenizer, memory=memory, document=document)
                cache = prefill(model, prefix)
            stats[condition] = {"prefix_positions": prefix.shape[1], "prefill_seconds": time.perf_counter() - start}
            print("PREFIX", condition, stats[condition], flush=True)
            for q in selected:
                result = answer(model, tokenizer, cache, q["question"], args.max_tokens)
                row = dict(q, condition=condition, **result)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(condition, q["id"], repr(row["answer"]), f'{row["seconds"]:.2f}s', flush=True)
            del prefix, cache
    (args.output / "prefixes.json").write_text(json.dumps(stats, indent=2) + "\n")


def score_answer(case, condition, answer):
    patterns = case["counterfactual_patterns"] if condition == "counterfactual" else case["patterns"]
    text = answer.lower().replace("−", "-").replace("–", "-").replace("`", "")
    hits = [bool(re.search(p, text)) for p in patterns]
    forbidden = any(re.search(p, text) for p in case.get("forbidden", []))
    return all(hits) and not forbidden, hits


def summarize(args):
    gold = {c["id"]: c for c in json.loads((args.data / "gold.json").read_text())}
    rows = [json.loads(line) for line in (args.output / "answers.jsonl").read_text().splitlines()]
    for extra in args.extra_answers:
        rows.extend(json.loads(line) for line in extra.read_text().splitlines())
    keys = [(row["condition"], row["id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate condition/question trials")
    scores, detail = {}, []
    review = json.loads(args.review.read_text()) if args.review else {"overrides": []}
    overrides = {(o["condition"], o["id"]): o for o in review["overrides"]}
    for row in rows:
        passed, hits = score_answer(gold[row["id"]], row["condition"], row["answer"])
        automatic = passed
        override = overrides.get((row["condition"], row["id"]))
        if override:
            passed = override["correct"]
        row.update(correct=passed, automatic_correct=automatic, review=override,
                   pattern_hits=hits, gold=gold[row["id"]])
        detail.append(row)
        groups = scores.setdefault(row["condition"], {})
        for group in (row["category"], "all"):
            s = groups.setdefault(group, {"correct": 0, "total": 0})
            s["correct"] += int(passed)
            s["total"] += 1
    result = {"scores": scores, "manual_review": review, "answers": detail}
    (args.output / "scored.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Qwen 4B XDNA2 encoded-memory recall pilot", "", "Local CPU inference, no training, $0 paid compute.", "",
             "| Condition | Real note facts | Synthetic canaries | Absent facts (abstention) |",
             "|---|---:|---:|---:|"]
    for condition, groups in scores.items():
        counts = [f'{groups[g]["correct"]}/{groups[g]["total"]}' if g in groups else "—" for g in ("fact", "canary", "absent")]
        lines.append(f"| {condition} | " + " | ".join(counts) + " |")
    manifest = json.loads((args.memories / "manifest.json").read_text())
    m = manifest["banks"]["original"]
    lines += ["", f'Encoded {m["source_tokens"]:,} source tokens into {m["memory_slots"]:,} continuous memory slots '
              f'({m["memory_bytes"]:,} bytes), once before questions; decoder width {m["hidden_size"]}.', "",
              "The latent and zero-memory conditions read only the saved memory file and question text. "
              "Original notes are read by the encoder and the full-text control. "
              "The first five conditions use the same released LCLM Qwen3-4B decoder, which was trained by the authors. "
              "Stock controls, when included, use the original Qwen3-4B-Instruct-2507 weights. "
              "This is not a frozen-stock-Qwen surgery experiment. No query-time retrieval or tool access is enabled.", "",
              "The LCLM full-text control ended immediately in prompt preflight, with and without document markers. "
              "Stock controls were added afterward to check that the corpus and questions are answerable. "
              "A comparison across stock and LCLM weights does not isolate compression cost. "
              "The no-memory, zero-memory, and changed-bank controls establish whether encoded content is being used.", "",
              "Scores use preregistered answer patterns. Check the raw answers below: matching is a small-pilot diagnostic, "
              "not a broad benchmark. The six notes are hand-selected, and some contain disputed or stack-specific findings. "
              "Questions ask what those notes report, not universal hardware truth. Canary strings are synthetic controls.", "",
              "When review.json is supplied, scores include explicit human-readable adjudications. "
              "The raw regex result remains in scored.json. The latent uint8-view response is rejected for contradicting itself; "
              "the firmware response correctly says the version is unspecified despite missing the regex wording. "
              "The latter response and the contradictory response also hit the output cap.", "",
              "## Raw answers", "", "| Condition | Question | Answer | Pass |", "|---|---|---|---|"]
    for row in detail:
        ans = row["answer"].replace("|", "\\|").replace("\n", " ")
        lines.append(f'| {row["condition"]} | {row["id"]} | {ans} | {"yes" if row["correct"] else "no"} |')
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(scores, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "encode", "evaluate", "summarize"])
    parser.add_argument("--notes", type=Path, default=Path("../xdna2-notes"))
    parser.add_argument("--checkpoint", type=Path, required=False)
    parser.add_argument("--upstream", type=Path, required=False)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--memories", type=Path, default=Path("memory"))
    parser.add_argument("--output", type=Path, default=Path("results/pilot"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--extra-answers", nargs="*", type=Path, default=[])
    parser.add_argument("--review", type=Path)
    parser.add_argument("--conditions", nargs="+", default=["no_memory", "full_text", "latent", "zero_memory", "counterfactual"],
                        choices=["no_memory", "full_text", "latent", "zero_memory", "counterfactual"])
    args = parser.parse_args()
    globals()[args.action](args)


if __name__ == "__main__":
    main()
