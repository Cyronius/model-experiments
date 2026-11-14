"""Freeze complete calibration conversations and held-out scale-probe tasks.

No model outputs are used to select tasks or thresholds. Token counts are
recorded independently for every tokenizer; all models see the same messages.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from healing_data import REVISION, chat_ids, encode_complete, eval_cases, pair


def load_tokenizer(path):
    config = json.loads((Path(path) / "config.json").read_text())
    options = {"fix_mistral_regex": True} if config["model_type"] == "mistral" else {}
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, **options)
    # Mistral's default system message contains today's date. Freeze its normal
    # template, rather than changing it between preparation and the GPU run.
    tok.chat_template = tok.chat_template.replace('strftime_now("%Y-%m-%d")', '"2026-10-24"')
    return tok


def encoding_digest(data, tok):
    encoded = [[chat_ids(tok, r["messages"]) for r in data[split]] for split in ("calibration", "validation")]
    encoded.append([chat_ids(tok, c["messages"], True) for c in data["generation_eval"]])
    return hashlib.sha256(json.dumps(encoded, separators=(",", ":")).encode()).hexdigest()


def generation_cases():
    cases = eval_cases()
    for i in range(8):
        a, b = 31 + 6 * i, 13 + 2 * i
        cases.append({"id": f"arithmetic-{i}", "messages": [{"role": "user", "content":
                      f"What is {a} multiplied by {b}? Output only the integer answer."}],
                      "check": {"kind": "exact", "expected": str(a * b)}})
        code = f"R{i+3}k-Z{19+i}p"
        cases.append({"id": f"literal-{i}", "messages": [{"role": "user", "content":
                      f"Copy this identifier exactly, with no explanation: {code}"}],
                      "check": {"kind": "exact", "expected": code}})
        cases.append({"id": f"json-{i}", "messages": [{"role": "user", "content":
                      f'Return only JSON with keys "label" and "count". Set label to "item{i}" and count to {i+7}.'}],
                      "check": {"kind": "json", "expected": {"label": f"item{i}", "count": i + 7}}})
        values = [17 + i, -11 - i, 6 - i, 33 + 2 * i]
        cases.append({"id": f"ordering-{i}", "messages": [{"role": "user", "content":
                      f"Sort {', '.join(map(str, values))} in increasing order. Output only a comma-separated list."}],
                      "check": {"kind": "numbers", "expected": sorted(values)}})
        cases.append({"id": f"memory-{i}", "messages": pair(
                      f"Remember that my locker identifier is {code}.", f"Your locker identifier is {code}.") +
                      [{"role": "user", "content": "What is my locker identifier? Output only that identifier."}],
                      "check": {"kind": "exact", "expected": code}})
        values = [i + 2, i + 5, i + 9]
        cases.append({"id": f"python-{i}", "messages": [{"role": "user", "content":
                      f"What does this Python code print? xs = {values}; print(sum(xs[1:]))\nOutput only the number."}],
                      "check": {"kind": "exact", "expected": str(sum(values[1:]))}})
    prompts = [
        "Explain how a refrigerator moves heat. Include the compressor, condenser, expansion valve, and evaporator. Write 250 to 350 words.",
        "Write a Python function that parses a CSV string and returns the sum of a named numeric column. Explain error handling and give three tests.",
        "Plan a small community vegetable garden for beginners. Explain site selection, planting, watering, maintenance, and a realistic first-season schedule in 250 to 350 words.",
        "Explain the difference between correlation and causation using three concrete examples. Write 250 to 350 words and finish with a brief conclusion.",
        "Solve this problem step by step: a tank initially holds 120 liters; an inlet adds 8 liters per minute and an outlet removes 5 liters per minute. After 10 minutes the inlet stops. When will the tank be empty? Explain how to check the answer.",
        "Write a short complete story about a lost key, a bicycle mechanic, and an unexpected visitor. Give it a clear ending. Use 250 to 350 words.",
        "Explain binary search to a beginner, provide a Python implementation, and walk through a small example. Discuss empty input and a missing target.",
        "Compare solar and wind power for a small remote community. Explain reliability, maintenance, storage, and how to choose. Use 250 to 350 words.",
    ]
    for i, prompt in enumerate(prompts):
        cases.append({"id": f"long-{i}", "messages": [{"role": "user", "content": prompt}],
                      "check": None, "max_new_tokens": 1024})
    for c in cases:
        c.setdefault("max_new_tokens", 512)
    return cases


def prepare(source, tokenizers, out, calibration_tokens=65536, validation_rows=32, max_length=768):
    cases = generation_cases()
    seen = {c["messages"][0]["content"].strip().casefold() for c in cases}
    rows = {}
    totals = {name: 0 for name in tokenizers}
    sources = {}
    for split in ("train_sft", "test_sft"):
        file = next((source / "data").glob(f"{split}-00000-*.parquet"))
        sources[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        table = pq.read_table(file, columns=["prompt_id", "messages"])
        order = list(range(len(table)))
        random.Random(24027).shuffle(order)
        selected = []
        for idx in order:
            raw = table.slice(idx, 1).to_pylist()[0]
            m = raw["messages"]
            if len(m) < 2 or m[0]["role"] != "user":
                continue
            prompt = m[0]["content"].strip().casefold()
            if prompt in seen:
                continue
            candidates = [m[:4], m[:2]] if len(m) >= 4 else [m[:2]]
            for messages in candidates:
                if [x["role"] for x in messages] != ["user", "assistant"] * (len(messages) // 2):
                    continue
                if any(len(x["content"]) > 10000 for x in messages):
                    continue
                encoded = {name: encode_complete(tok, messages, max_length) for name, tok in tokenizers.items()}
                if any(x is None for x in encoded.values()):
                    continue
                selected.append({"id": f"{split}-{raw['prompt_id']}", "messages": messages})
                seen.add(prompt)
                if split == "train_sft":
                    for name, row in encoded.items():
                        totals[name] += len(row["input_ids"])
                break
            if ((split == "train_sft" and min(totals.values()) >= calibration_tokens)
                    or (split == "test_sft" and len(selected) == validation_rows)):
                break
        if not selected or (split == "test_sft" and len(selected) != validation_rows):
            raise ValueError(f"Insufficient complete data in {split}")
        rows["calibration" if split == "train_sft" else "validation"] = selected
    if min(totals.values()) < calibration_tokens:
        raise ValueError("Insufficient calibration tokens")
    counts = {}
    for name, tok in tokenizers.items():
        counts[name] = {}
        for split, messages in rows.items():
            encoded = [encode_complete(tok, row["messages"], max_length) for row in messages]
            counts[name][split] = {"conversations": len(encoded),
                "input_tokens": sum(len(r["input_ids"]) for r in encoded),
                "assistant_tokens": sum(len(r["targets"]) for r in encoded)}
        # Verify all prompts can be rendered before renting a GPU.
        counts[name]["generation_prompt_tokens"] = sum(len(chat_ids(tok, c["messages"], True)) for c in cases)
    result = {"dataset": "HuggingFaceH4/ultrachat_200k", "revision": REVISION,
              "source_sha256": sources, "seed": 24027, "max_length": max_length,
              "minimum_calibration_tokens": calibration_tokens, **rows,
              "generation_eval": cases, "counts": counts,
              "limits": "Screening tasks, not a benchmark or proof of parity. Eight long-form answers require human review."}
    result["tokenizer_policy"] = {"fix_mistral_regex": True, "template_date": "2026-10-24",
                                   "thinking": False, "default_system_messages": "preserved"}
    result["encoded_sha256"] = {name: encoding_digest(result, tok) for name, tok in tokenizers.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path("target/manifold-audit/ultrachat"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tokenizers = {name: load_tokenizer(path) for name, path in {
        "qwen27": "target/manifold-scale/qwen27", "mistral24": "target/manifold-scale/mistral24",
        "qwen2": "models/manifold-audit/Qwen3.5-2B"}.items()}
    prepare(args.source, tokenizers, args.out)
