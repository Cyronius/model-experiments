#!/usr/bin/env python3
"""Pinned, cached, local inference; gold annotations are never model input."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import time
from pathlib import Path

from curator import fingerprint

MODELS = {
    "nuextract2b": {"id": "numind/NuExtract-2.0-2B", "revision": "5eb4a99746edc2ac44900f56a5c9431d3c03b16d"},
    "qwen4b": {"id": "Qwen/Qwen3-4B-Instruct-2507", "revision": "cdbee75f17c01a7cc42f958dc650907174af0554"},
    "nuextract3": {"id": "numind/NuExtract3", "revision": "c99dc8f5641b866aa0192b6ea78f84bf9f3535f1"},
}


def tasks(data, prompts):
    manifest = json.loads((data / "manifest.json").read_text())
    for c in manifest["passages"]:
        yield {"id": c["id"], "task": "extraction", "text": c["text"],
               "source_sha256": manifest["sources"][c["source_key"]]["sha256"]}
    # Deliberately select only claim text; gold labels/reasons never enter prompts.
    for pair in json.loads((data / "pairs.json").read_text()):
        yield {"id": pair["id"], "task": "relation",
               "text": f'Claim A: {pair["claim_a"]}\nClaim B: {pair["claim_b"]}'}


def render(tokenizer, task, config, family):
    key = "extraction" if task["task"] == "extraction" else "relation"
    instruction = config[f"{key}_instruction"]
    template = json.dumps(config[f"{key}_template"], ensure_ascii=False)
    examples = config[f"{key}_examples"]
    if family == "nuextract2b":
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": instruction}, {"role": "user", "content": task["text"]}],
            template=template, examples=examples, tokenize=False, add_generation_prompt=True,
        )
    if family == "nuextract3":
        messages = [{"role": "developer", "content": [
            {"type": "text", "text": e["input"]},
            {"type": "text", "text": e["output"]},
        ]} for e in examples]
        messages.append({"role": "user", "content": task["text"]})
        return tokenizer.apply_chat_template(messages, template=template, instructions=instruction,
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
    content = "# Template:\n" + template + "\n# Examples:\n"
    for e in examples:
        content += "## Input:\n" + e["input"] + "\n## Output:\n" + e["output"] + "\n"
    content += "# Context:\n" + task["text"]
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": instruction}, {"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def run(args):
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2VLForConditionalGeneration, Qwen3_5ForConditionalGeneration
    torch.set_num_threads(args.threads)
    torch.manual_seed(2718)
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    prompt = json.loads(args.prompt.read_text())
    metadata_path = args.checkpoint / ".cache/huggingface/download/config.json.metadata"
    if not metadata_path.exists() or metadata_path.read_text().splitlines()[0] != MODELS[args.model]["revision"]:
        raise ValueError("Checkpoint download metadata does not match the pinned revision; use download.py")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = {"model": MODELS[args.model], "prompt_sha256": fingerprint(prompt),
              "max_new_tokens": args.max_tokens, "batch_size": args.batch_size,
              "threads": args.threads, "dtype": "bfloat16", "device": "cpu", "greedy": True,
              "torch": torch.__version__, "transformers": transformers.__version__}
    # This release's generation_config ends on 248044 while its tokenizer and
    # chat template end on 248046. Honor both native end markers explicitly.
    if args.model == "nuextract3":
        config["eos_token_ids"] = [248044, tokenizer.eos_token_id]
    jobs = []
    for task in tasks(args.data, prompt):
        text = render(tokenizer, task, prompt, args.model)
        key = fingerprint({"config": config, "task": task, "rendered_prompt": text})
        jobs.append({**task, "prompt": text, "cache_key": key})
    if args.limit:
        jobs = jobs[:args.limit]
    pending = [j for j in jobs if not (args.cache / f'{j["cache_key"]}.json').exists()]
    print(f"{args.model}: {len(jobs) - len(pending)} cached, {len(pending)} pending", flush=True)
    model, batches = None, []
    start_all = time.perf_counter()
    if pending:
        cls = {"nuextract2b": Qwen2VLForConditionalGeneration,
               "nuextract3": Qwen3_5ForConditionalGeneration,
               "qwen4b": AutoModelForCausalLM}[args.model]
        model = cls.from_pretrained(args.checkpoint, dtype=torch.bfloat16,
                                   attn_implementation="sdpa", local_files_only=True).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        # Do not mix extraction and relation tasks inside a generation batch.
        for kind in ("extraction", "relation"):
            subset = [j for j in pending if j["task"] == kind]
            for offset in range(0, len(subset), args.batch_size):
                batch = subset[offset:offset + args.batch_size]
                inputs = tokenizer([j["prompt"] for j in batch], padding=True, return_tensors="pt")
                start = time.perf_counter()
                with torch.inference_mode():
                    stop_kwargs = {"eos_token_id": config["eos_token_ids"]} if "eos_token_ids" in config else {}
                    output = model.generate(**inputs, do_sample=False, num_beams=1,
                                            max_new_tokens=args.max_tokens,
                                            pad_token_id=tokenizer.pad_token_id, **stop_kwargs)
                seconds = time.perf_counter() - start
                batch_meta = {"task": kind, "ids": [j["id"] for j in batch], "seconds": seconds,
                              "padded_input_tokens": inputs.input_ids.shape[1], "output_tokens": 0}
                eos = config.get("eos_token_ids", model.generation_config.eos_token_id)
                eos = set(eos if isinstance(eos, list) else [eos])
                for job, ids in zip(batch, output[:, inputs.input_ids.shape[1]:]):
                    ids = ids.tolist()
                    stop = next((n for n, token in enumerate(ids) if token in eos), len(ids))
                    generated = ids[:stop]
                    row = {k: v for k, v in job.items() if k not in ("prompt", "text")}
                    row.update(output=tokenizer.decode(generated, skip_special_tokens=True),
                               input_tokens=int(tokenizer(job["prompt"], return_length=True)["length"][0]),
                               output_tokens=len(generated), truncated=stop == len(ids) and len(ids) >= args.max_tokens,
                               rendered_prompt=job["prompt"], batch_ids=batch_meta["ids"], batch_seconds=seconds)
                    (args.cache / f'{job["cache_key"]}.json').write_text(json.dumps(row, indent=2) + "\n")
                    batch_meta["output_tokens"] += len(generated)
                batches.append(batch_meta)
                print(args.model, kind, batch_meta["ids"], f'{seconds:.1f}s', batch_meta["output_tokens"], "tokens", flush=True)
    rows = [json.loads((args.cache / f'{j["cache_key"]}.json').read_text()) for j in jobs]
    (args.out / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    metadata = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "python": platform.python_version(),
                "config": config, "jobs": len(jobs), "cache_hits": len(jobs) - len(pending),
                "new_generations": len(pending), "wall_seconds": time.perf_counter() - start_all,
                "batches": batches, "dataset_sha256": fingerprint(json.loads((args.data / "manifest.json").read_text())),
                "gold_sha256": fingerprint(json.loads((args.data / "gold.json").read_text())),
                "training_steps": 0, "paid_compute_cost_usd": 0}
    name = "execution.json" if pending else "cache-replay.json"
    (args.out / name).write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--prompt", type=Path, default=Path(__file__).with_name("prompt.json"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cache", type=Path, default=Path("cache"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--limit", type=int, default=0)
    run(p.parse_args())
