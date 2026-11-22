"""Teacher-export curated append-form groups: base rows plus only the changed rows of each enriched prompt (2026-11-11)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import harness

TAIL = 9  # tokens after the user text: <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_groups(paths: list[Path], min_details: int, models: set[str] | None) -> list[dict]:
    groups, seen = [], set()
    for path in paths:
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if not r.get("keep") or r["id"] in seen or (models and r.get("model") not in models):
                continue
            details = list(dict.fromkeys(d for d in r["details"] + r["extra"] if d))
            if len(details) >= min_details and r["base"]:
                seen.add(r["id"])
                groups.append({"id": r["id"], "model": r.get("model"), "base": r["base"], "details": details})
    return groups


def done_ids(out: Path) -> set[str]:
    ids = set()
    for index in out.glob("shard-*.json"):
        ids.update(g["id"] for g in json.loads(index.read_text(encoding="utf-8"))["groups"])
    return ids


def encode(model, tok, weight, texts: list[str], device: str) -> tuple[list[list[int]], torch.Tensor, list[int]]:
    enc = tok([harness.chat_text(t) for t in texts], return_tensors="pt", padding=True, add_special_tokens=False)
    with torch.inference_mode():
        hs = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True, use_cache=False).hidden_states
        cond = F.linear(torch.cat([hs[k] for k in harness.TAPS], -1), weight)
    lens = enc["attention_mask"].sum(1).tolist()
    return [enc["input_ids"][i, :n].tolist() for i, n in enumerate(lens)], cond.cpu(), lens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Teacher snapshot (diffusers layout)")
    parser.add_argument("--curated", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--shard-groups", type=int, default=200)
    parser.add_argument("--min-details", type=int, default=4)
    parser.add_argument("--models", nargs="*", help="Only use curated rows from these curation models")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    from transformers import AutoTokenizer, Qwen3Model

    groups = load_groups(args.curated, args.min_details, set(args.models) if args.models else None)
    args.output.mkdir(parents=True, exist_ok=True)
    skip = done_ids(args.output)
    groups = [g for g in groups if g["id"] not in skip]
    if args.limit:
        groups = groups[: args.limit]
    print(f"{len(groups)} groups to export, {len(skip)} already done", flush=True)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model / "tokenizer", local_files_only=True)
    model = Qwen3Model.from_pretrained(args.model / "text_encoder", local_files_only=True, torch_dtype=dtype,
                                       attn_implementation="eager").to(args.device).eval()
    weights, _ = harness.projection_files(args.model)
    weight = weights["context_embedder.weight"].to(args.device, dtype)
    provenance = {"script_sha256": sha256(Path(__file__)), "harness_sha256": sha256(Path(harness.__file__)),
                  "teacher": harness.TEACHER, "revision": harness.REVISION, "model_path": str(args.model),
                  "upstream_verification_sha256": sha256(args.model / "verified-upstream.json")
                  if (args.model / "verified-upstream.json").exists() else None,
                  "dtype": args.dtype, "device": args.device, "threads": args.threads,
                  "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "safetensors")},
                  "curated_sha256": {str(p): sha256(p) for p in args.curated}}
    shard = len(list(args.output.glob("shard-*.json")))
    t0, n_prompts, skipped_details = time.time(), 0, 0
    for s in range(0, len(groups), args.shard_groups):
        chunk = groups[s:s + args.shard_groups]
        texts = [t for g in chunk for t in [g["base"]] + [f"{g['base']}, {d}" for d in g["details"]]]
        # Sort by length so batches pad little; keep the original order through an index.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        ids_out, rows_out = [None] * len(texts), [None] * len(texts)
        for b in range(0, len(order), args.batch):
            idx = order[b:b + args.batch]
            ids, cond, lens = encode(model, tok, weight, [texts[i] for i in idx], args.device)
            for j, i in enumerate(idx):
                ids_out[i], rows_out[i] = ids[j], cond[j, :lens[j]].to(torch.bfloat16)
        n_prompts += len(texts)
        tensors, index, offset, pos = [], [], 0, 0
        prefix_mismatch = 0.0
        for g in chunk:
            base_ids, base_rows = ids_out[pos], rows_out[pos]
            k = len(base_ids) - TAIL
            entry = {"id": g["id"], "curation_model": g["model"], "base": g["base"],
                     "base_offset": offset, "base_rows": len(base_rows), "details": []}
            tensors.append(base_rows)
            offset += len(base_rows)
            for d_i, detail in enumerate(g["details"]):
                e_ids, e_rows = ids_out[pos + 1 + d_i], rows_out[pos + 1 + d_i]
                # Append form needs the base text tokens as an exact prefix of the enriched tokens.
                if e_ids[:k] != base_ids[:k] or len(e_ids) <= len(base_ids):
                    skipped_details += 1
                    continue
                prefix_mismatch = max(prefix_mismatch, float((e_rows[:k].float() - base_rows[:k].float()).abs().max()))
                changed = e_rows[k:]
                entry["details"].append({"text": detail, "first_changed_row": k, "offset": offset, "rows": len(changed)})
                tensors.append(changed)
                offset += len(changed)
            index.append(entry)
            pos += 1 + len(g["details"])
        name = f"shard-{shard:05d}"
        save_file({"rows": torch.cat(tensors).contiguous()}, str(args.output / f"{name}.safetensors"),
                  metadata={"protocol": "klein-append-rows-v1"})
        meta = {"protocol": "klein-append-rows-v1", "provenance": provenance, "width": harness.WIDTH,
                "prefix_rows_max_abs_diff": prefix_mismatch, "skipped_details": skipped_details,
                "rows_sha256": sha256(args.output / f"{name}.safetensors"), "groups": index}
        tmp = args.output / f"{name}.json.tmp"
        tmp.write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(args.output / f"{name}.json")
        shard += 1
        dt = time.time() - t0
        print(f"{name}: {len(chunk)} groups, {n_prompts} prompts, {dt / n_prompts:.3f} s/prompt, "
              f"prefix diff {prefix_mismatch:g}, skipped details {skipped_details}", flush=True)


if __name__ == "__main__":
    main()
