"""Paired local output-normalization ablation with a modest residual-width cut.

Keep all MLP neurons, heads and layers. Only the final norm differs between
legacy, dense/inverse-scale and corrected dense/multiply-scale arms. This is
an initialization diagnostic; no healing or quality-parity claim is made.
"""
import argparse
import copy
import gc
import hashlib
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from audit_rotation import collect_basis, compare, norm_gain, norm_epsilon, save, cache_check
from projection_models import (install_output_matrix, load_causal_model,
                               projected_output_matrix, project_residual_width)


def log(text):
    print(text, flush=True)


def data(tokenizer, max_tokens):
    root = Path(__file__).parent / "smoke"
    prompts = [json.loads(s) for s in (root / "prompts.jsonl").read_text().splitlines()]
    answers = {r["id"]: r["answer"] for r in map(json.loads, (root / "answers_stock.jsonl").read_text().splitlines())}
    rows = {}
    for p in prompts:
        text = tokenizer.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                              tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prefix = tokenizer(text, add_special_tokens=False).input_ids
        ans = tokenizer(answers[p["id"]], add_special_tokens=False).input_ids
        ids = (prefix + ans + [tokenizer.eos_token_id])[:max_tokens]
        positions = list(range(len(prefix) - 1, len(ids) - 1, 16))
        if not positions:
            raise ValueError("No assistant positions available")
        rows[p["id"]] = {"ids": torch.tensor([ids]), "positions": torch.tensor(positions), "prompt": p["prompt"]}
    fit_ids = list(range(0, 40, 2))
    eval_ids = [1, 7, 11, 17, 21, 27, 33, 39]
    return rows, fit_ids, eval_ids, hashlib.sha256((root / "answers_stock.jsonl").read_bytes()).hexdigest()


@torch.no_grad()
def rms_scales(model, rows, p):
    norms = [n for layer in model.model.layers for n in (layer.input_layernorm, layer.post_attention_layernorm)] + [model.model.norm]
    totals, counts, handles = [0.] * len(norms), [0] * len(norms), []
    for i, norm in enumerate(norms):
        eps = norm_epsilon(norm)
        def hook(module, args, i=i, eps=eps):
            x = args[0].reshape(-1, p.shape[0]).float()
            z = x @ p
            ratio = torch.sqrt((z.square().mean(-1) + eps) / (x.square().mean(-1) + eps))
            totals[i] += ratio.sum().item()
            counts[i] += len(x)
        handles.append(norm.register_forward_pre_hook(hook))
    try:
        for row in rows:
            model.model(input_ids=row, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return [s / n for s, n in zip(totals, counts, strict=True)]


@torch.no_grad()
def pre_final_states(model, rows):
    output = []
    for row in rows:
        def hook(module, args):
            output.append(args[0][:, row["positions"]].clone())
        h = model.model.norm.register_forward_pre_hook(hook)
        try:
            model.model(input_ids=row["ids"], use_cache=False)
        finally:
            h.remove()
    return output


@torch.no_grad()
def output_logits(model, hidden):
    return [model.lm_head(model.model.norm(h)).float() for h in hidden]


@torch.no_grad()
def generate(model, tokenizer, rows, max_new):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                          tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    batch = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
    start = time.perf_counter()
    out = model.generate(**batch, max_new_tokens=max_new, do_sample=False, use_cache=True,
                         pad_token_id=tokenizer.pad_token_id)
    result = []
    stop = tokenizer.eos_token_id
    for row, seq in zip(rows, out[:, batch.input_ids.shape[1]:], strict=True):
        ids = seq.tolist()
        if stop in ids:
            ids = ids[:ids.index(stop) + 1]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        words = re.findall(r"\w+", text.lower())
        grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
        result.append({"prompt": row["prompt"], "answer": text, "tokens": len(ids),
                       "ended": stop in ids, "repeated_4gram_fraction": 1 - len(set(grams)) / len(grams) if grams else 0.})
    return {"decoding": "greedy diagnostic, thinking off", "max_new_tokens": max_new,
            "elapsed_seconds": time.perf_counter() - start, "answers": result}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-model", type=Path, required=True)
    ap.add_argument("--width", type=int, default=1792)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    start = time.perf_counter()
    result = {"args": vars(args) | {"out": str(args.out), "save_model": str(args.save_model)},
              "arms": {}, "timings": {}, "device": "cpu", "dtype": "float32", "spend_usd": 0}
    manifest = Path(args.model) / "download_manifest.json"
    if manifest.exists():
        result["source_manifest"] = json.loads(manifest.read_text())

    def timed(name, function):
        t = time.perf_counter()
        value = function()
        result["timings"][name] = time.perf_counter() - t
        log(f"{name}: {result['timings'][name]:.2f}s")
        save(result, args.out)
        return value

    model = load_causal_model(args.model, dtype=torch.float32, attn_implementation="eager", local_files_only=True).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows, fit_ids, eval_ids, checksum = data(tokenizer, args.max_tokens)
    fit = [rows[i]["ids"] for i in fit_ids]
    ev = [rows[i] for i in eval_ids]
    gen = [rows[i] for i in [7, 11, 21, 33]]
    result["data"] = {"calibration_ids": fit_ids, "evaluation_ids": eval_ids, "generation_ids": [7, 11, 21, 33],
                      "calibration_tokens": sum(x.numel() for x in fit), "reference_answer_sha256": checksum,
                      "source": "Cached stock Qwen2B greedy answers, used as prefixes only; not correctness labels."}
    result["original_parameters"] = sum(p.numel() for p in model.parameters())
    result["source_width"] = model.config.hidden_size
    gain = norm_gain(model.model.norm).clone()
    reference = timed("reference", lambda: output_logits(model, pre_final_states(model, ev)))
    result["arms"]["stock"] = {"generation": timed("stock_generation", lambda: generate(model, tokenizer, gen, args.max_new))}
    q, result["pca"] = timed("pca", lambda: collect_basis(model, fit))
    p = q[:, :args.width].contiguous()
    scales = timed("rms_scales", lambda: rms_scales(model, fit, p))
    result["rms_scales"] = scales
    c = scales[-1]
    timed("project_body", lambda: project_residual_width(model, p, scales))
    projected_hidden = timed("projected_hidden", lambda: pre_final_states(model, ev))
    diagonal = copy.deepcopy(model.model.norm)
    dense = projected_output_matrix(gain, p, 1.)
    offset = 1 if model.config.model_type == "qwen3_5_text" else 0
    with torch.no_grad():
        diagonal.weight.copy_(dense.diagonal() / c - offset)
    for arm in ["legacy_diagonal_divide", "dense_divide", "dense_multiply"]:
        if arm == "legacy_diagonal_divide":
            model.model.norm = diagonal
        else:
            install_output_matrix(model, dense / c if arm == "dense_divide" else dense * c)
        metrics = compare(reference, output_logits(model, projected_hidden))
        result["arms"][arm] = {"comparison": metrics, "parameters": sum(w.numel() for w in model.parameters())}
        log(f"{arm}: KL {metrics['mean_kl']:.6f}, top1 {metrics['top1_agreement']:.3f}")
        result["arms"][arm]["generation"] = timed(arm + "_generation", lambda: generate(model, tokenizer, gen, args.max_new))
        save(result, args.out)

    result["adapter_parameters"] = model.model.norm.matrix.numel()
    expected = output_logits(model, projected_hidden)
    timed("save_fp32", lambda: model.save_pretrained(args.save_model))
    tokenizer.save_pretrained(args.save_model)
    del model, projected_hidden, reference
    gc.collect()
    reloaded, loading = timed("reload_fp32", lambda: load_causal_model(
        args.save_model, dtype=torch.float32, local_files_only=True,
        attn_implementation="eager", output_loading_info=True))
    result["loading"] = loading
    save(result, args.out)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError("Checkpoint reload reported unmatched weights; see result loading info")
    reloaded.eval().requires_grad_(False)
    result["roundtrip"] = compare(expected, output_logits(reloaded, pre_final_states(reloaded, ev)))
    result["cache"] = cache_check(reloaded, ev[0]["ids"])
    result["elapsed_seconds"] = time.perf_counter() - start
    save(result, args.out)
    if not result["roundtrip"]["numerical_pass"] or not result["cache"]["numerical_pass"]:
        raise RuntimeError("Checkpoint roundtrip or cache verification failed; see result")
    log(f"Completed in {result['elapsed_seconds']:.1f}s; roundtrip {result['roundtrip']}")


if __name__ == "__main__":
    main()
