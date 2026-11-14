"""Sequential, no-healing scale experiment with explicit parameter targets.

The GPU run uses BF16 weights/inference and FP32 projection arithmetic. A
separate FP32, full-width rotation gate isolates algebra from BF16 rounding.
Only one model is resident; original checkpoints are never mutated on disk.
"""
import argparse
import copy
import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from audit_rotation import compare, norm_epsilon, norm_gain, cache_check
from healing_data import (chat_ids, encode_complete, check_answer, generation_metrics,
                          scoring_spec)
from projection_models import (CLASSES, install_output_matrix, load_causal_model,
                               projected_output_matrix, project_residual_width)
from scale_data import load_tokenizer, encoding_digest

SOURCES = {
    "qwen27": {"repo": "Qwen/Qwen3.5-27B", "revision": "fc05daec18b0a78c049392ed2e771dde82bdf654"},
    "mistral24": {"repo": "mistralai/Mistral-Small-24B-Instruct-2501", "revision": "9527884be6e5616bdd54de542f9ae13384489724"},
    "qwen2": {"repo": "Qwen/Qwen3.5-2B", "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc"},
}


def save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    def encode(obj):
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError(f"Cannot serialize {type(obj)}")
    tmp.write_text(json.dumps(value, indent=2, default=encode, allow_nan=False) + "\n")
    tmp.replace(path)


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def text_config(path):
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    config = copy.deepcopy(getattr(config, "text_config", config))
    if not getattr(config, "head_dim", None):
        config.head_dim = config.hidden_size // config.num_attention_heads
    if config.model_type not in CLASSES:
        raise ValueError(f"Unsupported architecture: {config.model_type}")
    return config


def count_config(config, width=None, adapter=False):
    config = copy.deepcopy(config)
    if width is not None:
        config.hidden_size = width
    if adapter:
        config.manifold_output_norm = "matrix_v1"
    with torch.device("meta"):
        model = CLASSES[config.model_type](config)
    return sum(p.numel() for p in model.parameters())


def parameter_plan(config, fractions=(.75, .5), alignment=16):
    original = count_config(config)
    d = config.hidden_size
    # The unchanged internal attention/MLP dimensions make parameter count
    # affine in residual width, plus the k*k final output matrix.
    c1, c2 = count_config(config, 1), count_config(config, 2)
    slope, intercept = c2 - c1, 2 * c1 - c2
    if slope * d + intercept != original:
        raise ValueError("Architecture is not affine in residual width")
    arms = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Retention must be in (0, 1]")
        widths = range(alignment, d + 1, alignment)
        eligible = [k for k in widths if slope * k + intercept + k * k <= original * fraction]
        if not eligible:
            raise ValueError("Target too small for unchanged internal dimensions")
        k = max(eligible)
        count = count_config(config, k, adapter=True)
        if count != slope * k + intercept + k * k:
            raise ValueError("Parameter prediction disagrees with actual model construction")
        arms.append({"target_fraction": fraction, "width": k, "parameters": count,
                     "retained_fraction": count / original, "bf16_weight_gb": count * 2 / 1e9,
                     "output_matrix_parameters": k * k})
    return {"model_type": config.model_type, "source_width": d, "source_parameters": original,
            "source_bf16_weight_gb": 2 * original / 1e9,
            "source_fp32_weight_gb": 4 * original / 1e9, "arms": arms,
            "fixed_parameters": intercept, "parameters_per_residual_coordinate": slope,
            "count_scope": "text causal model, including embeddings, output head, and output adapter; no vision/MTP"}


def residual_norms(model):
    return [n for layer in model.model.layers for n in
            (layer.input_layernorm, layer.post_attention_layernorm)] + [model.model.norm]


@torch.no_grad()
def collect_basis(model, rows):
    norms = residual_norms(model)
    d = model.config.hidden_size
    device = next(model.parameters()).device
    covs = [torch.zeros(d, d, device=device, dtype=torch.float32) for _ in norms]
    counts = [0] * len(norms)
    hooks = []
    for i, norm in enumerate(norms):
        def hook(module, args, i=i):
            x = args[0].reshape(-1, d).float()
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite calibration states")
            covs[i].addmm_(x.T, x)
            counts[i] += len(x)
        hooks.append(norm.register_forward_pre_hook(hook))
    try:
        for i, row in enumerate(rows):
            model.model(input_ids=torch.tensor([row["input_ids"]], device=device), use_cache=False)
            if (i + 1) % 16 == 0:
                log(f"PCA: {i+1}/{len(rows)} conversations")
    finally:
        for h in hooks:
            h.remove()
    pooled = torch.zeros(d, d, device=device, dtype=torch.float64)
    traces = []
    for cov in covs:
        trace = cov.double().diagonal().sum()
        if not torch.isfinite(trace) or trace <= 0:
            raise ValueError("Invalid calibration covariance")
        traces.append(trace.item())
        pooled.add_(cov.double() / trace)
    del covs, cov
    # CPU eigensolve avoids CUDA solver workspace surprises; this is only d*d.
    values, vectors = torch.linalg.eigh(pooled.cpu())
    basis = vectors.flip(1).float().contiguous()
    return basis, {"sites": len(norms), "tokens_per_site": counts, "traces": traces,
                   "eigenvalues_descending": values.flip(0).tolist(),
                   "method": "shared uncentered PCA, each residual site normalized by its covariance trace"}


@torch.no_grad()
def collect_scales(model, rows, basis, widths):
    norms = residual_norms(model)
    device = next(model.parameters()).device
    q = basis.to(device)
    totals = torch.zeros(len(norms), len(widths), dtype=torch.float64, device=device)
    counts = [0] * len(norms)
    hooks = []
    for i, norm in enumerate(norms):
        eps = norm_epsilon(norm)
        def hook(module, args, i=i, eps=eps):
            x = args[0].reshape(-1, q.shape[0]).float()
            energy = (x @ q).square().cumsum(-1)
            denom = x.square().mean(-1) + eps
            for j, k in enumerate(widths):
                totals[i, j] += ((energy[:, k-1] / k + eps) / denom).sqrt().double().sum()
            counts[i] += len(x)
        hooks.append(norm.register_forward_pre_hook(hook))
    try:
        for i, row in enumerate(rows):
            model.model(input_ids=torch.tensor([row["input_ids"]], device=device), use_cache=False)
            if (i + 1) % 16 == 0:
                log(f"RMS scales: {i+1}/{len(rows)} conversations")
    finally:
        for h in hooks:
            h.remove()
    return {str(k): (totals[:, j].cpu() / torch.tensor(counts)).tolist() for j, k in enumerate(widths)}


def validation_rows(data, tokenizer, positions_per_row=16):
    output = []
    for item in data["validation"]:
        row = encode_complete(tokenizer, item["messages"], data["max_length"])
        if row is None:
            raise ValueError("Frozen validation conversation no longer fits")
        # Equally spaced final-assistant positions, always including EOS.
        p = row["positions"]
        indices = torch.linspace(0, len(p)-1, min(positions_per_row, len(p))).round().long().tolist()
        row["positions"] = [p[i] for i in indices]
        row["targets"] = [row["targets"][i] for i in indices]
        row["id"] = item["id"]
        output.append(row)
    return output


@torch.no_grad()
def row_logits(model, row):
    device = next(model.parameters()).device
    ids = torch.tensor([row["input_ids"]], device=device)
    h = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, row["positions"]]
    return torch.cat([model.lm_head(x).float().cpu() for x in h.split(8, dim=1)], dim=1)


@torch.no_grad()
def evaluate_logits(model, rows, reference_dir, write_reference=False):
    total = {"positions": 0, "kl_sum": 0., "correct": 0, "same": 0, "nll_sum": 0.}
    cases = []
    reference_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        candidate = row_logits(model, row)
        if not torch.isfinite(candidate).all():
            raise ValueError("Nonfinite evaluation logits")
        path = reference_dir / f"{i:03d}.pt"
        if write_reference:
            torch.save(candidate, path)
            reference = candidate
        else:
            reference = torch.load(path, weights_only=True)
        metrics = compare([reference], [candidate])
        targets = torch.tensor(row["targets"])
        lp = candidate[0].double().log_softmax(-1)
        nll = -lp[torch.arange(len(targets)), targets].sum().item()
        n = len(targets)
        correct = (candidate[0].argmax(-1) == targets).sum().item()
        total["positions"] += n
        total["kl_sum"] += n * metrics["mean_kl"]
        total["same"] += round(n * metrics["top1_agreement"])
        total["correct"] += correct
        total["nll_sum"] += nll
        cases.append({"id": row["id"], **metrics, "target_nll": nll / n, "target_top1": correct / n})
    n = total["positions"]
    return {"positions": n, "mean_kl": total["kl_sum"] / n, "top1_agreement": total["same"] / n,
            "target_nll": total["nll_sum"] / n, "target_top1": total["correct"] / n, "cases": cases}


@torch.no_grad()
def generate(model, tokenizer, cases, out, batch_size=4):
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    result = {"decoding": "greedy; thinking off; no repetition/frequency/presence penalty",
              "cases": []}
    start = time.monotonic()
    # Fresh config prevents repository-specific sampling penalties affecting arms.
    generation_config = GenerationConfig(do_sample=False, use_cache=True,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, repetition_penalty=1., no_repeat_ngram_size=0)
    for limit in sorted({c["max_new_tokens"] for c in cases}):
        generation_config.max_new_tokens = limit
        group = [c for c in cases if c["max_new_tokens"] == limit]
        for start_idx in range(0, len(group), batch_size):
            batch_cases = group[start_idx:start_idx + batch_size]
            texts = [tokenizer.apply_chat_template(c["messages"], tokenize=False,
                add_generation_prompt=True, enable_thinking=False) for c in batch_cases]
            batch = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt").to(device)
            seqs = model.generate(**batch, generation_config=generation_config)
            for case, seq in zip(batch_cases, seqs[:, batch.input_ids.shape[1]:], strict=True):
                ids = seq.tolist()
                if tokenizer.eos_token_id in ids:
                    ids = ids[:ids.index(tokenizer.eos_token_id)+1]
                answer = tokenizer.decode(ids, skip_special_tokens=True)
                spec = scoring_spec(case) if case["check"] is not None else None
                result["cases"].append({"id": case["id"], "answer": answer, "check": spec,
                    "check_pass": check_answer(answer, spec) if spec else None, "max_new_tokens": limit,
                    **generation_metrics(answer, ids, tokenizer.eos_token_id)})
            log(f"Generation {len(result['cases'])}/{len(cases)}")
            save(result, out)
    result["summary"] = {"cases": len(cases), "scored_cases": sum(c["check"] is not None for c in cases),
        "check_passes": sum(c["check_pass"] is True for c in result["cases"]),
        "completed": sum(c["ended"] for c in result["cases"]),
        "loop_flags": sum(c["loop_flag"] for c in result["cases"]),
        "generated_tokens": sum(c["generated_tokens"] for c in result["cases"]),
        "elapsed_seconds": time.monotonic() - start}
    save(result, out)
    return result["summary"]


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def use_reference_delta_kernels(model):
    """Use FP32-capable kernels for both sides of the algebra gate."""
    if model.config.model_type != "qwen3_5_text":
        return []
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule, torch_causal_conv1d_update)
    previous = []
    for layer in model.model.layers:
        if hasattr(layer, "linear_attn"):
            m = layer.linear_attn
            previous.append((m, {name: getattr(m, name) for name in
                ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule", "recurrent_gated_delta_rule")}))
            m.causal_conv1d_fn = None
            m.causal_conv1d_update = torch_causal_conv1d_update
            m.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
            m.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
    return previous


def cast_parameters(model, dtype):
    # Preserve native FP32 rotary buffers when moving weights between FP32 and
    # BF16. model.bfloat16() would round those buffers, adding unrelated error.
    for parameter in model.parameters():
        parameter.data = parameter.data.to(dtype)


def load_source(path, device):
    config = text_config(path)
    model, info = AutoModelForCausalLM.from_pretrained(path, config=config, dtype=torch.bfloat16,
        device_map={"": device}, local_files_only=True, attn_implementation="eager",
        output_loading_info=True)
    # Multimodal and optional MTP weights are outside this text-only experiment.
    allowed = ("model.visual.", "visual.", "mtp.", "model.mtp.")
    unmatched = [k for k in info.get("unexpected_keys", []) if not k.startswith(allowed)]
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs") or unmatched:
        raise RuntimeError(f"Unmatched causal model weights: {info}")
    return model.eval().requires_grad_(False), info


def run(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use an empty output directory; never mix partial runs")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(24027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = json.loads(args.data.read_text())
    tokenizer = load_tokenizer(args.source)
    if encoding_digest(data, tokenizer) != data["encoded_sha256"][args.model]:
        raise ValueError("Tokenization differs from the frozen local preparation")
    config = text_config(args.source)
    plan = parameter_plan(config)
    if args.device.startswith("cuda"):
        available = torch.cuda.get_device_properties(args.device).total_memory
        # FP32 full-width rotation + two vocabulary projections + scratch.
        needed = plan["source_parameters"] * 4 + 2 * config.vocab_size * config.hidden_size * 4 + 6 * 2**30
        if available < needed:
            raise RuntimeError(f"Full-width FP32 gate needs about {needed/2**30:.1f} GiB; device has {available/2**30:.1f}")
    calibration = [encode_complete(tokenizer, r["messages"], data["max_length"]) for r in data["calibration"]]
    if any(r is None for r in calibration):
        raise ValueError("Frozen calibration conversation no longer fits")
    rows = validation_rows(data, tokenizer)
    cases = data["generation_eval"]
    if args.smoke:
        calibration, rows, cases = calibration[:2], rows[:2], cases[:2]
        cases = [c | {"max_new_tokens": 16} for c in cases]
    result = {"status": "running", "model": args.model, "source": SOURCES[args.model], "plan": plan,
              "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(), "smoke_only": args.smoke,
              "calibration_tokens": sum(len(r["input_ids"]) for r in calibration),
              "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
              "device": args.device, "projection_dtype": "float32", "inference_dtype": "bfloat16", "arms": {}}
    result_path = args.out / "result.json"
    start = time.monotonic()
    def checkpoint(stage):
        result["stage"] = stage
        result["elapsed_seconds"] = time.monotonic() - start
        if args.device.startswith("cuda"):
            result["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
        save(result, result_path)
        log(stage)
    checkpoint("load stock")
    model, result["loading"] = load_source(args.source, args.device)
    if sum(p.numel() for p in model.parameters()) != plan["source_parameters"]:
        raise ValueError("Loaded parameter count differs from preflight")
    checkpoint("stock held-out logits")
    reference_dir = args.out / "teacher_logits"
    stock = {"validation": evaluate_logits(model, rows, reference_dir, True)}
    result["arms"]["stock"] = stock
    checkpoint("stock generation")
    stock["generation"] = generate(model, tokenizer, cases, args.out / "stock-generation.json", args.batch_size)
    cache_ids = torch.tensor([rows[0]["input_ids"][:128]], device=args.device)
    stock["cache"] = cache_check(model, cache_ids)
    if stock["cache"]["max_token_kl"] > .05:
        raise RuntimeError("Stock cached inference gate failed")
    checkpoint("calibration PCA")
    basis, result["pca"] = collect_basis(model, calibration)
    torch.save(basis, args.out / "basis.pt")
    cleanup()
    checkpoint("calibration RMS correction")
    widths = [a["width"] for a in plan["arms"]]
    scales = collect_scales(model, calibration, basis, widths)
    save(scales, args.out / "rms-scales.json")
    checkpoint("FP32 full-width rotation gate")
    original_kernels = use_reference_delta_kernels(model)
    cast_parameters(model, torch.float32)
    cleanup()
    # Short fixed prefixes suffice for the algebra gate; quality uses full rows.
    gate_rows = [{"input_ids": r["input_ids"][:128], "positions": list(range(min(len(r["input_ids"]), 128)-8,
                  min(len(r["input_ids"]), 128)))} for r in rows[:2]]
    expected = [row_logits(model, r) for r in gate_rows]
    q = basis.to(args.device)
    gain = norm_gain(model.model.norm)
    project_residual_width(model, q, [1.] * len(residual_norms(model)), chunk_size=1024)
    install_output_matrix(model, projected_output_matrix(gain, q, 1.))
    result["rotation_gate"] = compare(expected, [row_logits(model, r) for r in gate_rows])
    result["rotation_cache"] = cache_check(model, cache_ids)
    checkpoint("full-width gate measured")
    if not result["rotation_gate"]["numerical_pass"] or not result["rotation_cache"]["numerical_pass"]:
        raise RuntimeError("FP32 rotation/cache gate failed; compressed arms are blocked")
    cast_parameters(model, torch.bfloat16)
    for module, functions in original_kernels:
        for name, function in functions.items():
            setattr(module, name, function)
    checkpoint("BF16 full-width rounding control")
    result["bf16_rotation_control"] = evaluate_logits(model, rows, reference_dir)
    rounding = result["bf16_rotation_control"]
    checkpoint("BF16 full-width control measured")
    if rounding["mean_kl"] > .02 or rounding["top1_agreement"] < .95:
        raise RuntimeError("BF16 full-width control drifted too far; fix precision before interpreting compression")
    del original_kernels
    del model, q, gain, expected
    cleanup()
    for arm in plan["arms"]:
        label = f"retain-{round(100 * arm['target_fraction'])}"
        checkpoint(f"{label}: load original")
        model, _ = load_source(args.source, args.device)
        p = basis[:, :arm["width"]].contiguous().to(args.device)
        gains = norm_gain(model.model.norm)
        correction = scales[str(arm["width"])]
        project_residual_width(model, p, correction, chunk_size=1024)
        matrix = projected_output_matrix(gains, p, correction[-1]).to(torch.bfloat16)
        install_output_matrix(model, matrix)
        actual = sum(w.numel() for w in model.parameters())
        if actual != arm["parameters"]:
            raise ValueError("Projected parameter count differs from plan")
        entry = {"parameters": actual, "width": arm["width"], "retained_fraction": actual / plan["source_parameters"],
                 "cache": cache_check(model, cache_ids)}
        result["arms"][label] = entry
        checkpoint(f"{label}: cache measured")
        if entry["cache"]["max_token_kl"] > max(.05, 5 * stock["cache"]["max_token_kl"]):
            raise RuntimeError("Projected cached inference gate failed")
        entry["validation"] = evaluate_logits(model, rows, reference_dir)
        checkpoint(f"{label}: generation")
        entry["generation"] = generate(model, tokenizer, cases, args.out / f"{label}-generation.json", args.batch_size)
        # Basis/scales plus pinned source reconstruct these exact unhealed arms.
        # Do not spend the GPU budget transferring tens of GB of weight copies.
        model.config.save_pretrained(args.out / f"{label}-config")
        del model, p, gains, matrix
        cleanup()
    result["status"] = "complete"
    checkpoint("complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["plan", "run"])
    ap.add_argument("--model", choices=list(SOURCES), required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="Two rows, two short generations; never a quality result")
    args = ap.parse_args()
    if args.stage == "plan":
        report = parameter_plan(text_config(args.source)) | {"source": SOURCES[args.model]}
        save(report, args.out)
        print(json.dumps(report, indent=2))
    else:
        if args.data is None:
            ap.error("run requires --data")
        run(args)
