"""Zero-compression rotation and cached-inference checks for Qwen3.5 and Llama.

This is a diagnostic, not a compressed checkpoint builder. Every residual
coordinate and MLP neuron is retained. The exact arm retains the full rotated
final gain matrix; the diagonal arm reproduces build_projected.py's shortcut.
Run in FP32 to distinguish algebraic error from BF16 rounding.
"""
import argparse
import copy
import json
import platform
import time
from pathlib import Path

import torch
from torch import nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


CALIBRATION = [
    "Explain how a bicycle's gears help a rider climb a hill. Use a concrete example.",
    "Write a Python function that counts the occurrences of each word in a sentence.",
    "A train travels 180 kilometres in two and a half hours. Explain how to find its average speed.",
    "Rewrite this politely: Your report is late and I need it before tomorrow's meeting.",
]
EVALUATION = [
    "What causes the seasons on Earth? Answer in two sentences.",
    "Calculate 17 times 23 and briefly explain your calculation.",
    "Write a JavaScript function that removes duplicate items from an array.",
    "Give exactly three practical steps for planning a small vegetable garden.",
]


def log(message):
    print(message, flush=True)


def backbone(model):
    return model.model


def norm_gain(norm):
    offset = 1.0 if norm.__class__.__name__ == "Qwen3_5RMSNorm" else 0.0
    return norm.weight.detach().float() + offset


def norm_epsilon(norm):
    return norm.eps if hasattr(norm, "eps") else norm.variance_epsilon


class RotatedFinalNorm(nn.Module):
    """RMS normalization followed by Q.T @ diag(gain) @ Q.

    Keeping this small dense matrix preserves tied input/output embeddings.
    This custom module is used only in memory, not exported as a stock HF model.
    """
    def __init__(self, matrix, eps):
        super().__init__()
        self.register_buffer("matrix", matrix)
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (y @ self.matrix.float()).to(x.dtype)


@torch.no_grad()
def rotate_full_width(model, q, progress=False):
    """Fold a common orthogonal basis into every reader and writer, without cuts."""
    lm = backbone(model)
    if model.config.model_type not in ("qwen3_5_text", "llama", "mistral"):
        raise ValueError(f"Unsupported architecture: {model.config.model_type}")
    d = lm.embed_tokens.weight.shape[1]
    if q.shape != (d, d):
        raise ValueError("The audit requires a square, full-width rotation")
    if (q.T @ q - torch.eye(d, device=q.device)).abs().max().item() > 1e-5:
        raise ValueError("Basis is not orthogonal")
    tied = lm.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    old_final = copy.deepcopy(lm.norm)
    g = norm_gain(old_final)
    full_gain = q.double().T @ (g.double()[:, None] * q.double())
    full_gain = full_gain.float()

    def reader(module, gain):
        # A residual-space input enters the existing head/neuron coordinates.
        module.weight.copy_((module.weight * gain[None, :]) @ q)

    def writer(module):
        module.weight.copy_(q.T @ module.weight)
        if module.bias is not None:
            module.bias.copy_(q.T @ module.bias)

    lm.embed_tokens.weight.copy_(lm.embed_tokens.weight @ q)
    if not tied:
        model.lm_head.weight.copy_(model.lm_head.weight @ q)
    for i, layer in enumerate(lm.layers):
        if hasattr(layer, "linear_attn"):
            mixer = layer.linear_attn
            names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
            output = "out_proj"
        else:
            mixer = layer.self_attn
            names, output = ("q_proj", "k_proj", "v_proj"), "o_proj"
        gain = norm_gain(layer.input_layernorm)
        for name in names:
            reader(getattr(mixer, name), gain)
        writer(getattr(mixer, output))
        gain = norm_gain(layer.post_attention_layernorm)
        for name in ("gate_proj", "up_proj"):
            reader(getattr(layer.mlp, name), gain)
        writer(layer.mlp.down_proj)
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            norm.weight.fill_(0 if norm.__class__.__name__ == "Qwen3_5RMSNorm" else 1)
        if progress and (i + 1) % 4 == 0:
            log(f"Rotated {i + 1}/{len(lm.layers)} layers")

    exact = RotatedFinalNorm(full_gain, norm_epsilon(old_final))
    diagonal = copy.deepcopy(old_final)
    offset = 1 if old_final.__class__.__name__ == "Qwen3_5RMSNorm" else 0
    diagonal.weight.copy_(full_gain.diagonal() - offset)
    lm.norm = exact
    off = full_gain - torch.diag(full_gain.diagonal())
    details = {
        "tied_embeddings": tied,
        "final_gain_min": g.min().item(),
        "final_gain_max": g.max().item(),
        "discarded_off_diagonal_frobenius_fraction": (off.norm() / full_gain.norm()).item(),
    }
    return exact, diagonal, details


@torch.no_grad()
def collect_basis(model, rows):
    """Match the project's shared, uncentered, per-site trace-normalized PCA."""
    lm = backbone(model)
    norms = [n for layer in lm.layers for n in (layer.input_layernorm, layer.post_attention_layernorm)] + [lm.norm]
    d = lm.embed_tokens.weight.shape[1]
    covs = [torch.zeros(d, d) for _ in norms]
    counts = [0] * len(norms)
    handles = []
    for i, norm in enumerate(norms):
        def hook(module, args, i=i):
            x = args[0].reshape(-1, d).float()
            covs[i].addmm_(x.T, x)
            counts[i] += len(x)
        handles.append(norm.register_forward_pre_hook(hook))
    try:
        for i, ids in enumerate(rows):
            lm(input_ids=ids, use_cache=False)
            log(f"Calibration {i + 1}/{len(rows)}: {ids.numel()} tokens")
    finally:
        for handle in handles:
            handle.remove()
    pooled = sum(c.double() / c.trace().double() for c in covs)
    values, vectors = torch.linalg.eigh(pooled)
    q = vectors.flip(1).float().contiguous()
    return q, {"sites": len(norms), "tokens_per_site": counts, "eigenvalue_min": values[0].item()}


@torch.no_grad()
def logits(model, rows):
    # Backbone first, then short head chunks avoid full 248K-vocabulary temporaries.
    outputs = []
    for ids in rows:
        h = backbone(model)(input_ids=ids, use_cache=False).last_hidden_state
        outputs.append(torch.cat([model.lm_head(x).float() for x in h.split(16, dim=1)], dim=1))
    return outputs


def compare(reference, candidate):
    max_abs = sum_sq = kl_sum = max_kl = 0.0
    elements = tokens = same = 0
    for a, b in zip(reference, candidate, strict=True):
        assert a.shape == b.shape
        for x, y in zip(a.reshape(-1, a.shape[-1]).split(8), b.reshape(-1, b.shape[-1]).split(8), strict=True):
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError("Cannot compare non-finite logits")
            diff = (x - y).double()
            max_abs = max(max_abs, diff.abs().max().item())
            sum_sq += diff.square().sum().item()
            elements += diff.numel()
            lp, lq = x.double().log_softmax(-1), y.double().log_softmax(-1)
            kl = (lp.exp() * (lp - lq)).sum(-1)
            kl_sum += kl.sum().item()
            max_kl = max(max_kl, kl.max().item())
            tokens += len(x)
            same += (x.argmax(-1) == y.argmax(-1)).sum().item()
    return {"positions": tokens, "max_abs_logit_error": max_abs,
            "rms_logit_error": (sum_sq / elements) ** 0.5,
            "mean_kl": kl_sum / tokens, "max_token_kl": max_kl,
            "top1_agreement": same / tokens,
            "numerical_pass": max_abs < 0.005 and max_kl < 1e-6}


@torch.no_grad()
def cache_check(model, ids, decode_steps=8):
    """Feed the SAME tokens by full sequence vs prefill + incremental cache."""
    n = min(decode_steps, ids.shape[1] - 1)
    prefix = ids.shape[1] - n
    full = model(input_ids=ids, use_cache=False, logits_to_keep=n).logits.float()
    out = model(input_ids=ids[:, :prefix], use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    step_logits = []
    for pos in range(prefix, ids.shape[1]):
        out = model(input_ids=ids[:, pos:pos + 1], past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        step_logits.append(out.logits.float())
    return compare([full], [torch.cat(step_logits, dim=1)])


def render_rows(tokenizer, prompts, limit):
    return [tokenizer(tokenizer.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False),
        add_special_tokens=False, return_tensors="pt").input_ids[:, :limit] for p in prompts]


def save(result, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(value):
        if isinstance(value, set):
            return sorted(value)
        raise TypeError(f"Cannot serialize {type(value)}")
    path.write_text(json.dumps(result, indent=2, default=encode) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--cache-only", action="store_true", help="Check a recovered checkpoint without rotating it")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    start = time.perf_counter()
    result = {"args": vars(args) | {"out": str(args.out)},
              "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                              "platform": platform.platform(), "dtype": "float32", "device": "cpu"},
              "calibration_prompts": CALIBRATION, "evaluation_prompts": EVALUATION,
              "timings_seconds": {}, "checks": {}}
    manifest = Path(args.model) / "download_manifest.json"
    if manifest.exists():
        result["source"] = json.loads(manifest.read_text())

    def timed(name, fn):
        t = time.perf_counter()
        value = fn()
        result["timings_seconds"][name] = time.perf_counter() - t
        log(f"{name}: {result['timings_seconds'][name]:.2f}s")
        save(result, args.out)
        return value

    model, loading = timed("load", lambda: AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager", local_files_only=True,
        output_loading_info=True))
    # Qwen's source checkpoint also contains a vision encoder. Only unused vision
    # and MTP weights are acceptable; missing text weights invalidate this audit.
    result["loading"] = {k: v for k, v in loading.items() if k != "conversion_errors"}
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        save(result, args.out)
        raise RuntimeError(f"Invalid text-model load: {loading}")
    model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    calib = render_rows(tokenizer, CALIBRATION, args.max_tokens)
    evaluation = render_rows(tokenizer, EVALUATION, args.max_tokens)
    result["model_type"] = model.config.model_type
    result["parameters"] = sum(p.numel() for p in model.parameters())
    ref = timed("reference_forward", lambda: logits(model, evaluation))
    result["checks"]["original_cache"] = timed("original_cache", lambda: cache_check(model, evaluation[0]))
    if args.cache_only:
        result["elapsed_seconds"] = time.perf_counter() - start
        save(result, args.out)
        log(json.dumps(result["checks"], indent=2))
        return
    q, result["basis"] = timed("pca", lambda: collect_basis(model, calib))
    exact, diagonal, result["rotation"] = timed("fold_rotation", lambda: rotate_full_width(model, q, progress=True))
    rotated = timed("exact_forward", lambda: logits(model, evaluation))
    result["checks"]["exact_rotation"] = compare(ref, rotated)
    result["checks"]["rotated_cache"] = timed("rotated_cache", lambda: cache_check(model, evaluation[0]))
    backbone(model).norm = diagonal
    approx = timed("diagonal_forward", lambda: logits(model, evaluation))
    result["checks"]["diagonal_final_norm"] = compare(ref, approx)
    result["elapsed_seconds"] = time.perf_counter() - start
    save(result, args.out)
    log(json.dumps(result["checks"], indent=2))


if __name__ == "__main__":
    main()
