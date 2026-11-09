"""Qwen3.6-35B-A3B with 4-bit routed experts, for LoRA training on this machine.

The bf16 checkpoint is 67 GB, of which the routed experts (40 layers x 256 experts, stored as
3D tensors, so bitsandbytes leaves them alone) are 64 GB. GPU memory here is the 123 GB of
system RAM everyone shares, and the bf16 load exhausted it (2026-11-08). So:

  convert  once, on the CPU, shard by shard: drop the vision tower and the MTP layer, rename
           to the text-only CausalLM layout, and quantize every expert tensor to 4 bits with an
           asymmetric scale and minimum per group of 64 input weights (4.5 bits per weight).
           Everything else (attention, linear attention, shared experts, router, norms,
           embeddings, lm_head) stays bf16. About 21 GB on disk.
  load     builds the model on the meta device, swaps each experts module for Q4Experts, then
           fills it tensor by tensor from the converted shards straight onto the GPU.

Q4Experts keeps the packed weights as frozen buffers. Its forward dequantizes the layer's
experts to bf16 with a Triton kernel (1.6 GB, transient; recomputed under gradient
checkpointing) and computes the same function as transformers' eager Qwen3_5MoeExperts, as two
grouped matmuls over tokens sorted by expert. The linear-attention layers get a
channel-contiguous causal conv in place of MIOpen's; run with PYTHONPATH=.fla so they also use
flash-linear-attention's Triton delta-rule kernel. LoRA goes on the bf16
linear layers only; experts and router stay frozen.

    python3 q4moe.py convert SRC_DIR DST_DIR          (CPU only)
    python3 q4moe.py check SRC_DIR DST_DIR            (CPU: quantization error on a few layers)
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

GROUP = 64
SHARD_BYTES = 4 << 30


# ---------------------------------------------------------------- 4-bit format

def quantize(w: torch.Tensor, group: int = GROUP) -> dict[str, torch.Tensor]:
    """w [..., out, in] -> packed uint8 [..., out, in/2], scale and min bf16 [..., out, in/group]."""
    *lead, o, i = w.shape
    assert i % group == 0 and group % 2 == 0
    g = w.float().reshape(*lead, o, i // group, group)
    lo = g.amin(-1, keepdim=True)
    hi = g.amax(-1, keepdim=True)
    scale = ((hi - lo) / 15).clamp_min(1e-8)
    q = ((g - lo) / scale).round_().clamp_(0, 15).to(torch.uint8).reshape(*lead, o, i)
    packed = q[..., 0::2] | (q[..., 1::2] << 4)
    return {"q": packed.contiguous(), "s": scale.squeeze(-1).to(torch.bfloat16), "m": lo.squeeze(-1).to(torch.bfloat16)}


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _dequant_kernel(q_ptr, s_ptr, m_ptr, o_ptr, HALF: tl.constexpr, NG: tl.constexpr,
                        GHALF: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < HALF
        b = tl.load(q_ptr + row * HALF + offs, mask=mask, other=0).to(tl.int32)
        g = offs // GHALF
        sc = tl.load(s_ptr + row * NG + g, mask=mask, other=0).to(tl.float32)
        mn = tl.load(m_ptr + row * NG + g, mask=mask, other=0).to(tl.float32)
        lo = (b & 15).to(tl.float32) * sc + mn
        hi = (b >> 4).to(tl.float32) * sc + mn
        out = tl.interleave(lo, hi).to(tl.bfloat16)
        o = tl.program_id(1) * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
        tl.store(o_ptr + row * (2 * HALF) + o, out, mask=o < 2 * HALF)
except ImportError:  # CPU-only environments
    triton = None


def dequantize(q: torch.Tensor, s: torch.Tensor, m: torch.Tensor, group: int = GROUP) -> torch.Tensor:
    """Packed 4-bit -> bf16. One Triton pass on the GPU; plain torch elsewhere."""
    if triton is not None and q.is_cuda:
        *lead, o, half = q.shape
        rows = q.numel() // half
        out = torch.empty(*lead, o, 2 * half, dtype=torch.bfloat16, device=q.device)
        block = min(1024, triton.next_power_of_2(half))
        _dequant_kernel[(rows, triton.cdiv(half, block))](q, s, m, out, HALF=half, NG=2 * half // group,
                                                           GHALF=group // 2, BLOCK=block)
        return out
    return dequantize_torch(q, s, m, group)


def dequantize_torch(q: torch.Tensor, s: torch.Tensor, m: torch.Tensor, group: int = GROUP) -> torch.Tensor:
    *lead, o, half = q.shape
    i = half * 2
    x = torch.stack((q & 15, q >> 4), dim=-1).reshape(*lead, o, i // group, group).to(torch.bfloat16)
    return (x * s.unsqueeze(-1) + m.unsqueeze(-1)).reshape(*lead, o, i)


class Q4Experts(nn.Module):
    """Drop-in for Qwen3_5MoeExperts with 4-bit frozen weights."""

    def __init__(self, num_experts: int, hidden: int, inter: int, act_fn, device=None, group: int = GROUP):
        super().__init__()
        self.num_experts, self.hidden_dim, self.intermediate_dim, self.group = num_experts, hidden, inter, group
        self.act_fn = act_fn
        for name, (o, i) in {"gate_up": (2 * inter, hidden), "down": (hidden, inter)}.items():
            self.register_buffer(f"{name}_q", torch.empty(num_experts, o, i // 2, dtype=torch.uint8, device=device))
            self.register_buffer(f"{name}_s", torch.empty(num_experts, o, i // group, dtype=torch.bfloat16, device=device))
            self.register_buffer(f"{name}_m", torch.empty(num_experts, o, i // group, dtype=torch.bfloat16, device=device))

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        """Tokens sorted by expert, then one grouped matmul per projection (torch._grouped_mm with
        per-expert offsets). A zero-padded [E, max count, H] batch was faster on uniform routing
        but real routing is skewed: one hot expert made the padding 12.6 GB at 16k tokens."""
        k = top_k_index.shape[-1]
        flat_e = top_k_index.reshape(-1)
        order = torch.argsort(flat_e, stable=True)
        token = order // k
        offs = torch.cumsum(torch.bincount(flat_e, minlength=self.num_experts), 0).to(torch.int32)
        w_gu = dequantize(self.gate_up_q, self.gate_up_s, self.gate_up_m, self.group)
        w_d = dequantize(self.down_q, self.down_s, self.down_m, self.group)
        gate, up = torch._grouped_mm(hidden_states[token], w_gu.transpose(1, 2), offs=offs).chunk(2, dim=-1)
        y = torch._grouped_mm(self.act_fn(gate) * up, w_d.transpose(1, 2), offs=offs)
        contrib = y * top_k_weights.reshape(-1)[order, None].to(y.dtype)
        return torch.zeros_like(hidden_states).index_add(0, token, contrib.to(hidden_states.dtype))


def causal_conv1d_btc(x, weight, bias=None, activation="silu", seq_idx=None):
    """Drop-in for causal_conv1d_fn (the depthwise 4-tap conv of the linear-attention layers).
    transformers' fallback nn.Conv1d goes through MIOpen, which is slow here (390 ms forward at
    16k tokens). x arrives as a transposed view of a contiguous [B, T, C] tensor, so this shifts
    along T in that layout, channels contiguous: 29 ms (bench_linattn.py)."""
    xt = x.transpose(1, 2)
    k, t = weight.shape[1], xt.shape[1]
    xp = torch.nn.functional.pad(xt, (0, 0, k - 1, 0))
    w = weight.t()
    y = xp[:, k - 1:k - 1 + t] * w[k - 1]
    for i in range(k - 1):
        y = y + xp[:, i:i + t] * w[i]
    if bias is not None:
        y = y + bias
    y = torch.nn.functional.silu(y) if activation in ("silu", "swish") else y
    return y.transpose(1, 2)


def causal_sdpa(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None, **kwargs):
    """Full attention for single unpadded sequences (batch 1, as training runs): SDPA with
    is_causal and no mask tensor, so ROCm's flash kernel runs (head_dim 256 works here: 1.6 s
    fwd+bwd at 16k tokens, 1.3 GB). Transformers' sdpa path passed a mask and fell back to the
    math kernel, which needed 12.8 GB at 16k and ran out of memory."""
    assert query.shape[0] == 1, "causal_sdpa assumes one unpadded sequence"
    rep = query.shape[1] // key.shape[1]
    if rep > 1:
        key = key.repeat_interleave(rep, dim=1)
        value = value.repeat_interleave(rep, dim=1)
    out = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=query.shape[2] > 1,
                                                           scale=scaling)
    return out.transpose(1, 2).contiguous(), None


# ---------------------------------------------------------------- conversion

def text_key(k: str) -> str | None:
    if k.startswith(("model.visual.", "mtp.")):
        return None
    if k.startswith("model.language_model."):
        return "model." + k[len("model.language_model."):]
    return k  # lm_head.weight


def convert(src: Path, dst: Path) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    dst.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(index.values()))
    out_map, buf, size, n_out = {}, {}, 0, 0

    def flush():
        nonlocal buf, size, n_out
        if not buf:
            return
        name = f"model-q4-{n_out:05d}.safetensors"
        save_file(buf, str(dst / name), metadata={"format": "pt"})
        for k in buf:
            out_map[k] = name
        n_out += 1
        buf, size = {}, 0

    t0 = time.time()
    for sh in shards:
        with safe_open(str(src / sh), framework="pt") as f:
            for k in f.keys():
                nk = text_key(k)
                if nk is None:
                    continue
                t = f.get_tensor(k)
                if nk.endswith(("mlp.experts.gate_up_proj", "mlp.experts.down_proj")):
                    base = nk.rsplit(".", 1)[0] + "." + ("gate_up" if nk.endswith("gate_up_proj") else "down")
                    for part, v in quantize(t).items():
                        buf[f"{base}_{part}"] = v
                        size += v.numel() * v.element_size()
                else:
                    buf[nk] = t.contiguous()
                    size += t.numel() * t.element_size()
                del t
                if size >= SHARD_BYTES:
                    flush()
        print(f"  {sh} done ({time.time() - t0:.0f}s)", flush=True)
    flush()
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": out_map}, indent=1) + "\n")
    cfg = json.loads((src / "config.json").read_text())
    text = dict(cfg["text_config"])
    text["model_type"] = "qwen3_5_moe_text"
    text["architectures"] = ["Qwen3_5MoeForCausalLM"]
    text["tie_word_embeddings"] = False
    text["q4moe"] = {"group": GROUP, "format": "asymmetric uint4, scale+min bf16 per group, low nibble first",
                     "source": str(src)}
    (dst / "config.json").write_text(json.dumps(text, indent=1) + "\n")
    for f in ("LICENSE", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
              "generation_config.json"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    print(f"wrote {n_out} shards to {dst} in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- loading

def load(path: Path, device: str = "cuda", attn: str = "causal_sdpa"):
    """The text-only CausalLM with Q4Experts, filled from the converted shards onto `device`."""
    from safetensors import safe_open
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as mq

    from transformers import AttentionInterface
    AttentionInterface.register("causal_sdpa", causal_sdpa)
    config = Qwen3_5MoeTextConfig.from_pretrained(path)
    group = config.q4moe["group"] if hasattr(config, "q4moe") else GROUP
    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM._from_config(config, attn_implementation=attn, dtype=torch.bfloat16)
    for layer in model.model.layers:
        old = layer.mlp.experts
        layer.mlp.experts = Q4Experts(old.num_experts, old.hidden_dim, old.intermediate_dim, old.act_fn,
                                      device="meta", group=group)
    model.to_empty(device=device)
    want = {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}
    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    loaded = set()
    with torch.no_grad():
        for sh in sorted(set(index.values())):
            with safe_open(str(path / sh), framework="pt") as f:
                for k in f.keys():
                    if k not in params:
                        raise KeyError(f"unexpected tensor {k}")
                    t = f.get_tensor(k)
                    params[k].copy_(t)
                    loaded.add(k)
                    del t
    for layer in model.model.layers:
        if hasattr(layer, "linear_attn") and layer.linear_attn.causal_conv1d_fn is None:
            layer.linear_attn.causal_conv1d_fn = causal_conv1d_btc
    # Non-persistent buffers (rotary frequencies) aren't in the checkpoint: rebuild them.
    model.model.rotary_emb = mq.Qwen3_5MoeTextRotaryEmbedding(config=config, device=device)
    missing = {n for n in want - loaded if "rotary_emb" not in n}
    if missing:
        raise KeyError(f"{len(missing)} tensors not in checkpoint, e.g. {sorted(missing)[:5]}")
    model.eval()
    return model


# ---------------------------------------------------------------- check

def check(src: Path, dst: Path, layers=(0, 19, 39)) -> None:
    """Quantization error of the experts on a few layers, and of one MoE expert on random input."""
    from safetensors import safe_open

    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    qindex = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    for li in layers:
        for part, name in (("gate_up", "gate_up_proj"), ("down", "down_proj")):
            k = f"model.language_model.layers.{li}.mlp.experts.{name}"
            with safe_open(str(src / index[k]), framework="pt") as f:
                w = f.get_tensor(k)
            qk = f"model.layers.{li}.mlp.experts.{part}"
            parts = {}
            for p in "qsm":
                with safe_open(str(dst / qindex[f"{qk}_{p}"]), framework="pt") as f:
                    parts[p] = f.get_tensor(f"{qk}_{p}")
            wd = dequantize(parts["q"], parts["s"], parts["m"])
            rel = ((wd.float() - w.float()).norm() / w.float().norm()).item()
            x = torch.randn(64, w.shape[-1], dtype=torch.float32)
            y, yd = x @ w[3].float().T, x @ wd[3].float().T
            print(f"layer {li} {name}: weight rel err {rel:.4f}, expert 3 output rel err "
                  f"{((yd - y).norm() / y.norm()).item():.4f}")


if __name__ == "__main__":
    import memguard
    memguard.start(12)
    torch.set_num_threads(8)
    cmd, src, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    {"convert": convert, "check": check}[cmd](src, dst)
