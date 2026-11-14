"""Gated cross-attention memory for a frozen Qwen3 decoder.

Memory vectors stay outside the token sequence: they occupy no positions, no
self-attention KV cache and no RoPE slots.  Before selected decoder layers, a
new block lets every token attend to the memory bank and adds the result to
the residual stream.  Output projections start at zero, so at initialization
the wrapped model computes exactly what the stock model computes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryAdapter(nn.Module):
    """Encoder width -> decoder width; same shape as LCLM's released MLP bridge."""

    def __init__(self, enc_dim: int, dec_dim: int, max_slot: int):
        super().__init__()
        self.norm = nn.RMSNorm(enc_dim, eps=1e-6)
        self.fc1 = nn.Linear(enc_dim, dec_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dec_dim, dec_dim)
        self.slot = nn.Embedding(max_slot, dec_dim)
        nn.init.zeros_(self.slot.weight)

    def forward(self, memory, slots):
        return self.fc2(self.act(self.fc1(self.norm(memory)))) + self.slot(slots)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.norm_h = nn.RMSNorm(dim, eps=eps)
        self.norm_m = nn.RMSNorm(dim, eps=eps)
        self.q = nn.Linear(dim, heads * head_dim, bias=False)
        self.k = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.v = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)
        self.o = nn.Linear(heads * head_dim, dim, bias=False)
        nn.init.zeros_(self.o.weight)

    def keys_values(self, memory):
        b, m, _ = memory.shape
        x = self.norm_m(memory)
        k = self.k_norm(self.k(x).view(b, m, self.kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v(x).view(b, m, self.kv_heads, self.head_dim).transpose(1, 2)
        return k, v

    def forward(self, h, k, v, mask):
        b, t, _ = h.shape
        q = self.q_norm(self.q(self.norm_h(h)).view(b, t, self.heads, self.head_dim)).transpose(1, 2)
        out = F.scaled_dot_product_attention(q.to(k.dtype), k, v, attn_mask=mask, enable_gqa=True)
        return h + self.o(out.transpose(1, 2).reshape(b, t, -1).to(h.dtype)).to(h.dtype)


@dataclass
class MemoryState:
    """Per-forward memory; None disables every block (exactly the stock model)."""
    memory: torch.Tensor | None = None      # [B, M, dec_dim] after the adapter
    mask: torch.Tensor | None = None        # [B, 1, 1, M] bool, True = attend
    kv: dict = field(default_factory=dict)  # block index -> (k, v), reused across decode steps


class XLayer(nn.Module):
    def __init__(self, layer: nn.Module, block: CrossAttention, index: int, state: MemoryState):
        super().__init__()
        self.layer, self.index = layer, index
        # Lists keep the block and state out of the lm's module tree, so every
        # lm parameter stays frozen and the blocks are owned by MemoryModel.blocks.
        self._block, self._state = [block], [state]

    def forward(self, hidden_states, *args, **kwargs):
        s = self._state[0]
        if s.memory is not None:
            block = self._block[0]
            if self.index not in s.kv:
                s.kv[self.index] = block.keys_values(s.memory)
            k, v = s.kv[self.index]
            hidden_states = block(hidden_states, k, v, s.mask)
        return self.layer(hidden_states, *args, **kwargs)


class MemoryModel(nn.Module):
    """Wraps a Hugging Face Qwen3ForCausalLM. Only `self.memory_modules` train."""

    def __init__(self, lm, enc_dim: int, cfg: dict):
        super().__init__()
        self.lm = lm
        for p in lm.parameters():
            p.requires_grad_(False)
        dim = lm.config.hidden_size
        self.state = MemoryState()
        self.adapter = MemoryAdapter(enc_dim, dim, cfg["max_slot"])
        self.blocks = nn.ModuleList(
            CrossAttention(dim, cfg["heads"], cfg["kv_heads"], cfg["head_dim"], lm.config.rms_norm_eps)
            for _ in cfg["before_layers"])
        self.max_slot = cfg["max_slot"]
        layers = lm.model.layers
        for j, li in enumerate(cfg["before_layers"]):
            layers[li] = XLayer(layers[li], self.blocks[j], j, self.state)

    def trainable(self):
        return [p for n, p in self.named_parameters() if not n.startswith("lm.")]

    def trainable_state(self):
        return {n: p.detach().cpu() for n, p in self.named_parameters() if not n.startswith("lm.")}

    def load_trainable(self, sd):
        missing = self.load_state_dict(sd, strict=False)
        bad = [k for k in missing.missing_keys if not k.startswith("lm.")]
        if bad or missing.unexpected_keys:
            raise ValueError(f"Bad memory checkpoint: missing {bad}, unexpected {missing.unexpected_keys}")

    def set_memory(self, banks: list[torch.Tensor] | None, slots: list[torch.Tensor] | None = None):
        """banks: per-row [M_i, enc_dim] raw encoder vectors; None disables memory."""
        self.state.kv = {}
        if banks is None:
            self.state.memory, self.state.mask = None, None
            return
        dev = self.adapter.fc1.weight.device
        m = max(b.shape[0] for b in banks)
        mem = torch.zeros(len(banks), m, banks[0].shape[1], device=dev, dtype=self.adapter.fc1.weight.dtype)
        slot = torch.zeros(len(banks), m, device=dev, dtype=torch.long)
        mask = torch.zeros(len(banks), 1, 1, m, device=dev, dtype=torch.bool)
        for i, b in enumerate(banks):
            n = b.shape[0]
            if n == 0:
                raise ValueError("Empty memory bank; use banks=None for no memory")
            mem[i, :n] = b.to(dev, mem.dtype)
            slot[i, :n] = slots[i].to(dev) if slots is not None else torch.arange(n, device=dev) % self.max_slot
            mask[i, ..., :n] = True
        dtype = next(self.lm.parameters()).dtype
        self.state.memory = self.adapter(mem, slot).to(dtype)
        self.state.mask = mask

    def forward(self, *args, **kwargs):
        return self.lm(*args, **kwargs)


def init_adapter_from_lclm(model: MemoryModel, adapter_path):
    """Start the bridge from LCLM's released MLP weights (an initialization only)."""
    from safetensors.torch import load_file
    sd = load_file(adapter_path)
    mapped = {k: v for k, v in sd.items() if k.split(".")[0] in ("norm", "fc1", "fc2")}
    missing = model.adapter.load_state_dict(mapped, strict=False)
    if [k for k in missing.missing_keys if not k.startswith("slot")] or missing.unexpected_keys:
        raise ValueError(f"LCLM adapter mismatch: {missing}")
    return sorted(mapped)
