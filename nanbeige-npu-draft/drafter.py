"""Tiny target-hidden-conditioned parallel drafter.

The GPU supplies its normal first token and the final hidden vector at the
anchor. The drafter predicts six following tokens in one block. During
training, a random subset of true future tokens is revealed to teach a
second refinement pass; position 0 is always the GPU-supplied seed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    vocab: int = 166_912
    hidden: int = 7_168
    rank: int = 16
    width: int = 96
    block: int = 7  # one seed + six drafted tokens
    layers: int = 2
    heads: int = 4


def fake_q4(table: torch.Tensor) -> torch.Tensor:
    """Q4 forward values with a straight-through gradient to the FP master."""
    scale = (table.detach().abs().amax(dim=-1, keepdim=True) / 7).clamp_min(
        torch.finfo(torch.float16).tiny).half().float()
    quant = (table / scale).round().clamp(-8, 7) * scale
    return table + (quant - table).detach()


class Drafter(nn.Module):
    def __init__(self, config: Config = Config()):
        super().__init__()
        self.config = config
        self.factors = nn.Parameter(torch.empty(config.vocab, config.rank))
        nn.init.normal_(self.factors, std=config.rank**-0.5)
        self.condition_norm = nn.LayerNorm(config.hidden)
        self.condition = nn.Linear(config.hidden, config.width, bias=False)
        self.token_up = nn.Linear(config.rank, config.width, bias=False)
        self.token_down = nn.Linear(config.width, config.rank, bias=False)
        self.mask = nn.Parameter(torch.zeros(config.width))
        self.position = nn.Parameter(torch.randn(config.block, config.width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            config.width, config.heads, dim_feedforward=2 * config.width,
            dropout=0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.body = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(config.width)

    def forward(self, hidden: torch.Tensor, tokens: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        c = self.config
        if tokens.shape != (hidden.shape[0], c.block) or visible.shape != tokens.shape:
            raise ValueError("tokens and visible must be [batch, block]")
        if not bool(torch.all(visible[:, 0])):
            raise ValueError("the GPU seed token must be visible")
        table = fake_q4(self.factors)
        x = self.token_up(F.embedding(tokens, table))
        x = torch.where(visible[..., None], x, self.mask)
        x = x + self.position + self.condition(self.condition_norm(hidden))[:, None, :]
        x = self.body(x)
        logits = F.linear(self.token_down(self.out_norm(x[:, 1:])), table)
        return logits

    @torch.no_grad()
    def draft(self, hidden: torch.Tensor, seed: torch.Tensor, passes: int = 1) -> torch.Tensor:
        if passes < 1:
            raise ValueError("passes must be positive")
        tokens = torch.zeros((len(seed), self.config.block), dtype=torch.long, device=seed.device)
        tokens[:, 0] = seed
        visible = torch.zeros_like(tokens, dtype=torch.bool)
        visible[:, 0] = True
        for step in range(passes):
            logits = self(hidden, tokens, visible)
            top = logits.topk(2, dim=-1)
            tokens[:, 1:] = top.indices[..., 0]
            if step + 1 < passes:
                # Reveal two confident guesses to the refinement pass; the
                # remaining positions stay masked, as in training.
                margin = top.values[..., 0] - top.values[..., 1]
                chosen = margin[:, :-1].topk(2, dim=-1).indices
                visible[:, 1:] = False
                visible[:, 1:].scatter_(1, chosen, True)
        return tokens

    def manifest(self) -> dict:
        return {"config": asdict(self.config), "parameters": sum(p.numel() for p in self.parameters())}
