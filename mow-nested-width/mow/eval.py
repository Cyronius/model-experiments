"""Per-condition evaluation: loss, exact-match accuracy, FLOP accounting, and the
router diagnostics that figures 3-5 are built from.

Two different token populations are used on purpose:
  - FLOPs are averaged over every non-pad token, because every token costs
    compute whether or not it carries loss.
  - Routing histograms are reported over answer tokens, because that is where
    the task difficulty actually lives and what the interpretability claim is
    about. Prompt-token stats are reported alongside so the split is visible.
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import PAD
from .data import CELLS, MAX_ANSWER_LEN, decode_tokens, exact_match
from .model import Ctx, Transformer, MoWMLP, MoEMLP


def _marginal_entropy(counts: torch.Tensor) -> float:
    """Entropy of the choice histogram across tokens, in nats.

    This is the collapse signal, not per-token entropy: eval runs with hard
    selection, so every token's distribution is one-hot and per-token entropy is
    always 0. What "collapse" means is that the whole token population funnels
    into one choice, which is exactly what this measures. 0 == fully collapsed,
    log(n_choices) == perfectly spread.
    """
    p = counts / counts.sum().clamp_min(1)
    return float(-(p.clamp_min(1e-9).log() * p).sum())


def _tensors(batch, device):
    return (torch.from_numpy(batch.x).to(device),
            torch.from_numpy(batch.y).to(device),
            torch.from_numpy(batch.mask).to(device))


@torch.no_grad()
def masked_loss(model, batch, ctx: Ctx, device="cpu", chunk: int = 256):
    """Mean cross-entropy over answer tokens, plus mean MLP FLOPs fraction."""
    x, y, m = _tensors(batch, device)
    tot, ntok, frac_sum, nvalid = 0.0, 0.0, 0.0, 0.0
    for i in range(0, len(x), chunk):
        xb, yb, mb = x[i:i + chunk], y[i:i + chunk], m[i:i + chunk]
        logits, infos = model(xb, ctx)
        ls = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1),
                             reduction="none").view_as(yb)
        tot += (ls * mb).sum().item()
        ntok += mb.sum().item()
        valid = (xb != PAD).float()
        _, _, mean_frac = model.routing_losses(infos, xb != PAD, 0.0)
        frac_sum += mean_frac.item() * valid.sum().item()
        nvalid += valid.sum().item()
    return tot / max(ntok, 1), frac_sum / max(nvalid, 1)


@torch.no_grad()
def greedy_exact_match(model, batch, ctx: Ctx, block_size: int, device="cpu",
                       limit: Optional[int] = None):
    """Autoregressively decode the answer and compare strings exactly."""
    n = len(batch) if limit is None else min(limit, len(batch))
    x = torch.from_numpy(batch.x[:n]).clone().to(device)
    pl = torch.from_numpy(batch.prompt_len[:n]).to(device)
    rows = torch.arange(n, device=device)
    for i in range(n):                      # blank everything after the prompt
        x[i, pl[i]:] = PAD
    for t in range(MAX_ANSWER_LEN):
        pos = pl - 1 + t                    # logits here predict answer token t
        write = pos + 1
        if int(write.max()) >= block_size:
            break
        logits, _ = model(x, ctx)
        nxt = logits[rows, pos].argmax(-1)
        x[rows, write] = nxt
    ok = 0
    for i in range(n):
        pred = decode_tokens(x[i, pl[i]:].tolist())
        ok += exact_match(pred, batch.answers[i])
    return ok / max(n, 1)


@torch.no_grad()
def routing_stats(model, batch, ctx: Ctx, device="cpu"):
    """Per-layer routing histograms over answer vs prompt tokens, and by difficulty."""
    x, _, m = _tensors(batch, device)
    valid = (x != PAD)
    ans = m > 0
    prompt = valid & ~ans
    _, infos = model(x, ctx)

    n_layer = len(infos)
    out = {"choice_hist": [], "choice_hist_prompt": [], "choice_hist_all": [],
           "k_hist": [], "k_hist_all": [], "entropy": [], "k_entropy": []}
    per_tok_frac = torch.zeros_like(x, dtype=torch.float32)

    for info in infos:
        if info.probs is None:
            out["choice_hist"].append(None)
            per_tok_frac += info.fixed_frac
            continue
        C = info.probs.shape[-1]
        oh = F.one_hot(info.choice, C).float()
        hist_all = (oh * valid.unsqueeze(-1)).sum((0, 1))
        out["choice_hist"].append((oh * ans.unsqueeze(-1)).sum((0, 1)).tolist())
        out["choice_hist_prompt"].append((oh * prompt.unsqueeze(-1)).sum((0, 1)).tolist())
        out["choice_hist_all"].append(hist_all.tolist())
        out["entropy"].append(_marginal_entropy(hist_all))
        if info.k_probs is not None:
            K = info.k_probs.shape[-1]
            koh = F.one_hot(info.k_choice, K).float()
            k_all = (koh * valid.unsqueeze(-1)).sum((0, 1))
            out["k_hist"].append((koh * ans.unsqueeze(-1)).sum((0, 1)).tolist())
            out["k_hist_all"].append(k_all.tolist())
            out["k_entropy"].append(_marginal_entropy(k_all))
        if info.flops_frac is not None:
            per_tok_frac += info.flops_frac
        else:
            per_tok_frac += info.fixed_frac
    per_tok_frac /= n_layer

    # mean FLOPs fraction per example, restricted to answer tokens
    ans_f = ans.float()
    ex_frac = (per_tok_frac * ans_f).sum(1) / ans_f.sum(1).clamp_min(1)
    by_diff = defaultdict(list)
    for d, f in zip(batch.difficulty.tolist(), ex_frac.tolist()):
        by_diff[int(d)].append(f)
    out["by_difficulty"] = {d: {"mlp_frac": float(np.mean(v)), "n": len(v)}
                            for d, v in sorted(by_diff.items())}
    out["mlp_frac_answer"] = float(ex_frac.mean())
    return out


def tier_scales(model) -> Optional[list]:
    vals = []
    for blk in model.blocks:
        if isinstance(blk.mlp, MoWMLP) and blk.mlp.scale is not None:
            vals.append(blk.mlp.scale.detach().cpu().tolist())
    return vals or None


def is_routed(model) -> bool:
    return any(getattr(b.mlp, "router", None) is not None for b in model.blocks)


@torch.no_grad()
def evaluate(model, eval_sets, ctx: Ctx, cfg, device="cpu", do_exact=True,
             do_routing=True) -> dict:
    """Full per-condition report. Everything here is JSON-serializable."""
    model.eval()
    res = {"per_cell": {}}
    losses, fracs, accs = [], [], []
    for cell in CELLS:
        b = eval_sets[cell]
        loss, frac = masked_loss(model, b, ctx, device)
        entry = {"loss": loss, "mlp_frac": frac}
        if do_exact:
            entry["acc"] = greedy_exact_match(model, b, ctx, cfg.model.block_size,
                                              device, limit=cfg.data.n_exact_per_cell)
            accs.append(entry["acc"])
        if do_routing and is_routed(model):
            entry.update(routing_stats(model, b, ctx, device))
        res["per_cell"][cell] = entry
        losses.append(loss); fracs.append(frac)

    res["loss"] = float(np.mean(losses))
    res["mlp_frac"] = float(np.mean(fracs))
    if accs:
        res["acc"] = float(np.mean(accs))
    res["flops"] = model.flops_per_token(res["mlp_frac"])
    res["tier_scales"] = tier_scales(model)

    if do_routing and is_routed(model):
        ents = [e for c in CELLS for e in res["per_cell"][c].get("entropy", [])]
        res["entropy"] = float(np.mean(ents)) if ents else None
        kents = [e for c in CELLS for e in res["per_cell"][c].get("k_entropy", [])]
        res["k_entropy"] = float(np.mean(kents)) if kents else None
        res["collapsed"] = bool(res["entropy"] is not None and
                                res["entropy"] < cfg.train.entropy_collapse_thresh)
    model.train()
    return res


def sweep_fixed_tiers(model, eval_sets, cfg, device="cpu") -> dict:
    """Figure 1: evaluate one nested model at each fixed tier (router bypassed)."""
    out = {}
    for t, hidden in enumerate(cfg.model.tiers):
        ctx = Ctx(tiers=[t] * cfg.model.n_layer)
        r = evaluate(model, eval_sets, ctx, cfg, device, do_exact=True, do_routing=False)
        r["hidden"] = hidden
        r["mlp_frac"] = hidden / cfg.model.d_ff
        r["flops"] = model.flops_per_token(r["mlp_frac"])
        out[str(hidden)] = r
    return out


# ------------------------------------------------- cross-tier KV reuse (phase 0)
# Deployment scenario: the context was processed at tier A, so its KV cache holds
# projections of tier-A residual streams; the router switches to tier B for the
# answer, and every generated token attends to the stale tier-A entries. Because
# the model is causal and only the MLP is tiered, one forward pass whose MLP runs
# tier A at positions < split and tier B at positions >= split reproduces that
# cached computation exactly -- no incremental-decoding implementation needed.

@torch.no_grad()
def mixed_tier_logits(model, x, tier_a: int, tier_b: int, split: torch.Tensor):
    """Forward with a per-example tier boundary. split[i] = first position of
    example i computed at tier B (everything earlier is tier A)."""
    B, T = x.shape
    h = model.tok(x)
    if model.pos is not None:
        h = h + model.pos(torch.arange(T, device=x.device))
    early = (torch.arange(T, device=x.device)[None, :] < split[:, None]).unsqueeze(-1)
    for blk in model.blocks:
        h = h + blk.attn(blk.ln1(h))
        u = blk.ln2(h)
        if tier_a == tier_b:
            h = h + blk.mlp.forward_tier(u, tier_a)
        else:
            h = h + torch.where(early, blk.mlp.forward_tier(u, tier_a),
                                blk.mlp.forward_tier(u, tier_b))
    return model.head(model.ln_f(h))


def _switch_split(batch, refresh_k: int, device):
    """Boundary for a tier switch at the prompt/answer seam. refresh_k >= 1 is the
    number of trailing prompt tokens recomputed at the new tier; 1 is the natural
    minimum, since generating at tier B already forwards the final prompt token."""
    pl = torch.from_numpy(batch.prompt_len).to(device)
    return (pl - refresh_k).clamp_min(1)


@torch.no_grad()
def kv_switch_loss(model, batch, tier_a, tier_b, device="cpu", refresh_k=1,
                   chunk: int = 256):
    """Answer-token loss when the prompt's KV comes from tier A but the answer is
    computed at tier B."""
    x, y, m = _tensors(batch, device)
    split = _switch_split(batch, refresh_k, device)
    tot, ntok = 0.0, 0.0
    for i in range(0, len(x), chunk):
        logits = mixed_tier_logits(model, x[i:i + chunk], tier_a, tier_b,
                                   split[i:i + chunk])
        ls = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                             y[i:i + chunk].reshape(-1), reduction="none"
                             ).view_as(y[i:i + chunk])
        tot += (ls * m[i:i + chunk]).sum().item()
        ntok += m[i:i + chunk].sum().item()
    return tot / max(ntok, 1)


@torch.no_grad()
def kv_switch_exact(model, batch, tier_a, tier_b, block_size, device="cpu",
                    refresh_k=1, limit: Optional[int] = None):
    """Greedy exact-match under the same stale-KV regime. Generated tokens run at
    tier B and attend to tier-A projections of the prompt."""
    n = len(batch) if limit is None else min(limit, len(batch))
    x = torch.from_numpy(batch.x[:n]).clone().to(device)
    pl = torch.from_numpy(batch.prompt_len[:n]).to(device)
    split = (pl - refresh_k).clamp_min(1)
    rows = torch.arange(n, device=device)
    for i in range(n):
        x[i, pl[i]:] = PAD
    for t in range(MAX_ANSWER_LEN):
        pos = pl - 1 + t
        write = pos + 1
        if int(write.max()) >= block_size:
            break
        logits = mixed_tier_logits(model, x, tier_a, tier_b, split)
        x[rows, write] = logits[rows, pos].argmax(-1)
    ok = 0
    for i in range(n):
        ok += exact_match(decode_tokens(x[i, pl[i]:].tolist()), batch.answers[i])
    return ok / max(n, 1)
