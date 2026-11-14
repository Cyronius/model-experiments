"""Reviewed building blocks for the next retention experiment; no run entry point.

The paired design is in PAIR_PLAN_20261026.md. These helpers do not load a
source model, select data, run training, or claim that a protocol is frozen.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def expand_lora(model, rank=64, seed=26102691):
    """Expand trained LoRA factors without changing B @ A or alpha/rank.

    Call BEFORE constructing a fresh optimizer; existing optimizer parameter
    references and moments are deliberately not transferred. Actual-model
    logit/cache/generation equivalence still needs its separate numerical gate.
    """
    from correctness_heal import LowRankLinear

    modules = [(n, m) for n, m in model.named_modules() if isinstance(m, LowRankLinear)]
    if not modules:
        raise ValueError("No LoRA modules to expand")
    if not isinstance(rank, int) or isinstance(rank, bool):
        raise ValueError("Rank must be an integer")
    if len({(m.rank, m.alpha) for _, m in modules}) != 1:
        raise ValueError("The existing adapter loader requires uniform rank and alpha")
    # Validate every module before mutating any of them.
    for name, module in modules:
        if rank <= module.rank or module.rank < 1:
            raise ValueError(f"Expansion must increase rank: {name}")
        if module.a.shape != (module.rank, module.base.in_features) or module.b.shape != (
            module.base.out_features, module.rank
        ):
            raise ValueError(f"Invalid adapter shapes: {name}")
        if not math.isfinite(module.alpha) or not all(
            torch.isfinite(p).all() for p in (module.a, module.b)
        ):
            raise ValueError(f"Nonfinite adapter: {name}")
    for index, (_, module) in enumerate(modules):
        old_rank = module.rank
        generator = torch.Generator(device=module.a.device).manual_seed(seed + index)
        a = torch.empty(rank, module.base.in_features, device=module.a.device, dtype=module.a.dtype)
        a.uniform_(-1 / math.sqrt(module.base.in_features),
                   1 / math.sqrt(module.base.in_features), generator=generator)
        b = module.b.new_zeros(module.base.out_features, rank)
        a[:old_rank].copy_(module.a)
        b[:, :old_rank].copy_(module.b)
        module.a, module.b = nn.Parameter(a), nn.Parameter(b)
        module.alpha *= rank / old_rank
        module.rank = rank
    return {"rank": rank, "alpha": modules[0][1].alpha,
            "modules": [n for n, _ in modules], "seed": seed,
            "optimizer": "create fresh after expansion",
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def _logits(logits, temperature):
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("Expected a nonempty [tokens, vocabulary] tensor")
    if not math.isfinite(temperature) or temperature <= 0 or not torch.isfinite(logits).all():
        raise ValueError("Finite logits and a positive temperature are required")
    # Keep float64 for independent numerical tests, use FP32 for BF16 models.
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    scaled = logits.to(dtype) / temperature
    if not torch.isfinite(scaled).all():
        raise ValueError("Temperature scaling overflowed")
    return scaled


def replay_token_keys(row_id, input_ids, positions):
    """Bind each cached prediction to its row, complete encoded text and position."""
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("A nonempty row ID is required")
    if len(input_ids) < 2 or any(type(i) is not int or i < 0 for i in input_ids):
        raise ValueError("Input IDs must be a nonempty integer sequence")
    if (not positions or len(set(positions)) != len(positions)
            or any(type(p) is not int or not 0 <= p < len(input_ids) - 1 for p in positions)):
        raise ValueError("Prediction positions must be distinct and have next-token targets")
    digest = hashlib.sha256(json.dumps(input_ids, separators=(',', ':')).encode()).hexdigest()
    return [f"{row_id}:{digest}:{p}" for p in positions]


def _token_keys(keys, count):
    if (keys is None or len(keys) != count
            or any(not isinstance(k, str) or not k for k in keys)):
        raise ValueError("One token identity is required per cached prediction")
    return tuple(keys)


@torch.no_grad()
def teacher_topk_tail(logits, k, temperature=1.0, token_keys=None):
    """Cache absolute top-k probabilities AND the aggregate omitted mass.

    Call on bounded token chunks. A top-k-plus-tail KL is a coarsened lower
    bound on full KL, not a substitute for measuring its approximation error.
    """
    scaled = _logits(logits, temperature)
    vocab = scaled.shape[-1]
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= vocab:
        raise ValueError("k must be an integer between 1 and vocabulary size")
    log_z = scaled.logsumexp(-1, keepdim=True)
    values, indices = scaled.topk(k, dim=-1)
    log_probs = values - log_z
    if k < vocab:
        # Avoid 1 - sum(top-k): rounding can erase a small, nonzero tail.
        tail = scaled.scatter(-1, indices, -torch.inf).logsumexp(-1, keepdim=True) - log_z
        log_probs = torch.cat((log_probs, tail), dim=-1)
    result = {"indices": indices.detach(), "log_probs": log_probs.detach(),
              "vocab_size": vocab, "temperature": float(temperature)}
    if token_keys is not None:
        result["token_keys"] = _token_keys(token_keys, scaled.shape[0])
    return result


def coarsened_forward_kl(student_logits, cache):
    """Per-token T² KL(teacher || student) on top-k IDs plus one tail bucket."""
    scaled = _logits(student_logits, cache["temperature"])
    indices = cache["indices"].to(scaled.device)
    teacher = cache["log_probs"].to(device=scaled.device, dtype=scaled.dtype).detach()
    vocab = scaled.shape[-1]
    if (vocab != cache["vocab_size"] or indices.ndim != 2
            or indices.shape[0] != scaled.shape[0] or indices.dtype != torch.int64
            or not 1 <= indices.shape[1] <= vocab):
        raise ValueError("Teacher cache dimensions or vocabulary differ")
    k = indices.shape[1]
    if ((indices < 0).any() or (indices >= vocab).any()
            or (k > 1 and (indices.sort(-1).values.diff(dim=-1) == 0).any())):
        raise ValueError("Teacher IDs must be distinct and within the vocabulary")
    if teacher.shape != (scaled.shape[0], k + int(k < vocab)) or not torch.isfinite(teacher).all():
        raise ValueError("Invalid cached probabilities")
    if not torch.allclose(teacher.logsumexp(-1), teacher.new_zeros(scaled.shape[0]), atol=2e-5, rtol=0):
        raise ValueError("Cached probabilities including the tail must sum to one")
    log_z = scaled.logsumexp(-1, keepdim=True)
    student = scaled.gather(-1, indices) - log_z
    if k < vocab:
        tail = scaled.scatter(-1, indices, -torch.inf).logsumexp(-1, keepdim=True) - log_z
        student = torch.cat((student, tail), dim=-1)
    return (teacher.exp() * (teacher - student)).sum(-1) * cache["temperature"] ** 2


def retention_objective(student_logits, targets, replay_mask, cache=None, kd_weight=1.0,
                        replay_keys=None):
    """Return token-summed SFT + replay-only KL and the shared denominator.

    cache covers exactly the rows selected by replay_mask, in the same order.
    replay_keys must come from replay_token_keys for the current encoded rows;
    the cache must carry identical keys. A model runner must gather answer
    hidden states and apply lm_head in bounded token chunks, not materialize
    [all update tokens, vocabulary] logits at once.
    Divide the summed loss by ALL answer tokens in the full optimizer update,
    not by each microbatch's size or by its replay-token count. Thus lambda=1
    adds one KL contribution per replay token without overweighting rare replay.
    Pass cache=None, kd_weight=0 for the SFT control. Dataset assembly must
    establish replay verification; this numerical helper cannot verify text.
    """
    _logits(student_logits, 1.0)
    count = student_logits.shape[0]
    if targets.shape != (count,) or targets.dtype != torch.int64:
        raise ValueError("One integer target is required per answer token")
    if replay_mask.shape != (count,) or replay_mask.dtype != torch.bool:
        raise ValueError("One Boolean replay flag is required per answer token")
    if not math.isfinite(kd_weight) or kd_weight < 0:
        raise ValueError("Invalid KL weight")
    if targets.device != student_logits.device or replay_mask.device != student_logits.device:
        raise ValueError("Logits, targets and replay flags must share a device")
    ce = F.cross_entropy(student_logits.float(), targets, reduction="sum")
    kd = student_logits.new_zeros((), dtype=torch.float32)
    replay_count = int(replay_mask.sum())
    if kd_weight and replay_count:
        if cache is None:
            raise ValueError("Missing teacher cache for replay tokens")
        current_keys = _token_keys(replay_keys, replay_count)
        cached_keys = _token_keys(cache.get("token_keys"), replay_count)
        if current_keys != cached_keys:
            raise ValueError("Teacher cache token identities or order differ")
        kd = coarsened_forward_kl(student_logits[replay_mask], cache).sum()
    return {"loss_sum": ce + kd_weight * kd, "ce_sum": ce, "kl_sum": kd,
            "answer_tokens": count, "replay_tokens": replay_count}


@dataclass(frozen=True)
class SelectionPolicy:
    min_primary_gain: float
    max_retention_loss_fraction: float
    max_program_net_loss: int = 0
    max_science_net_loss: int = 0
    max_new_loops: int = 0
    max_lost_completions: int = 1

    def __post_init__(self):
        for value in (self.min_primary_gain, self.max_retention_loss_fraction):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Fraction thresholds must be finite and in [0, 1]")
        for value in (self.max_program_net_loss, self.max_science_net_loss,
                      self.max_new_loops, self.max_lost_completions):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("Count thresholds must be nonnegative integers")


def _cases(rows):
    result = {}
    for row in rows:
        if row["id"] in result:
            raise ValueError("Duplicate case ID")
        if row["category"] not in {"primary", "retention", "program", "science"}:
            raise ValueError("Unknown case category")
        if any(type(row[k]) is not bool for k in ("correct", "loop", "ended")):
            raise ValueError("Correctness, loop and ended require Boolean judgments")
        if any(not isinstance(row[k], str) or not row[k] for k in ("id", "family", "group")):
            raise ValueError("Nonempty IDs, families and split groups are required")
        result[row["id"]] = row
    return result


def compare_cases(baseline, candidate, policy):
    """Development-only eligibility; missing science/code judgments fail closed.

    This does not select the winning checkpoint or make significance claims.
    Parent remains the default if no eligible candidate improves development.
    Report paired, group-aware uncertainty separately from these practical gates.
    """
    before, after = _cases(baseline), _cases(candidate)
    if before.keys() != after.keys():
        raise ValueError("Paired evaluation must contain identical case IDs")
    groups, families = {}, {}
    for id_, old in before.items():
        new = after[id_]
        if any(old[k] != new[k] for k in ("category", "family", "group")):
            raise ValueError("Case identity changed between arms")
        for table, key in ((groups, old["category"]), (families, old["category"] + ":" + old["family"])):
            entry = table.setdefault(key, {"n": 0, "before": 0, "after": 0, "gained": 0, "lost": 0})
            entry["n"] += 1
            entry["before"] += old["correct"]
            entry["after"] += new["correct"]
            entry["gained"] += not old["correct"] and new["correct"]
            entry["lost"] += old["correct"] and not new["correct"]
    if set(groups) != {"primary", "retention", "program", "science"}:
        raise ValueError("All four evaluation categories must be scored before selection")
    reasons = []
    primary = groups["primary"]
    if (primary["after"] - primary["before"]) / primary["n"] < policy.min_primary_gain:
        reasons.append("primary gain below frozen minimum")
    retention = groups["retention"]
    if retention["before"] == 0:
        reasons.append("retention baseline has no correct cases")
    elif retention["lost"] / retention["before"] > policy.max_retention_loss_fraction:
        reasons.append("too many previously correct retention answers lost")
    for category, limit in (("program", policy.max_program_net_loss), ("science", policy.max_science_net_loss)):
        # Prevent a gain in one code/science family hiding a loss in another.
        for family, entry in families.items():
            if family.startswith(category + ":") and entry["before"] - entry["after"] > limit:
                reasons.append(f"{family} correctness regressed")
    extra_loops = sum(r["loop"] for r in after.values()) - sum(r["loop"] for r in before.values())
    new_loops = sum(not r["loop"] and after[id_]["loop"] for id_, r in before.items())
    # Count individual lost completions, not a net count masked by new endings.
    lost_completions = sum(r["ended"] and not after[id_]["ended"] for id_, r in before.items())
    if new_loops > policy.max_new_loops:
        reasons.append("too many new loop cases")
    if lost_completions > policy.max_lost_completions:
        reasons.append("too many previously completed answers lost")
    return {"eligible": not reasons, "reasons": reasons, "categories": groups,
            "families": families, "extra_loops": extra_loops, "new_loops": new_loops,
            "lost_completions": lost_completions}


def require_frozen_protocol(protocol):
    """Readiness check for a future runner; the shipped design must fail it.

    Presence is only a design check. A runner must independently hash the
    referenced files, validate disjoint splits and obtain its Dibs lease.
    """
    if protocol.get("status") != "frozen":
        raise ValueError("Protocol is design-only; no experiment is ready to run")
    required = {"data_manifest", "source_manifest", "code_manifest", "numerical_gates",
                "selection_policy", "paired_comparison", "runtime_bounds", "peer_review"}
    freeze = protocol.get("freeze", {})
    missing = [key for key in sorted(required) if not freeze.get(key)]
    if missing:
        raise ValueError("Missing frozen run requirements: " + ", ".join(missing))
    SelectionPolicy(**freeze["selection_policy"])
    return True
