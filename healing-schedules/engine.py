"""Qwen3.5 block reconstruction, full-student evaluation, and operator FLOP ledger.

The FLOP convention is PyTorch FlopCounterMode's registered dense operators
(matmul/attention/convolution forward and backward). This is an estimate, not a
hardware instruction count; elementwise, normalization, and optimizer work are
excluded and the convention must accompany every comparison.
"""
from contextlib import nullcontext
import math

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask


def precision(model, enabled=True):
    device = next(model.parameters()).device.type
    return torch.autocast(device, dtype=torch.bfloat16) if enabled else nullcontext()


def context(model, ids):
    inner = model.model
    h = inner.embed_tokens(ids)
    position = torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1).expand(4, ids.shape[0], -1)
    kw = dict(config=inner.config, inputs_embeds=h, attention_mask=None,
              past_key_values=None, position_ids=position[0])
    masks = {'full_attention': create_causal_mask(**kw),
             'linear_attention': create_recurrent_attention_mask(**kw)}
    return h, dict(position_embeddings=inner.rotary_emb(h, position[1:]),
                   position_ids=position[0], masks=masks)


def layer(model, i, h, ctx):
    return model.model.layers[i](h, position_embeddings=ctx['position_embeddings'],
                                position_ids=ctx['position_ids'],
                                attention_mask=ctx['masks'][model.config.layer_types[i]],
                                past_key_values=None, use_cache=False)


def stream(model, ids):
    h, ctx = context(model, ids)
    outputs = []
    for i in range(model.config.num_hidden_layers):
        h = layer(model, i, h, ctx)
        outputs.append(h)
    return outputs


def discrepancy(student, teacher):
    # Raw residuals, before final RMSNorm, with a per-token teacher-energy denominator.
    return ((student.float() - teacher.float()).square().sum(-1) /
            teacher.float().square().sum(-1).clamp_min(1e-8)).mean()


def logit_loss(student, teacher, hs, ht, ids, chunk=16):
    hs = student.model.norm(hs)[:, :-1].reshape(-1, hs.shape[-1])
    with torch.no_grad():
        ht = teacher.model.norm(ht)[:, :-1].reshape(-1, ht.shape[-1])
    labels = ids[:, 1:].reshape(-1)
    ce = hs.new_zeros((), dtype=torch.float32)
    kl = ce.clone()
    # Full vocabulary teacher||student KL at T=1; never top-k-renormalized.
    for start in range(0, len(labels), chunk):
        stop = min(start + chunk, len(labels))
        sl = student.lm_head(hs[start:stop]).float()
        with torch.no_grad():
            tl = teacher.lm_head(ht[start:stop]).float().log_softmax(-1)
        ce = ce + F.cross_entropy(sl, labels[start:stop], reduction='sum')
        kl = kl + F.kl_div(sl.log_softmax(-1), tl, log_target=True, reduction='sum')
    return ce / len(labels), kl / len(labels)


def prepare_student(model):
    """Fixed trainable set for every strategy, FP32 masters: all decoder layers, the tied
    embedding/output head, and the final norm (v2; v1 froze embedding, head and final norm).

    Which of them a step updates depends on its window (`set_window`).
    """
    model.requires_grad_(False)
    for block in model.model.layers:
        block.float()
    model.model.embed_tokens.float()
    model.model.norm.float()
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise ValueError('Expected a tied embedding/output head')
    for _, params in parameter_groups(model):
        for p in params:
            p.requires_grad_(True)
    model.eval()  # no dropout; gradients remain enabled for active windows


def parameter_groups(model):
    """Optimizer groups; warmup counts updates per group."""
    return ([('embedding', [model.model.embed_tokens.weight]), ('final_norm', list(model.model.norm.parameters()))] +
            [(f'layer{i}', list(block.parameters())) for i, block in enumerate(model.model.layers)])


def set_window(student, window):
    """Layers [a, b) train. The tied embedding/head trains when the window touches either end of
    the stack (a == 0 reads the embedding, b == depth uses the head); the final norm when b == depth."""
    a, b = window
    depth = student.config.num_hidden_layers
    for i, block in enumerate(student.model.layers):
        block.requires_grad_(a <= i < b)
    student.model.embed_tokens.weight.requires_grad_(a == 0 or b == depth)
    student.model.norm.requires_grad_(b == depth)


def training_loss(student, teacher, ids, residual_indices, window, ce_weight=0.1, amp=True,
                  teacher_input=False):
    """Window loss. By default the window reads the current STUDENT prefix (layers [0, a) run
    under no_grad). teacher_input=True (follow-up 3) feeds it the teacher's residual stream after
    layer a - 1 instead, restricted to the student's residual coordinates (the same restriction
    the discrepancy target uses), and skips the student prefix; a == 0 still reads the student's
    own embedding. Evaluation always runs the student's own full stream."""
    a, b = window
    set_window(student, window)
    with precision(student, amp):
        with torch.no_grad():
            target = stream(teacher, ids)
        with torch.set_grad_enabled(a == 0):
            h, ctx = context(student, ids)
        if teacher_input and a > 0:
            h = target[a - 1][..., residual_indices].float()
        else:
            with torch.no_grad():
                for i in range(a):
                    h = layer(student, i, h, ctx)
        if not h.requires_grad:
            h = h.detach().requires_grad_(True)
        for i in range(a, b):
            h = layer(student, i, h, ctx)
        loss = discrepancy(h, target[b - 1][..., residual_indices])
        if b == student.config.num_hidden_layers:
            ce, kl = logit_loss(student, teacher, h, target[-1], ids)
            loss = loss + kl + ce_weight * ce
    return loss


@torch.no_grad()
def stream_errors(student, teacher, batches, residual_indices, amp=True):
    """Per-depth discrepancy only (no vocabulary head): the adaptive controllers' signal."""
    errors, total = None, 0
    for ids in batches:
        with precision(student, amp):
            s, t = stream(student, ids), stream(teacher, ids)
            n = ids.numel()
            values = [float(discrepancy(x, y[..., residual_indices])) * n for x, y in zip(s, t)]
        errors = values if errors is None else [e + v for e, v in zip(errors, values)]
        total += n
    if not total:
        raise ValueError('Empty controller batch')
    return [e / total for e in errors]


@torch.no_grad()
def evaluate(student, teacher, batches, residual_indices, amp=True):
    student.eval()
    teacher.eval()
    total, ce_sum, kl_sum = 0, 0.0, 0.0
    depth = student.config.num_hidden_layers
    errors = [0.0] * depth
    for ids in batches:
        with precision(student, amp):
            s, t = stream(student, ids), stream(teacher, ids)
            ce, kl = logit_loss(student, teacher, s[-1], t[-1], ids)
            n = ids[:, 1:].numel()
            ce_sum += float(ce) * n
            kl_sum += float(kl) * n
            total += n
            for i in range(depth):
                errors[i] += float(discrepancy(s[i], t[i][..., residual_indices])) * n
    if not total:
        raise ValueError('Empty evaluation split')
    ce = ce_sum / total
    return dict(ce=ce, perplexity=math.exp(min(ce, 80)), teacher_kl=kl_sum / total,
                discrepancy_by_depth=[x / total for x in errors], target_tokens=total)


def counted(fn):
    with FlopCounterMode(display=False) as counter:
        result = fn()
    return result, counter.get_total_flops()


def profile_costs(student, teacher, ids, amp=True):
    """Profile each component once; no optimizer steps or weight changes.

    Input gradients are always computed for active layers, including the first,
    so their backward costs compose exactly under the registered-op convention.
    The real loss uses no extra counted operators except the final vocabulary head.
    """
    student.zero_grad(set_to_none=True)
    # The head cost below includes the tied head's weight gradient (it trains whenever b == depth).
    student.model.embed_tokens.weight.requires_grad_(True)
    student.model.norm.requires_grad_(True)
    forward, backward, teacher_fwd = [], [], []
    with precision(student, amp):
        with torch.no_grad():
            (th, tc), teacher_context = counted(lambda: context(teacher, ids))
            (h, sc), student_context = counted(lambda: context(student, ids))
        for i in range(student.config.num_hidden_layers):
            with torch.no_grad():
                th, cost = counted(lambda: layer(teacher, i, th, tc))
                teacher_fwd.append(cost)
            student.model.layers[i].requires_grad_(True)
            h = h.detach().requires_grad_(True)
            out, f = counted(lambda: layer(student, i, h, sc))
            _, back = counted(lambda: out.sum().backward())
            forward.append(f)
            backward.append(back)
            student.zero_grad(set_to_none=True)
            h = out.detach()
        h = h.detach().requires_grad_(True)

        def head_cost():
            ce, kl = logit_loss(student, teacher, h, th.detach(), ids)
            (ce + kl).backward()

        _, head = counted(head_cost)
    student.zero_grad(set_to_none=True)
    return dict(student_forward=forward, student_backward=backward,
                teacher_forward=teacher_fwd, final_heads_and_loss=head,
                context_flops=teacher_context + student_context,
                profiling_flops=sum(forward) + sum(backward) + sum(teacher_fwd) + head + teacher_context + student_context,
                convention='torch registered matmul/attention/convolution FLOPs; excludes elementwise/norm/optimizer')


def step_cost(profile, window, teacher_input=False):
    a, b = window
    depth = len(profile['student_forward'])
    prefix = a if teacher_input else 0  # a teacher-fed window skips the student prefix forward
    return (profile['context_flops'] + sum(profile['teacher_forward']) + sum(profile['student_forward'][prefix:b]) +
            sum(profile['student_backward'][a:b]) +
            (profile['final_heads_and_loss'] if b == depth else 0))
