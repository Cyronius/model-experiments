"""Block-level comparison of a (healed) student with its teacher, layer by layer.

Joint healing trains only the final residual, the final-logit KL and CE, so the intermediate layers
are free to drift. For each layer i this measures, on the frozen validation sequences:

Stream (the student's own forward pass against the teacher's, at the student's residual coordinates):
  rel_err      ||s_i - t_i||^2 / ||t_i||^2 per token (the training curves' discrepancy_by_depth)
  cosine       cos(s_i, t_i) per token: direction only
  norm_ratio   ||s_i|| / ||t_i||: scale drift, which rel_err counts but the next RMSNorm mostly ignores
  lens_kl      KL(teacher lens || student lens), each layer read through its own model's final norm
               and output head (logit lens), full vocabulary, natural log, per token
Block (teacher-fed: student block i runs on the teacher's stream entering layer i, so upstream drift
is removed and only that block's own function is compared):
  block_rel_err   output vs teacher layer i output
  delta_cosine    cos(student block update, teacher block update), update = output - input
  delta_rel_err   ||du_s - du_t||^2 / ||du_t||^2
All are token means. Layer 0's teacher-fed input is the teacher's embedding at the student coordinates.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from canary import CanaryFailure, check as canary_check
from engine import context, layer, precision, stream
from pruning import sha256
from run import load_model, save_json

STREAM = ('rel_err', 'cosine', 'norm_ratio', 'lens_kl')
BLOCK = ('block_rel_err', 'delta_cosine', 'delta_rel_err')


def rel(a, b):
    return (a - b).square().sum(-1) / b.square().sum(-1).clamp_min(1e-8)


@torch.no_grad()
def lens_kl(student, teacher, hs, ht, chunk=256):
    hs = student.model.norm(hs).reshape(-1, hs.shape[-1])
    ht = teacher.model.norm(ht).reshape(-1, ht.shape[-1])
    total = 0.0
    for a in range(0, len(hs), chunk):
        sl = student.lm_head(hs[a:a + chunk]).float().log_softmax(-1)
        tl = teacher.lm_head(ht[a:a + chunk]).float().log_softmax(-1)
        total += float(F.kl_div(sl, tl, log_target=True, reduction='sum'))
    return total


@torch.no_grad()
def probe(student, teacher, rows, idx, device, amp):
    depth = student.config.num_hidden_layers
    sums = {k: [0.0] * depth for k in STREAM + BLOCK}
    tokens = 0
    for row in rows:
        ids = torch.tensor(row[None], device=device, dtype=torch.long)
        n = ids.numel()
        with precision(student, amp):
            s = stream(student, ids)
            t_full = stream(teacher, ids)
            t = [x[..., idx] for x in t_full]
            t_embed = teacher.model.embed_tokens(ids)[..., idx]
            _, ctx = context(student, ids)
            for i in range(depth):
                si, ti = s[i].float(), t[i].float()
                sums['rel_err'][i] += float(rel(si, ti).sum())
                sums['cosine'][i] += float(F.cosine_similarity(si, ti, dim=-1).sum())
                sums['norm_ratio'][i] += float((si.norm(dim=-1) / ti.norm(dim=-1).clamp_min(1e-8)).sum())
                sums['lens_kl'][i] += lens_kl(student, teacher, s[i], t_full[i])
                x = t_embed if i == 0 else t[i - 1]
                out = layer(student, i, x.to(s[i].dtype), ctx).float()
                x = x.float()
                sums['block_rel_err'][i] += float(rel(out, ti).sum())
                ds, dt = out - x, ti - x
                sums['delta_cosine'][i] += float(F.cosine_similarity(ds, dt, dim=-1).sum())
                sums['delta_rel_err'][i] += float(rel(ds, dt).sum())
        tokens += n
    return {k: [v / tokens for v in vals] for k, vals in sums.items()}, tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, action='append', required=True, help='name=folder; repeatable')
    ap.add_argument('--teacher', type=Path, required=True)
    ap.add_argument('--pruning', type=Path, required=True, help="The student artifact's pruning.json")
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--split', default='validation')
    ap.add_argument('--sequences', type=int, default=32)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--canary', type=Path)
    ap.add_argument('--canary-gpu', type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    amp = str(args.device).startswith('cuda')
    if amp and args.canary is None:
        raise ValueError('GPU runs need --canary')
    torch.set_num_threads(2)
    dtype = torch.bfloat16 if amp else torch.float32
    started = time.monotonic()
    teacher = load_model(args.teacher, args.device, dtype)
    canary = {}

    def run_canary(when):
        if args.canary is not None:
            try:
                canary[when] = canary_check(teacher, args.teacher, args.data, args.canary, args.device, amp, args.canary_gpu)
            except CanaryFailure as failure:
                save_json(args.output.with_suffix('.canary-failed.json'), dict(when=when, record=json.loads(str(failure))))
                raise SystemExit(3)

    run_canary('start')
    idx = torch.tensor(json.loads(args.pruning.read_text())['residual_indices'], device=args.device)
    rows = np.load(args.data / 'tokens.npz', allow_pickle=False)[args.split][:args.sequences]
    results = {}
    for spec in args.model:
        name, folder = str(spec).split('=', 1)
        student = load_model(Path(folder), args.device, dtype)
        metrics, tokens = probe(student, teacher, rows, idx, args.device, amp)
        results[name] = dict(folder=folder, tokens=tokens, **metrics)
        print(json.dumps({name: {k: round(sum(v) / len(v), 4) for k, v in metrics.items()}}), flush=True)
        del student
        torch.cuda.empty_cache() if amp else None
    run_canary('end')
    results['provenance'] = dict(teacher=str(args.teacher), split=args.split, sequences=args.sequences, canary=canary,
                                 code_sha256=sha256(__file__), data_manifest_sha256=sha256(args.data / 'manifest.json'),
                                 elapsed_seconds=time.monotonic() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, results)


if __name__ == '__main__':
    main()
