"""Hierarchical blockwise healing, with overlapping windows, of the S0_K24 carve (Qwen3.5-9B).

The user's proposal (2026-11-06): heal single layers first, then jointly heal progressively larger
groups of layers (2, 4, 8, ...), each trained on the current student's own stream against the
dense teacher's targets; at each scale shift the windows by half a width on alternate steps, so
every block boundary also sits inside a window; check the whole model between stages.

One step trains every window of one partition of the layers at once, on the step's batch. Each
window reads the student's stream at its first layer, detached, so no gradient crosses a window
boundary. A window that ends before the last layer is scored by the relative squared error of its
output stream against the teacher's at that layer; the window that ends at the last layer by
S0_K24's healing loss (KL over the kept rows plus ce_weight x CE). With one window of every layer
a step is slim_heal.micro_step's step (test_blockwise_heal checks the gradients). Everything else
is S0_K24's run (heal_geometry_rules.json): the carve, student, LoRA, routers, data, batch order,
token count and cosine schedule. Settings and predictions: blockwise_heal_rules.json.

  python blockwise_heal.py heal --rules blockwise_heal_rules.json --run HB_1M --device cuda
  python blockwise_heal.py baseline --rules blockwise_heal_rules.json --device cuda
  python blockwise_heal.py check --rules blockwise_heal_rules.json --run HB_1M \\
      --check-rules geometry_check_rules.json --device cuda
  python blockwise_heal.py compare --rules blockwise_heal_rules.json --run HB_1M

heal and check write <folder>/<run>/, baseline <folder>/baseline/, compare <folder>/<run>/compare.json.
Nothing is written beside the carve or into heal_geometry/."""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import heal_geometry as hg
import moe_carve as mc
import slim_batch as sb
import slim_heal as sh
from h0_mlp_probe import deadline, group_mean, log, paired_interval, save, sha

ROOT = mc.ROOT
CODE = ['blockwise_heal.py', 'test_blockwise_heal.py', 'heal_geometry.py', 'slim_heal.py', 'slim_batch.py',
        'moe_carve.py', 'h0_mlp_probe.py']


# ---------------------------------------------------------------- windows

def partition(layers, width, shifted):
    """Windows [a, b) that cover layers 0..layers once: starting at multiples of width, or (shifted)
    with a first window of width // 2 and the rest moved up by that much. Width 1 and widths of
    the whole model have only the unshifted partition."""
    if width < 1:
        raise ValueError(width)
    if shifted and 1 < width < layers:
        starts = [0] + list(range(width // 2, layers, width))
    else:
        starts = list(range(0, layers, width))
    return [(a, min(b, layers)) for a, b in zip(starts, starts[1:] + [layers])]


def plan(stages, layers):
    """Per step: (stage index, width, shifted, windows). Shifted on the odd steps of each stage."""
    steps = []
    for k, stage in enumerate(stages):
        for j in range(stage['steps']):
            shifted = j % 2 == 1 and 1 < stage['width'] < layers
            steps.append((k, stage['width'], shifted, partition(layers, stage['width'], shifted)))
    return steps


# ---------------------------------------------------------------- the stream, layer by layer

def context(model, x):
    """The embeddings of x (B, T) and what every decoder layer takes besides the stream, built as
    Qwen3_5TextModel.forward builds them for a batch with no padding and no cache."""
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
    inner = model.model
    embeds = inner.embed_tokens(x)
    position = torch.arange(x.shape[1], device=x.device).view(1, 1, -1).expand(4, x.shape[0], -1)
    text, position = position[0], position[1:]
    kw = dict(config=inner.config, inputs_embeds=embeds, attention_mask=None, past_key_values=None, position_ids=text)
    masks = {'full_attention': create_causal_mask(**kw), 'linear_attention': create_recurrent_attention_mask(**kw)}
    return embeds, {'rope': inner.rotary_emb(embeds, position), 'masks': masks, 'position': text}


def layer_call(model, i, ctx):
    layer, kind = model.model.layers[i], model.config.layer_types[i]
    return lambda h: layer(h, position_embeddings=ctx['rope'], attention_mask=ctx['masks'][kind],
                           position_ids=ctx['position'], past_key_values=None, use_cache=False)


def relative_error(hs, ht):
    """Per position (from 1 on): squared error over the teacher's squared norm, in fp32."""
    hs, ht = hs[:, 1:].float(), ht[:, 1:].float()
    return (hs - ht).pow(2).sum(-1) / ht.pow(2).sum(-1).clamp_min(1e-6)


@torch.no_grad()
def stream(model, slim, x, on):
    """The residual stream entering layer 0 and leaving every layer: a list of layers + 1 tensors."""
    slim.on = on
    h, ctx = context(model, x)
    out = [h]
    for i in range(model.config.num_hidden_layers):
        h = layer_call(model, i, ctx)(h)
        out.append(h)
    slim.on = False
    return out


def window_loss(model, vocab, out, x, teacher, b, cfg, bw):
    """Summed loss of a window ending at layer b, and (kl, ce, rel) sums for the log."""
    layers = model.config.num_hidden_layers
    if b < layers:
        rel = relative_error(out, teacher[b]).sum()
        return bw['mse_weight'] * rel, 0., 0., float(rel.detach())
    norm = model.model.norm
    hs = norm(out)[:, :-1].flatten(0, 1)
    with torch.no_grad():
        ht = norm(teacher[layers])[:, :-1].flatten(0, 1)
    target = vocab.index[x[:, 1:]].flatten()
    if (target < 0).any():
        raise ValueError('Healing ids outside the kept rows')
    kl = ce = 0
    for c in range(0, len(target), cfg['loss_chunk']):
        part = slice(c, c + cfg['loss_chunk'])
        k, e = checkpoint(sh.chunk_loss, hs[part], ht[part], target[part], vocab.head, use_reentrant=False)
        kl, ce = kl + k, ce + e
    return kl + cfg['ce_weight'] * ce, float(kl.detach()), float(ce.detach()), 0.


def block_step(model, slim, vocab, x, windows, cfg, bw, scale):
    """Teacher pass, then each window of the student trained on its own loss with its input
    detached; router targets recorded once per layer, as micro_step records them. Backward of the
    scaled losses; returns the summed KL and CE of the last window, the summed relative error at
    every layer output (measured on the student's stream, no gradient) and the mean router loss."""
    layers = model.config.num_hidden_layers
    if not partition_check(windows, layers):
        raise ValueError(f'Windows {windows} do not cover the {layers} layers in order')
    model.eval()
    teacher = stream(model, slim, x, False)
    slim.on, slim.stash = True, []
    model.train()
    h, ctx = context(model, x)
    rel_all = np.zeros(layers)
    kl = ce = 0.
    for a, b in windows:
        out = h.detach()
        for i in range(a, b):
            out = checkpoint(layer_call(model, i, ctx), out, use_reentrant=False)
            with torch.no_grad():
                rel_all[i] = float(relative_error(out, teacher[i + 1]).sum())
        loss, k, e, _ = window_loss(model, vocab, out, x, teacher, b, cfg, bw)
        (loss * scale).backward()
        kl, ce = kl + k, ce + e
        h = out.detach()
    router = slim.router_backward(scale * x[:, 1:].numel() * cfg['router_weight'])
    slim.on = False
    return kl, ce, rel_all, router


def partition_check(windows, layers):
    """The windows if they are contiguous and cover 0..layers, else an empty list."""
    edges = [a for a, _ in windows] + [windows[-1][1]] if windows else []
    ok = edges and edges[0] == 0 and edges[-1] == layers and all(a < b for a, b in windows) and all(
        windows[k][1] == windows[k + 1][0] for k in range(len(windows) - 1))
    return list(windows) if ok else []


def layer_parameters(slim, layers):
    """The LoRA parameters of each layer (index -> list)."""
    params = {i: [] for i in range(layers)}
    for name, m in slim.adapted.items():
        if m.a is not None:
            params[int(name.split('.')[0])] += [m.a, m.b]
    return params


@torch.no_grad()
def boundary_profile(model, slim, docs):
    """Domain-equal mean relative error of the student from the teacher at every layer output,
    and the per-document values."""
    model.eval()
    rows = []
    for doc in docs:
        x = sb.tensor([doc['input_ids']])
        t, s = stream(model, slim, x, False), stream(model, slim, x, True)
        rows.append([float(relative_error(s[j], t[j]).mean()) for j in range(1, len(t))])
        del t, s
    groups = [d['domain'] for d in docs]
    per_layer = [group_mean([r[j] for r in rows], groups) for j in range(len(rows[0]))]
    return {'per_layer': per_layer, 'mean': float(np.mean(per_layer)), 'median': float(np.median(per_layer)),
            'docs': [d['id'] for d in docs]}


# ---------------------------------------------------------------- set-up

def folder_of(rules, run):
    return ROOT / rules['folder'] / run


def install(rules, run):
    """heal_geometry's install for the carve, with every output moved into this run's folder;
    the healing data stays the pilot's. Returns the folder."""
    folder = folder_of(rules, run)
    hg.install(rules, run)
    data_out = sh.out
    sh.out = lambda n: data_out(n) if n in hg.DATA_FILES else folder / n
    sh.state_path = lambda: folder / 'state.pt'
    return folder


def boundary_docs(docs, per_domain):
    return [d for d in sb.docs_of(docs, 'eval', per_domain) if d['domain'] != 'copy']


def provenance(rules_path):
    return {'code_sha256': {n: sha(ROOT / n) for n in CODE}, 'rules_sha256': sha(rules_path),
            'carve_rules_sha256': sha(sb.CARVE_RULES), 'slim_rules_sha256': sha(sb.RULES),
            'corpus_sha256': sha(mc.OUT / 'corpus.json')}


# ---------------------------------------------------------------- heal

def heal(args, rules):
    folder = install(rules, args.run)
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('heal.json', 'adapter.pt'):
        if (folder / name).exists():
            raise ValueError(f'{folder / name} exists; refusing to overwrite')
    (folder / 'rules.json').write_text(json.dumps(rules, indent=1))
    sh.RULES = folder / 'rules.json'
    _, docs = sh.setup()
    cfg, bw = rules['train'], rules['blockwise']
    base = json.loads((ROOT / rules['baseline']['folder'] / 'heal.json').read_text())
    record = json.loads(sh.out('data.json').read_text())
    if sha(sh.out('data.npz')) != record['data_sha256'] or record['data_sha256'] != base['data_sha256']:
        raise RuntimeError('Healing data is not the data S0_K24 healed on')
    seqs = np.load(sh.out('data.npz'))['ids']
    t0 = time.time()
    stop_at = min(t0 + 3600 * cfg['max_hours'],
                  float(os.environ.get('MANIFOLD_JOB_DEADLINE', 'inf')) - 60 * cfg['reserve_minutes'])
    model = mc.load_model(rules['threads'])
    factors = sh.make_factors(model, docs)
    slim = sh.build(model, rules, factors)
    vocab = sh.Vocab(model)
    layers = model.config.num_hidden_layers
    evals, held_out = sb.docs_of(docs, 'eval'), rules['data']['held_out']
    probes = boundary_docs(docs, bw['boundary_docs'])
    steps_plan = plan(bw['stages'], layers)
    length, batch = seqs.shape[1], cfg['batch_sequences']
    steps = len(steps_plan)
    if steps != base['steps'] or steps != math.ceil(cfg['tokens'] / (batch * length)):
        raise ValueError(f'{steps} steps planned; S0_K24 ran {base["steps"]}')
    order = np.random.default_rng(cfg['seed']).permutation(len(seqs))
    stage_end = {s for s in range(1, steps + 1) if s == steps or steps_plan[s][0] != steps_plan[s - 1][0]}
    params_by_layer = layer_parameters(slim, layers)
    params = slim.lora_parameters()
    if sum(map(len, params_by_layer.values())) != len(params):
        raise RuntimeError('Not every LoRA parameter belongs to a layer')
    router_optimizer = torch.optim.Adam(slim.router_parameters(), lr=cfg['router_lr'])
    slim.stride = cfg['router_stride']
    milestones, losses = [], []

    def milestone(tokens, step):
        t = time.time()
        rows = sh.evaluate(model, slim, vocab, evals)
        sh.fresh(f'eval_{tokens}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        summary = dict(sh.summarize(rows, held_out), tokens=tokens, steps=step,
                       stage=steps_plan[step - 1][1] if step else 0,
                       boundary=boundary_profile(model, slim, probes), seconds=time.time() - t)
        milestones.append(summary)
        save({'milestones': milestones, 'loss_by_step': losses}, sh.out('progress.json'))
        log(f'{tokens} tokens (step {step}, width {summary["stage"]}): KL {summary["kl"]:.3f} (in-mix '
            f'{summary["in_mix"]["kl"]:.3f}, held-out {summary["held_out"]["kl"]:.3f}), 64K bits per byte '
            f'{summary["bpb_student"]:.4f}, agreement {summary["agree"]:.3f}, router recall {summary["recall"]:.3f}, '
            f'layer error mean {summary["boundary"]["mean"]:.4f} median {summary["boundary"]["median"]:.4f}')
        return summary

    micro, tokens, step, stopped, optimizer = cfg['micro_batch'], 0, 0, 'tokens', None
    with deadline(cfg['max_hours'] + 0.5):
        first = milestone(0, 0)
        start = base['milestones'][0]
        check = {'kl': first['kl'] - start['kl'], 'bpb': first['bpb_student'] / start['bpb_student'] - 1}
        if abs(check['kl']) > cfg['check']['kl'] or abs(check['bpb']) > cfg['check']['bpb']:
            raise RuntimeError(f'The student does not reproduce S0_K24\'s start: {check}')
        while step < steps:
            if time.time() > stop_at:
                stopped = 'time'
                break
            stage, width, shifted, windows = steps_plan[step]
            inside = step - min(s for s, p in enumerate(steps_plan) if p[0] == stage)
            if inside == 0 or optimizer is None:
                # A fresh optimizer per stage: the loss each layer sees changes with the width.
                optimizer = torch.optim.AdamW(params, lr=cfg['lr'], betas=tuple(cfg['betas']), weight_decay=0.0)
            factor = sh.schedule(step, steps, cfg['warmup_steps'], cfg['lr_floor'])
            warm = min(1., (inside + 1) / bw['stage_warmup']) if stage > 0 else 1.
            for group in optimizer.param_groups:
                group['lr'] = cfg['lr'] * factor * warm
            for group in router_optimizer.param_groups:
                group['lr'] = cfg['router_lr'] * factor
            chosen = order[step * batch:(step + 1) * batch]
            t = time.time()
            try:
                kl = ce = 0.
                rel, routers = np.zeros(layers), []
                for j in range(0, batch, micro):
                    x = torch.from_numpy(seqs[chosen[j:j + micro]].astype(np.int64)).to(mc.DEVICE)
                    slim.offset = step + j
                    k, e, r, router = block_step(model, slim, vocab, x, windows, cfg, bw, 1 / (batch * (length - 1)))
                    kl, ce, rel = kl + k, ce + e, rel + r
                    routers.append(router)
            except torch.cuda.OutOfMemoryError:
                if micro == 1:
                    raise
                optimizer.zero_grad(set_to_none=True)
                router_optimizer.zero_grad(set_to_none=True)
                slim.on, slim.stash, slim.done = False, None, set()
                torch.cuda.empty_cache()
                micro //= 2
                log(f'Out of memory: micro-batch now {micro}')
                continue
            norms = [float(torch.nn.utils.clip_grad_norm_([p for i in range(a, b) for p in params_by_layer[i]],
                                                          cfg['clip'])) for a, b in windows]
            optimizer.step()
            router_optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            router_optimizer.zero_grad(set_to_none=True)
            step += 1
            tokens += batch * length
            n = batch * (length - 1)
            losses.append({'step': step, 'tokens': tokens, 'width': width, 'shifted': shifted,
                           'windows': windows, 'kl': kl / n, 'ce': ce / n, 'layer_error': (rel / n).tolist(),
                           'router': float(np.mean(routers)), 'grad_norms': norms,
                           'lr': cfg['lr'] * factor * warm, 'seconds': time.time() - t})
            if step % cfg['log_every'] == 0 or step == 1:
                rate = batch * length / losses[-1]['seconds']
                memory = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0
                err = rel / n
                log(f'step {step}/{steps} width {width}{" shifted" if shifted else ""}: KL {kl / n:.4f}, '
                    f'CE {ce / n:.4f}, layer error median {np.median(err):.4f} max {err.max():.4f}, router '
                    f'{losses[-1]["router"]:.4f}, max window norm {max(norms):.2f}, {rate:.0f} tokens/s, {memory:.1f} GB')
            if step in stage_end:
                torch.save(slim.state(), sh.state_path())
                milestone(tokens, step)
    if not milestones or milestones[-1]['tokens'] != tokens:
        torch.save(slim.state(), sh.state_path())
        milestone(tokens, step)
    torch.save(slim.state(torch.bfloat16), sh.fresh('adapter.pt'))
    save(dict(provenance(sh.RULES), data_sha256=record['data_sha256'], factors_sha256=sha(sh.factors_path()),
              state_sha256=sha(sh.state_path()), adapter_sha256=sha(sh.out('adapter.pt')), reproduces_start=check,
              trainable={'lora': sum(p.numel() for p in params), 'routers': sum(p.numel() for p in slim.router_parameters())},
              planned_steps=steps, steps=step, tokens=tokens, stopped=stopped, final_micro_batch=micro,
              milestones=milestones, loss_by_step=losses, seconds=time.time() - t0),
         sh.fresh('heal.json'))
    log(f'Healed on {tokens} tokens ({stopped}); KL {milestones[0]["kl"]:.3f} -> {milestones[-1]["kl"]:.3f}')


# ---------------------------------------------------------------- the end-to-end run's profile

def baseline(args, rules):
    """The boundary profile of S0_K24 unhealed and at 1M, on the same documents."""
    hg_rules = json.loads((ROOT / rules['baseline']['rules']).read_text())
    hg.install(hg_rules, rules['baseline']['run'])
    sh.RULES = ROOT / rules['baseline']['folder'] / 'rules.json'
    heal_rules, docs = sh.setup()
    import heal_probe as hp
    target = ROOT / rules['folder'] / 'baseline' / 'boundary.json'
    if target.exists():
        raise ValueError(f'{target} exists; refusing to overwrite')
    t0 = time.time()
    model, slim, _, same = hp.load_healed(heal_rules, docs)
    probes = boundary_docs(docs, rules['blockwise']['boundary_docs'])
    result = {}
    for name, healed in (('unhealed', False), ('healed_1m', True)):
        hp.heal(slim, healed)
        result[name] = boundary_profile(model, slim, probes)
        log(f'S0_K24 {name}: layer error mean {result[name]["mean"]:.4f}, median {result[name]["median"]:.4f}')
    target.parent.mkdir(parents=True, exist_ok=True)
    save(dict(provenance(args.rules), baseline_adapter_sha256=sha(sh.out('adapter.pt')), factors_identical=same,
              seconds=time.time() - t0, **result), target)


# ---------------------------------------------------------------- generations

def check(args, rules):
    import slim_opd as so
    folder = install(rules, args.run)
    sh.RULES = folder / 'rules.json'
    so.RULES = args.check_rules.resolve()
    sub = json.loads(so.RULES.read_text())['folder']
    so.FOLDER = os.path.relpath(folder / sub, mc.OUT)
    so.finished_adapter = lambda spec: None
    so.stage_check(args)


# ---------------------------------------------------------------- comparison (local, no model)

def kl_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {r['id']: r for r in rows if r['domain'] != 'copy'}


def compare(args, rules):
    mc.configure(sb.CARVE_RULES)
    folder = folder_of(rules, args.run)
    mine = json.loads((folder / 'heal.json').read_text())
    base_folder = ROOT / rules['baseline']['folder']
    base = json.loads((base_folder / 'heal.json').read_text())
    final = mine['milestones'][-1]['tokens']
    if final != base['milestones'][-1]['tokens']:
        raise ValueError(f'Final milestones differ: {final} against {base["milestones"][-1]["tokens"]}')
    a, b = kl_rows(folder / f'eval_{final}.jsonl'), kl_rows(base_folder / f'eval_{final}.jsonl')
    if set(a) != set(b):
        raise ValueError('The two evaluations scored different documents')
    ids = sorted(a)
    groups = [a[i]['domain'] for i in ids]
    held = set(rules['data']['held_out'])
    test = {}
    for part, keep in (('all', lambda d: True), ('in_mix', lambda d: d not in held), ('held_out', lambda d: d in held)):
        chosen = [i for i in ids if keep(a[i]['domain'])]
        point, low, high = paired_interval([a[i]['kl']['student'] for i in chosen], [b[i]['kl']['student'] for i in chosen],
                                           1.0, 10000, 26110601, [a[i]['domain'] for i in chosen])
        test[part] = {'difference': point, 'ci95': [low, high]}
    d = test['all']
    verdict = 'better' if d['ci95'][1] < 0 else 'worse' if d['ci95'][0] > 0 else 'tie'
    result = {'final_tokens': final, 'kl': {'blockwise': mine['milestones'][-1]['kl'], 'end_to_end': base['milestones'][-1]['kl']},
              'difference': test, 'verdict_kl': verdict,
              'curves': {'blockwise': [(m['tokens'], m['stage'], m['kl'], m['held_out']['kl'], m['boundary']['mean'])
                                       for m in mine['milestones']],
                         'end_to_end': [(m['tokens'], m['kl'], m['held_out']['kl']) for m in base['milestones']]}}
    profile = ROOT / rules['folder'] / 'baseline' / 'boundary.json'
    if profile.exists():
        p = json.loads(profile.read_text())
        result['layer_error'] = {'end_to_end_start': p['unhealed']['mean'], 'end_to_end_1m': p['healed_1m']['mean'],
                                 'blockwise_start': mine['milestones'][0]['boundary']['mean'],
                                 'blockwise_1m': mine['milestones'][-1]['boundary']['mean']}
    for name, path in (('blockwise', folder / 'check' / 'check.json'), ('end_to_end', base_folder / 'check' / 'check.json')):
        if path.exists():
            result.setdefault('check', {})[name] = json.loads(path.read_text())['arms']
    save(dict(rules_sha256=sha(args.rules), **result), folder / 'compare.json')
    log(f'Final KL {result["kl"]["blockwise"]:.3f} against {result["kl"]["end_to_end"]:.3f}: difference '
        f'{d["difference"]:+.3f} [{d["ci95"][0]:+.3f}, {d["ci95"][1]:+.3f}], {verdict}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['heal', 'baseline', 'check', 'compare'])
    ap.add_argument('--rules', type=Path, required=True)
    ap.add_argument('--run')
    ap.add_argument('--check-rules', type=Path)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    mc.DEVICE = args.device
    rules = json.loads(args.rules.read_text())
    {'heal': heal, 'baseline': baseline, 'check': check, 'compare': compare}[args.stage](args, rules)


if __name__ == '__main__':
    main()
