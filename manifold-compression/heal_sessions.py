"""Healing in time-boxed sessions on generated answers, with the teacher's log-probs stored.

The pool is shards of answers the dense 9B wrote (in vLLM), each with the teacher's top-20 log-probs
at every answer position (vLLM's raw log-probs match the HF teacher's to 0.0075 nats: vllm_proof).
A session continues the student from the last session's state (LoRA, routers, both optimizers,
step, pool cursor), trains on the next answers the selection keeps until its time is up, and saves.
No teacher pass: the loss at each answer position is the KL from the stored top-20 (renormalized
over those of the student's 64K rows) plus ce_weight times the cross-entropy of the written token.
Prompts carry no loss. The learning rate warms up once, over the first warmup steps of all
sessions, and then stays constant; decay makes a separate evaluation copy that decays it to zero
over a few more steps, and leaves the main state as it was.

Pool layout (written by the generation stage):
  <pool>/<shard>/samples.jsonl   one row per answer: id (problem), sample, source, thinking,
                                 prompt_ids, answer_ids (full-tokenizer ids, ending with the stop
                                 token if the answer finished), correct, finished
  <pool>/<shard>/teacher*.npz    ids_<row> (L, 20) int32 and logprobs_<row> (L, 20) float16, one
                                 line per answer token: the teacher's distribution for that token
                                 (heal_shards.py writes one file per chunk, finished answers only)

    heal_sessions.py session --rules heal_sessions_rules.json --hours 4 --device cuda
    heal_sessions.py decay --rules heal_sessions_rules.json --steps 8 --device cuda
    heal_sessions.py check --copy decayed_88.pt --check-rules geometry_check_nothink_rules.json --device cuda
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import moe_carve as mc
import slim_heal as sh

ROOT = Path(__file__).resolve().parent


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


# ---------------------------------------------------------------- selection and packing

def keep_probability(length, median, alpha, low, high):
    """The length rule: answers outside [low, high] x the problem's median correct length are
    dropped; others are kept with probability min(1, (median / length) ** alpha)."""
    if length < low * median or length > high * median:
        return 0.0
    return min(1.0, (median / length) ** alpha)


def unit(seed, *key):
    """A uniform number in [0, 1) fixed by seed and key."""
    digest = hashlib.sha256(json.dumps([seed, *key]).encode()).digest()
    return int.from_bytes(digest[:8], 'big') / 2 ** 64


def select(rows, cfg):
    """Indices of the rows the selection keeps: not wrong (correct None means unchecked, as for
    chat), finished, and by the length rule against the median kept length of the problem's
    answers in the same thinking mode."""
    ok = [k for k, r in enumerate(rows) if r['correct'] is not False and r['finished']]
    by_problem = {}
    for k in ok:
        # Thinking on and off answers differ several-fold in length: each mode has its own median.
        by_problem.setdefault((rows[k]['id'], bool(rows[k].get('thinking'))), []).append(k)
    chosen = []
    for (problem, _), ks in by_problem.items():
        median = float(np.median([len(rows[k]['answer_ids']) for k in ks]))
        for k in ks:
            p = keep_probability(len(rows[k]['answer_ids']), median, cfg['alpha'], cfg['low'], cfg['high'])
            if unit(cfg['seed'], problem, rows[k]['sample']) < p:
                chosen.append(k)
    return sorted(chosen)


class StudentTokens:
    """The teacher writes in the full 248K vocabulary; the student reads and writes the 64K one.
    About 1% of answer tokens are outside it, but in most answers. Such a token becomes the trimmed
    tokenizer's pieces of its string (the same text, as the student would have to write it): the
    first piece takes the teacher's row, the rest are trained by cross-entropy only. In every
    teacher row, a token outside the vocabulary stands for its first piece (where the teacher's mass
    on it belongs; the duplicate entries this can make shift the KL's value, not its gradient).
    Prompts carry no loss and are re-encoded whole, as the student's checks encode them."""

    def __init__(self, full, short, kept, size):
        self.full, self.short, self.kept, self.split = full, short, set(kept), {}
        self.first = np.arange(size)
        for t in range(min(size, full.get_vocab_size())):
            if t not in self.kept and full.id_to_token(t) is not None:
                self.first[t] = self.pieces(t)[0]

    def pieces(self, t):
        if t in self.kept:
            return [t]
        if t not in self.split:
            got = self.short.tokenizer.model.tokenize(self.full.id_to_token(t))
            self.split[t] = [self.short.original[x.id] for x in got]
            if not self.split[t] or any(p not in self.kept for p in self.split[t]):
                raise ValueError(f'token {t} does not split into kept pieces')
        return self.split[t]

    def prompt(self, ids):
        return self.short.encode(self.full.decode(ids, skip_special_tokens=False))

    def answer(self, ids, tids, tlp):
        """The answer in the student's tokens and its teacher rows (-1 / -inf where none)."""
        out, source = [], []
        for p, t in enumerate(ids):
            parts = self.pieces(t)
            out += parts
            source += [p] + [-1] * (len(parts) - 1)
        source = np.asarray(source)
        rows_ids = np.full((len(out), tids.shape[1]), -1, np.int32)
        rows_lp = np.full((len(out), tids.shape[1]), -np.inf, np.float32)
        have = source >= 0
        t = tids[source[have]]
        rows_ids[have] = np.where(t >= 0, self.first[np.clip(t, 0, None)], -1)
        rows_lp[have] = tlp[source[have]]
        return out, rows_ids, rows_lp


class Pool:
    """The kept answers of every shard, in shard order then row order; packs them into fixed-length
    sequences from a cursor (shard index, kept-answer index)."""

    def __init__(self, folder, cfg, eot, tokens=None):
        """tokens: a StudentTokens, or None when the shards are already in the student's vocabulary."""
        self.folder, self.cfg, self.eot, self.tokens = Path(folder), cfg, eot, tokens
        self.shards = sorted(p.name for p in self.folder.iterdir() if (p / 'samples.jsonl').exists())
        self.cache = {}

    def shard(self, s):
        if s not in self.cache:
            path = self.folder / self.shards[s]
            rows = [json.loads(l) for l in (path / 'samples.jsonl').open()]
            teacher = {}
            # Only finished chunk files: a sync taken mid-write can leave a truncated teacher_<n>.tmp.npz.
            for f in sorted(p for p in path.glob('teacher*.npz') if not p.stem.endswith('.tmp')):
                z = np.load(f)
                teacher.update({k: z for k in z.files})
            self.cache = {s: (rows, select(rows, self.cfg), teacher)}
        return self.cache[s]

    def batch(self, cursor, rows_wanted, length):
        """rows_wanted packed sequences of `length` tokens: ids (B, L), the loss mask (B, L-1) of
        positions that predict an answer token, and that position's teacher ids and log-probs
        (B, L-1, 20; -1 and -inf where there is none). Returns them and the cursor after them,
        or None when the pool runs out first."""
        ids = np.full((rows_wanted, length), self.eot, np.int64)
        mask = np.zeros((rows_wanted, length - 1), bool)
        tids = np.full((rows_wanted, length - 1, 20), -1, np.int32)
        tlp = np.full((rows_wanted, length - 1, 20), -np.inf, np.float32)
        s, k = cursor
        for b in range(rows_wanted):
            pos = 0
            while True:
                if s >= len(self.shards):
                    return None
                rows, chosen, teacher = self.shard(s)
                if k >= len(chosen):
                    s, k = s + 1, 0
                    continue
                r, key = rows[chosen[k]], chosen[k]
                prompt, answer = r['prompt_ids'], r['answer_ids']
                a_ids, a_lp = teacher[f'ids_{key}'][f'ids_{key}'], teacher[f'logprobs_{key}'][f'logprobs_{key}']
                if self.tokens is not None:
                    prompt = self.tokens.prompt(prompt)
                    answer, a_ids, a_lp = self.tokens.answer(answer, a_ids, a_lp)
                item = prompt + answer + [self.eot]
                if len(item) > length:
                    k += 1
                    continue
                if pos + len(item) > length:
                    break
                ids[b, pos:pos + len(item)] = item
                first = pos + len(prompt) - 1     # predicts the first answer token
                n = len(answer)
                mask[b, first:first + n] = True
                tids[b, first:first + n] = a_ids
                tlp[b, first:first + n] = a_lp.astype(np.float32)
                pos += len(item)
                k += 1
        return (ids, mask, tids, tlp), [s, k]


# ---------------------------------------------------------------- loss

def stored_loss(hs, target, tids, tlp, head, index):
    """Summed KL from the stored top-20 (renormalized over the rows the student has) and summed
    cross-entropy of target, at the given positions; hs (P, H), target (P,), tids/tlp (P, 20)."""
    logp = F.linear(hs, head).float().log_softmax(-1)
    rows = index[tids.clamp_min(0)]
    valid = (tids >= 0) & (rows >= 0) & torch.isfinite(tlp)
    # Finite fill, so a position with nothing valid gives zeros, not NaN gradients.
    pt = torch.where(valid, tlp, -1e9).log_softmax(-1)
    ps = logp.gather(1, rows.clamp_min(0))
    kl = (valid * pt.exp() * (pt - ps)).sum()
    ce = -logp.gather(1, index[target][:, None]).sum()
    return kl, ce


def micro_step(model, slim, vocab, batch, cfg, scale):
    """Student pass over the packed ids and backward of the scaled loss at answer positions;
    returns the summed KL, cross-entropy, number of positions and the mean router loss."""
    ids, mask, tids, tlp = batch
    slim.on, slim.stash = True, []
    model.train()
    hs = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1]
    where = mask.nonzero(as_tuple=True)
    h, target, ti, tl = hs[where], ids[:, 1:][where], tids[where], tlp[where]
    kl = ce = 0
    for c in range(0, len(target), cfg['loss_chunk']):
        part = slice(c, c + cfg['loss_chunk'])
        k, e = checkpoint(stored_loss, h[part], target[part], ti[part], tl[part], vocab.head, vocab.index,
                          use_reentrant=False)
        kl, ce = kl + k, ce + e
    ((kl + cfg['ce_weight'] * ce) * scale).backward()
    router = slim.router_backward(scale * len(target) * cfg['router_weight'])
    slim.on = False
    return float(kl.detach()), float(ce.detach()), len(target), router


def rate(step, cfg):
    """Warmup over the first warmup steps of all sessions, then constant."""
    return min(1.0, (step + 1) / cfg['warmup_steps'])


# ---------------------------------------------------------------- sessions

def optimizers(slim, cfg):
    params = slim.lora_parameters()
    return (torch.optim.AdamW(params, lr=cfg['lr'], betas=tuple(cfg['betas']), weight_decay=0.0),
            torch.optim.Adam(slim.router_parameters(), lr=cfg['router_lr']))


def train(model, slim, vocab, pool, state, cfg, until, steps=None, factor=None, checkpoint=None, stop=None):
    """Optimizer steps from state (LoRA, routers, optimizers, step, cursor) until the time `until`
    or `steps` steps; factor(step, done) overrides the rate. checkpoint(state, losses), if given, is
    called every cfg['checkpoint_every'] steps (default 10) with the state so far; stop(), if given,
    is asked before each step and ends the loop when true. Returns the updated state and a log."""
    opt, ropt = optimizers(slim, cfg)
    if state.get('optimizer'):
        opt.load_state_dict(state['optimizer'])
        ropt.load_state_dict(state['router_optimizer'])
    params = slim.lora_parameters()
    step, cursor, done, losses = state['step'], list(state['cursor']), 0, []
    while time.time() < until and (steps is None or done < steps) and not (stop and stop()):
        f = factor(step, done) if factor else rate(step, cfg)
        for group in opt.param_groups:
            group['lr'] = cfg['lr'] * f
        for group in ropt.param_groups:
            group['lr'] = cfg['router_lr'] * f
        got = pool.batch(cursor, cfg['batch_sequences'], cfg['seq_len'])
        if got is None:
            log('Pool exhausted')
            break
        arrays, after = got
        t0 = time.time()
        tensors = [torch.from_numpy(a).to(mc.DEVICE) for a in arrays]
        positions = int(tensors[1].sum())
        totals = [0., 0., 0, []]
        for j in range(0, cfg['batch_sequences'], cfg['micro_batch']):
            part = [x[j:j + cfg['micro_batch']] for x in tensors]
            slim.offset = step + j
            kl, ce, n, router = micro_step(model, slim, vocab, part, cfg, 1 / max(1, positions))
            totals[0] += kl
            totals[1] += ce
            totals[2] += n
            totals[3].append(router)
        norm = float(torch.nn.utils.clip_grad_norm_(params, cfg['clip']))
        opt.step()
        ropt.step()
        opt.zero_grad(set_to_none=True)
        ropt.zero_grad(set_to_none=True)
        step, done, cursor = step + 1, done + 1, after
        n = max(1, totals[2])
        losses.append({'step': step, 'positions': totals[2], 'kl': totals[0] / n, 'ce': totals[1] / n,
                       'router': float(np.mean(totals[3])), 'grad_norm': norm, 'lr': cfg['lr'] * f,
                       'seconds': time.time() - t0, 'cursor': cursor,
                       'others_on_gpu': len(gpu_users())})   # sharing the GPU made evals compute wrong values
        if checkpoint and done % cfg.get('checkpoint_every', 10) == 0:
            checkpoint(dict(state, **slim.state(), optimizer=opt.state_dict(), router_optimizer=ropt.state_dict(),
                            step=step, cursor=cursor), losses)
        if step % cfg['log_every'] == 0 or done == 1:
            last = losses[-1]
            log(f"step {step}: KL {last['kl']:.4f}, CE {last['ce']:.4f}, router {last['router']:.4f}, norm {norm:.2f}, "
                f"{cfg['batch_sequences'] * cfg['seq_len'] / last['seconds']:.0f} tokens/s "
                f"({last['positions'] / last['seconds']:.0f} answer tokens/s), {last['others_on_gpu']} other GPU processes")
    state = dict(state, **slim.state(), optimizer=opt.state_dict(), router_optimizer=ropt.state_dict(),
                 step=step, cursor=cursor)
    return state, losses


def setup(args):
    """The student (from the base run's state, then the last session's), its vocabulary, the pool
    and the session state."""
    import heal_geometry as hg
    import heal_probe as hp
    rules = json.loads(args.rules.read_text())
    geometry = json.loads((ROOT / rules['base']['geometry_rules']).read_text())
    folder = hg.install(geometry, rules['base']['run'])
    sh.RULES = folder / 'rules.json'
    heal_rules, docs = sh.setup()
    model, slim, vocab, _ = hp.load_healed(heal_rules, docs)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    out = ROOT / rules['folder']
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'state.pt'
    if path.exists():
        state = torch.load(path, weights_only=False, map_location=mc.DEVICE)
    else:
        base = torch.load(folder / 'state.pt', weights_only=True, map_location=mc.DEVICE)
        state = dict(base, step=0, cursor=[0, 0], optimizer=None, router_optimizer=None)
    slim.load({'lora': state['lora'], 'routers': state['routers']})
    slim.stride = rules['train']['router_stride']
    import slim_batch as sb
    full, short, kept = sb.load_tokenizers()
    tokens = StudentTokens(full, short, kept, len(vocab.index))
    pool = Pool(ROOT / rules['pool'], rules['select'], vocab.eot, tokens)
    return rules, heal_rules, docs, model, slim, vocab, pool, state, out


YIELD = 75   # exit code: the session stopped to let other sessions use the GPU and will resume


def stage_session(args):
    """args.hours of training, counted across segments: with --segment-minutes the session stops at
    the end of each segment, or as soon as another session queues for the GPU (dibs run --preemptible
    creates $DIBS_YIELD_FILE), saves its state and exits YIELD; run again, it resumes where it stopped."""
    import os
    import sys
    rules, heal_rules, docs, model, slim, vocab, pool, state, out = setup(args)
    partial = out / 'session_partial.json'
    earlier = json.loads(partial.read_text()) if partial.exists() else {}
    start = earlier.get('started', {k: state[k] for k in ('step', 'cursor')})
    before, trained = earlier.get('losses', []), earlier.get('trained_seconds', 0.0)
    t0 = time.time()
    until = t0 + 3600 * args.hours - 60 * rules['train']['reserve_minutes'] - trained
    segment_end = t0 + 60 * args.segment_minutes if args.segment_minutes else math.inf
    yield_file = os.environ.get('DIBS_YIELD_FILE')

    def stop():
        return time.time() >= segment_end or bool(yield_file and os.path.exists(yield_file))

    def save(state, losses, final=False):
        # Atomic, so a crash mid-write leaves the last good state; a session cut short by a crash
        # resumes from its last checkpoint and its partial record says how far it got.
        torch.save(state, out / 'state.pt.tmp')
        (out / 'state.pt.tmp').replace(out / 'state.pt')
        if final:
            partial.unlink(missing_ok=True)
        else:
            partial.write_text(json.dumps({'started': start, 'checkpoint': {k: state[k] for k in ('step', 'cursor')},
                                           'hours': args.hours, 'losses': before + losses,
                                           'trained_seconds': trained + time.time() - t0, 'saved_at': time.time()}))
            log(f"checkpoint: step {state['step']}, cursor {state['cursor']}")

    if until <= t0:
        state, losses = state, []
    else:
        state, losses = train(model, slim, vocab, pool, state, rules['train'], until, checkpoint=save, stop=stop)
    if time.time() < until and stop():
        save(state, losses)
        log(f"yielding the GPU at step {state['step']} ({(trained + time.time() - t0) / 3600:.2f} of {args.hours} h trained)")
        sys.exit(YIELD)
    save(state, losses, final=True)
    record = {'started': start, 'ended': {k: state[k] for k in ('step', 'cursor')}, 'hours': args.hours,
              'losses': before + losses, 'finished_at': time.time()}
    sessions = out / 'sessions.jsonl'
    with sessions.open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    log(f"session: steps {start['step']} -> {state['step']}, cursor {state['cursor']}")


def stage_decay(args):
    """An evaluation copy: args.steps more steps on the next answers with the rate falling linearly
    to zero, saved as decayed_<step>.pt. The main state, step and cursor stay as they were."""
    rules, heal_rules, docs, model, slim, vocab, pool, state, out = setup(args)
    n = args.steps
    state, losses = train(model, slim, vocab, pool, state, rules['train'], math.inf, steps=n,
                          factor=lambda step, done: 1 - done / n)
    name = out / f"decayed_{state['step']}.pt"
    torch.save({'lora': state['lora'], 'routers': state['routers'], 'step': state['step']}, name)
    log(f'decayed copy: {name.name}')


# The dense model's eval pass never involves a healed copy, so its values are the same in every
# process. On this machine some processes compute wrong values (2026-11-05/06: the dense pass's NLL
# off by thousands of nats on 10 to 30 of 116 docs, nondeterministically, in a fresh process). Each
# check compares its dense pass with a clean run's and records how many docs differ.
CANARY = 'check_0_on/eval_healed.jsonl'


def dense_canary(rows, reference):
    """How many eval docs' dense NLL (full and 64K paths) differ from a clean run's."""
    ref = {r['id']: r for r in reference}
    faulty = [r['id'] for r in rows if r['id'] in ref and (
        abs(r['nll']['dense'] - ref[r['id']]['nll']['dense']) > 1e-3 or
        abs(r['trimmed']['nll_dense'] - ref[r['id']]['trimmed']['nll_dense']) > 1e-4 * abs(ref[r['id']]['trimmed']['nll_dense']) + 0.5)]
    return {'docs': sum(r['id'] in ref for r in rows), 'faulty_docs': len(faulty), 'faulty': faulty}


def gpu_users():
    """Other processes with the GPU open: ROCm through /dev/kfd, Vulkan (llama-server) through a render node.
    Desktop programs under /usr also hold the render node and are left out."""
    import os
    users = []
    for pid in filter(str.isdigit, os.listdir('/proc')):
        if int(pid) == os.getpid():
            continue
        try:
            fds = [os.readlink(f'/proc/{pid}/fd/{fd}') for fd in os.listdir(f'/proc/{pid}/fd')]
            cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if '/dev/kfd' in fds or (any(f.startswith('/dev/dri/renderD') for f in fds) and not cmd.startswith('/usr/')):
            users.append(cmd[:160])
    return users


def stage_check(args):
    """slim_opd's check (sampled arm only) of a decayed copy, in <folder>/check_<step>_<mode>/: the
    base run's student with the copy's LoRA and routers. The check's reproduction step would compare
    against the base run's last milestone, which a healed-on copy no longer matches; it records the
    copy's eval KL instead."""
    import heal_geometry as hg
    import heal_probe as hp
    import slim_opd as so
    rules = json.loads(args.rules.read_text())
    geometry = json.loads((ROOT / rules['base']['geometry_rules']).read_text())
    folder = hg.install(geometry, rules['base']['run'])
    sh.RULES = folder / 'rules.json'
    path = ROOT / rules['folder'] / args.copy
    copy = torch.load(path, weights_only=False, map_location='cpu')
    check = json.loads(args.check_rules.read_text())
    mode = 'off' if check['eval'].get('enable_thinking') is False else 'on'
    name = f"check_{copy['step']}_{mode}" + check.get('suffix', '')   # e.g. '_seed2' for a repeat at another seed
    so.FOLDER = str(Path(rules['folder']).relative_to(mc.OUT.relative_to(ROOT)) / name)
    check['eval']['arms'] = [['healed', 'sampled']]
    check['copy'] = {'path': str(path.relative_to(ROOT)), 'sha256': sh.sha(path), 'step': copy['step'],
                     'check_rules': str(args.check_rules)}
    so.out('check_rules.json').parent.mkdir(parents=True, exist_ok=True)
    so.RULES = so.out('check_rules.json')
    so.RULES.write_text(json.dumps(check, indent=1))
    so.finished_adapter = lambda spec: None
    base_load = hp.load_healed

    def load(heal_rules, docs):
        model, slim, vocab, same = base_load(heal_rules, docs)
        slim.adapters[True] = {'lora': copy['lora'], 'routers': copy['routers']}
        hp.heal(slim, True)
        return model, slim, vocab, same
    hp.load_healed = load

    kept = {}

    def reproduce(model, slim, vocab, docs, heal_rules, cfg):
        kept.update(model=model, slim=slim, vocab=vocab, docs=docs, heal_rules=heal_rules)
        rows, healed = so.drift(model, slim, vocab, docs, heal_rules, 'healed')
        reference = ROOT / rules['folder'] / CANARY
        if reference.exists():
            healed['dense_canary'] = dense_canary(rows, [json.loads(l) for l in reference.open()])
            healed['dense_canary']['other_gpu_processes'] = gpu_users()
            c = healed['dense_canary']
            log(f"dense canary: {c['faulty_docs']} of {c['docs']} eval docs differ from the reference dense pass"
                + (' -- THIS PROCESS COMPUTES WRONG VALUES; do not use its results' if c['faulty_docs'] else ''))
            if c['faulty_docs'] and args.abort_on_fault:
                # Stop before the hour of sampling; keep the faulty eval aside so a retry starts clean.
                folder = so.out('check_rules.json').parent
                (folder / 'dense_canary.json').write_text(json.dumps(c, indent=1))
                folder.rename(folder.with_name(f"faulty_{folder.name}_{time.strftime('%m%d_%H%M')}"))
                raise SystemExit(3)
        return healed, None
    so.reproduce = reproduce
    so.stage_check(args)
    reference = ROOT / rules['folder'] / CANARY
    if reference.exists():
        # The canary again after sampling: a process can start computing wrong values mid-run.
        hp.heal(kept['slim'], True)
        rows, _ = so.drift(kept['model'], kept['slim'], kept['vocab'], kept['docs'], kept['heal_rules'], 'healed_end')
        c = dense_canary(rows, [json.loads(l) for l in reference.open()])
        c['other_gpu_processes'] = gpu_users()
        result = json.loads(so.out('check.json').read_text())
        result['dense_canary_end'] = c
        so.out('check.json').write_text(json.dumps(result, indent=1))
        nonfinite = sum(a.get('nonfinite', 0) or 0 for a in result['arms'].values() if isinstance(a, dict))
        log(f"dense canary after sampling: {c['faulty_docs']} of {c['docs']} eval docs differ; {nonfinite} non-finite rows"
            + (' -- FAULTY RUN' if c['faulty_docs'] or nonfinite else ' -- clean'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['session', 'decay', 'check'])
    ap.add_argument('--copy', help='check: the decayed copy, e.g. decayed_88.pt')
    ap.add_argument('--check-rules', type=Path, help='check: geometry_check_rules.json (thinking on) or geometry_check_nothink_rules.json')
    ap.add_argument('--rules', type=Path, default=ROOT / 'heal_sessions_rules.json')
    ap.add_argument('--hours', type=float, default=4)
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--segment-minutes', type=float, default=0, help='session: yield the GPU after this many minutes (exit 75), 0 = never')
    ap.add_argument('--abort-on-fault', action='store_true', help='check: exit 3 when the dense canary fails, before sampling')
    args = ap.parse_args()
    mc.DEVICE = args.device
    {'session': stage_session, 'decay': stage_decay, 'check': stage_check}[args.stage](args)


if __name__ == '__main__':
    main()
