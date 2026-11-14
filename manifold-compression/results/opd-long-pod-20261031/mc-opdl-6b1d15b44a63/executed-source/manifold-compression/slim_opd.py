"""On-policy distillation of the healed Qwen3.5-9B student, to stop its long generations looping.

The student is slim_heal.py's healed student at 25% of MLP neurons with its learned routers, as
heal_probe.load_healed rebuilds it. The teacher is the dense model (slim.on False), or with the rules'
train.teacher 'every_neuron' the healed student itself with every neuron and the healing run's
adapter, frozen (Teacher): it differs from the student only in the neurons the student skips. Each
round:
1. The teacher answers half of the round's prompts, sampled; with finished_only, only the answers
   that end train.
2. The student answers the other half, half of them greedy (the decoding the checks use, where it
   loops) and half sampled.
3. The student trains on both. On the teacher's answers, toward the teacher's next-token
   distribution (forward KL, as healing did on text). On its own answers, its distribution against
   the teacher's at every position it reached (reverse KL): this trains it where it goes wrong when
   it runs on its own output, as the checks run it.
A teacher reading a looping prefix predicts more of the loop, and reverse KL on such text sharpened
the loops of an earlier student (findings.md, 2026-10-19). So the student's answers count only up to
the token that starts their first long repeat (loop_end), and stop being generated soon after it
(LoopStop).

Only the LoRA updates train; the routers, the low-rank factors and everything else stay as healed.
All distributions are over the 64K kept rows, renormalized. Generations are batched and left-padded,
with fp32 DeltaNet state like heal_probe.py's checks; each row's first real token runs every neuron
(FirstReal), as position 0 of an unpadded sequence does.

Two more settings carry a run on from an earlier one (opd_long_rules.json). train.start starts the
student from that run's adapter and its prompts where that run stopped; the teacher keeps the
healing run's adapter. train.text adds healing text each round, the sequences the healing run never
reached first, on which the student learns toward the dense model (forward KL), as healing did, to
hold its drift on general text.

Settings, predictions and how to read the result are in opd_rules.json (the dense teacher),
opd_self_rules.json (the every-neuron teacher) and opd_long_rules.json (the every-neuron teacher's
run carried on, with healing text); outputs go to the rules' folder (opd/ by default) beside the
carve's. Stages:

  prompts  (local) the training prompts (opd_prompts.json): GSM8K training questions and UltraChat
           first turns outside the benchmark, alternating.
  before   the healed student's drift (it must reproduce the healing run's last milestone), a check
           that padding changes nothing, and batched generations on the GSM8K check prompts,
           greedy and sampled, of the healed student, the healed student with every neuron, and
           the dense model (eval.before_arms chooses among them); with train.start, also the
           starting student's drift and generations.
  train    the distillation, until its time is up; halfway, greedy generations of the student so far.
           With --resume, a stopped train stage carries on from its last checkpoint (latest.pt): the
           rounds up to it, its prompts and healing text, and the rest of its time budget, with the
           learning-rate cosine where it was and a fresh optimizer warmed up again.
  after    the distilled student's drift and its generations, as in before.
  report   (local) one summary.
"""
import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

# Long generations between training steps fragment the allocator's fixed-size segments.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import heal_probe as hp
import moe_carve as mc
import slim_batch as sb
import slim_heal as sh
from h0_mlp_probe import deadline, key, log, paired_interval, save, sha, token_kl

ROOT = mc.ROOT
RULES = ROOT / 'opd_rules.json'
CODE = ['slim_opd.py', 'heal_probe.py', 'slim_heal.py', 'slim_batch.py', 'moe_carve.py', 'h0_mlp_probe.py',
        'test_slim_opd.py']
# Generation arms of the checks: (the student, not the dense model; its routing).
ARMS = {'healed': (True, 'learned'), 'every_neuron': (True, 'all'), 'dense': (False, 'learned'),
        'start': (True, 'learned'), 'mid': (True, 'learned'), 'distilled': (True, 'learned')}
REFERENCES = ['healed', 'every_neuron', 'dense']
# The run's folder beside the carve's (the rules' folder; main sets it).
FOLDER = 'opd'
# opd_rules.json, the dense-teacher run, predates the teacher setting.
DENSE = {'kind': 'dense', 'finished_only': False}


def out(name):
    return mc.OUT / FOLDER / name


def fresh(name):
    if out(name).exists():
        raise ValueError(f'{name} exists; refusing to overwrite')
    out(name).parent.mkdir(parents=True, exist_ok=True)
    return out(name)


def provenance():
    return {'code_sha256': {n: sha(ROOT / n) for n in CODE}, 'rules_sha256': sha(RULES),
            'heal_rules_sha256': sha(sh.RULES), 'carve_rules_sha256': sha(sb.CARVE_RULES),
            'slim_rules_sha256': sha(sb.RULES), 'adapter_sha256': sha(sh.out('adapter.pt'))}


def setup():
    heal_rules, docs = sh.setup()
    return json.loads(RULES.read_text()), heal_rules, docs


def template(tok, question):
    return tok.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------- prompts

def stage_prompts(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    rules, _, docs = setup()
    cfg = rules['prompts']
    tok = AutoTokenizer.from_pretrained(mc.MODEL, local_files_only=True)
    _, short, _ = sb.load_tokenizers()

    def fits(question):
        text = template(tok, question)
        ids = short.encode(text)
        return short.decode(ids) == text and len(ids) <= cfg['max_prompt_tokens']

    def take(items):
        picked = []
        for item in items:
            if fits(item['question']):
                picked.append(item)
                if len(picked) == cfg['per_kind']:
                    return picked
        raise ValueError(f'Only {len(picked)} prompts fit')

    gsm8k = [{'id': f'gsm8k-train-{n}', 'kind': 'gsm8k', 'question': r['question'],
              'answer': float(r['answer'].split('####')[-1].strip().replace(',', ''))}
             for n, r in sh.paged('openai/gsm8k', 'main', cfg['gsm8k'])]
    # Chat items the benchmark uses (calibration or evaluation) stay out.
    used = {i for d in docs if d.get('source_domain', d['domain']) == 'chat' for i in d['items']}
    rows = pq.read_table(mc.CHAT, columns=['prompt_id', 'messages']).to_pylist()
    chat = [{'id': f'ultrachat-{r["prompt_id"][:16]}', 'kind': 'chat', 'question': r['messages'][0]['content']}
            for r in rows if r['prompt_id'] not in used and r['messages'] and r['messages'][0]['role'] == 'user']
    order = lambda items: sorted(items, key=lambda p: key(cfg['salt'], p['kind'], p['id']))
    gsm8k, chat = take(order(gsm8k)), take(order(chat))
    prompts = [p for pair in zip(gsm8k, chat) for p in pair]
    path = ROOT / cfg['file']
    if path.exists():
        raise ValueError(f'{path.name} exists; refusing to overwrite')
    save({'source': {'gsm8k': ['openai/gsm8k', 'main', 'train', cfg['gsm8k']], 'chat': mc.CHAT.name,
                     'salt': cfg['salt'], 'max_prompt_tokens': cfg['max_prompt_tokens']}, 'prompts': prompts}, path)
    log(f'{len(prompts)} prompts: {len(gsm8k)} GSM8K, {len(chat)} chat')


# ---------------------------------------------------------------- generation

def generation_tools(model, vocab):
    """The chat tokenizer; vocab gains the stop tokens (the first ends an answer, the second also
    pads) and the logit offsets that remove every row outside the 64K."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mc.MODEL, local_files_only=True)
    vocab.stop = [tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|endoftext|>')]
    vocab.removed = torch.full((model.config.vocab_size,), -torch.inf, device=mc.DEVICE)
    vocab.removed[vocab.kept] = 0
    return tok


def left_pad(prompts, pad):
    """Left-padded ids (B, P), their attention mask and each row's first real position."""
    width = max(map(len, prompts))
    ids = torch.full((len(prompts), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for r, p in enumerate(prompts):
        ids[r, width - len(p):] = torch.tensor(p)
        mask[r, width - len(p):] = 1
    first = torch.tensor([width - len(p) for p in prompts])
    return ids.to(mc.DEVICE), mask.to(mc.DEVICE), first.to(mc.DEVICE)


class FirstReal:
    """In a left-padded batch, each row's first real token runs every neuron, as position 0 of an
    unpadded sequence does (Slim.select). Padding never reaches the real tokens: attention masks it
    and the DeltaNets zero it."""

    def __init__(self, slim, first):
        self.slim, self.first = slim, first

    def __enter__(self):
        self.saved = vars(self.slim).get('select')
        inner = self.slim.select
        rows = torch.arange(len(self.first), device=self.first.device)

        def select(i, a, x):
            keep = inner(i, a, x)
            if a.shape[1] > 1:
                keep[rows, self.first] = True
            return keep

        self.slim.select = select
        return self

    def __exit__(self, *exc):
        if self.saved is None:
            vars(self.slim).pop('select')
        else:
            self.slim.select = self.saved


def loop_end(tokens, window, run):
    """How many of tokens come before a long repeat: up to and including the first token of the
    first run of `run` consecutive window-token spans that each occurred earlier; all of them
    without one."""
    seen, streak = set(), 0
    for end in range(window, len(tokens) + 1):
        span = tuple(tokens[end - window:end])
        streak = streak + 1 if span in seen else 0
        seen.add(span)
        if streak == run:
            return end - run - window + 2
    return len(tokens)


def until_stop(tokens, stop):
    return next((i for i, t in enumerate(tokens) if t in stop), None)


def stopping_classes():
    from transformers import LogitsProcessor, StoppingCriteria

    class Decode(LogitsProcessor):
        """The 64K kept rows; then greedy rows keep only their best token and the others the
        sampling filter (heal_probe.recommended) of the tokens after start."""

        def __init__(self, removed, start, greedy, sampling):
            self.removed, self.start, self.greedy, self.sampling = removed, start, greedy, sampling

        def __call__(self, input_ids, scores):
            scores = scores + self.removed
            if self.sampling is None:
                return scores
            filtered = hp.recommended(scores, input_ids[:, self.start:], self.sampling)
            best = scores.argmax(-1, keepdim=True)
            only = torch.full_like(scores, -torch.inf).scatter(1, best, scores.gather(1, best))
            return torch.where(self.greedy[:, None], only, filtered)

    class LoopStop(StoppingCriteria):
        """Every `every` tokens, ends the rows still running whose answer so far has a long repeat
        (loop_end); stopped records them."""

        def __init__(self, start, rows, stop, cfg):
            self.start, self.stop, self.cfg, self.calls = start, stop, cfg, 0
            self.stopped = [False] * rows

        def __call__(self, input_ids, scores, **kwargs):
            self.calls += 1
            if self.calls % self.cfg['every'] == 0:
                for r, tokens in enumerate(input_ids[:, self.start:].tolist()):
                    if not self.stopped[r] and until_stop(tokens, self.stop) is None:
                        self.stopped[r] = loop_end(tokens, self.cfg['window'], self.cfg['run']) < len(tokens)
            return torch.tensor(self.stopped, device=input_ids.device)

    return Decode, LoopStop


@torch.no_grad()
def generate(model, slim, vocab, prompts, cfg, student, greedy, sampling=None, seed=0, loop=None):
    """Batched answers to prompts (64K-tokenizer ids) by the student or the dense model, with
    cfg's state and length. greedy is True for all rows or one flag per row (the other rows sample
    through the sampling filter); loop, if given, ends rows that start repeating. Per row: its
    tokens without the stop token, the stop token (None if it did not stop by itself) and whether
    LoopStop ended it. Halves the batch on running out of memory."""
    from transformers import LogitsProcessorList, StoppingCriteriaList
    result = None
    try:
        Decode, LoopStop = stopping_classes()
        ids, mask, first = left_pad(prompts, vocab.stop[1])
        start = ids.shape[1]
        flags = torch.tensor([True] * len(prompts) if greedy is True else greedy, device=mc.DEVICE)
        criteria = StoppingCriteriaList([LoopStop(start, len(prompts), vocab.stop, loop)] if loop else [])
        # All-greedy batches decode exactly as heal_probe's greedy checks do.
        decode = (dict(do_sample=False, temperature=None, top_p=None, top_k=None) if greedy is True else
                  dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0))
        torch.manual_seed(seed)
        model.eval()
        slim.on = student
        try:
            with FirstReal(slim, first), sb.StoredState(model, [cfg['state']] * len(prompts), seed):
                result = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=cfg['max_new_tokens'],
                                        **decode, eos_token_id=vocab.stop, pad_token_id=vocab.stop[1],
                                        logits_processor=LogitsProcessorList([
                                            Decode(vocab.removed, start, flags, None if greedy is True else sampling)]),
                                        stopping_criteria=criteria)
        finally:
            slim.on = False
    except torch.cuda.OutOfMemoryError:
        if len(prompts) == 1:
            raise
    if result is None:
        # Split outside the except clause: its traceback holds the failed attempt's cache.
        gc.collect()
        torch.cuda.empty_cache()
        half = len(prompts) // 2
        log(f'Out of memory generating {len(prompts)} rows: two batches instead')
        part = lambda s: True if greedy is True else greedy[s]
        return (generate(model, slim, vocab, prompts[:half], cfg, student, part(slice(None, half)), sampling, seed, loop) +
                generate(model, slim, vocab, prompts[half:], cfg, student, part(slice(half, None)), sampling, seed + 1, loop))
    rows = []
    for r, tokens in enumerate(result[:, start:].tolist()):
        end = until_stop(tokens, vocab.stop)
        looped = bool(loop) and criteria[0].stopped[r]
        rows.append({'tokens': tokens[:end] if end is not None else tokens,
                     'stop': None if end is None or looped else tokens[end], 'loop_stopped': looped})
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def check_rows(rows, prompts, tok, arm, sample):
    """generate's rows as heal_probe's generation rows."""
    out_rows = []
    for row, p in zip(rows, prompts, strict=True):
        answer = sb.final_number(tok.decode(row['tokens']))
        out_rows.append({'id': p['id'], 'arm': arm, 'router': ARMS[arm][1], 'state': 'fp32', 'student': ARMS[arm][0],
                         'sample': sample, 'batched': True, 'tokens': row['tokens'], 'length': len(row['tokens']),
                         'finished': row['stop'] is not None, 'answer': answer,
                         'correct': answer is not None and abs(answer - p['answer']) < 1e-6})
    return out_rows


def check_arm(model, slim, vocab, tok, prompts, rules, arm, sample, batch=None):
    """One arm of the checks on the GSM8K check prompts: generations saved as gen_<arm>_<decoding>.jsonl.
    batch, if given, caps how many prompts generate together (during training, where the optimizer
    holds memory, a round's)."""
    cfg = rules['eval']
    student, routing = ARMS[arm]
    hp.route(slim, routing)
    t = time.time()
    batch = batch or len(prompts)
    try:
        rows = [row for b in range(0, len(prompts), batch) for row in generate(
            model, slim, vocab, [p['ids'] for p in prompts[b:b + batch]], cfg, student,
            [False] * len(prompts[b:b + batch]) if sample else True, cfg['sampling'] if sample else None,
            cfg['seed'] + sample + b)]
    finally:
        hp.route(slim, 'learned')
    rows = check_rows(rows, prompts, tok, arm, sample)
    name = f'{arm}_{"sampled" if sample else "greedy"}'
    fresh(f'gen_{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    summary = dict(hp.generations(rows, rules['loop']), seconds=time.time() - t)
    log(f'{name}: {summary["correct"]}/{summary["prompts"]} right, {summary["finished"]} finished, '
        f'{summary["loops"]} loops, {sum(r["length"] for r in rows) / summary["seconds"]:.0f} tokens/s')
    return name, summary


def check_prompts(rules, tok, vocab):
    cfg = rules['eval']
    prompts = json.loads((ROOT / cfg['prompts']).read_text())['prompts'][:cfg['count']]
    return [dict(p, ids=vocab.short.encode(template(tok, p['question']))) for p in prompts]


@torch.no_grad()
def padding_drift(model, slim, prompts):
    """Mean KL, over the real positions of each prompt, between its next-token distributions in a
    left-padded batch and alone, for the student and the dense model (fp32 state, as generated)."""
    drift = {}
    ids, mask, first = left_pad(prompts, 0)
    # Positions count from each row's first real token, as generate numbers them.
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    for student in (True, False):
        slim.on = student
        kls = []
        with FirstReal(slim, first), sb.StoredState(model, ['fp32'] * len(prompts), 0):
            batch = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False).logits.float()
        for r, p in enumerate(prompts):
            with sb.StoredState(model, ['fp32'], 0):
                alone = model(input_ids=sb.tensor([p]), use_cache=False).logits[0].float()
            kls.append(float(token_kl(alone, batch[r, first[r]:]).mean()))
        drift['student' if student else 'dense'] = float(np.mean(kls))
    slim.on = False
    return drift


# ---------------------------------------------------------------- training

class Teacher:
    """The model the student learns from: 'dense', or 'every_neuron', the healed student with every
    neuron and the healing run's adapter (slim.adapters[True]), which stays as healed while the
    student's own copy trains. Inside `with teacher:` the model runs as the teacher once slim.on is
    teacher.on; leaving restores the student's adapter and its learned routing."""

    def __init__(self, slim, kind):
        if kind not in ('dense', 'every_neuron'):
            raise ValueError(kind)
        self.slim, self.on = slim, kind == 'every_neuron'
        if self.on:
            healed = slim.adapters[True]['lora']
            params = slim.lora_parameters()
            # In lora_parameters' order: a then b of each adapted projection.
            frozen = [t for n, m in slim.adapted.items() if m.a is not None for t in healed[n]]
            self.frozen = [t.to(p.device, p.dtype) for t, p in zip(frozen, params, strict=True)]
            self.saved = [torch.empty_like(p) for p in params]

    @torch.no_grad()
    def __enter__(self):
        if self.on:
            hp.route(self.slim, 'all')
            for p, saved, frozen in zip(self.slim.lora_parameters(), self.saved, self.frozen):
                saved.copy_(p)
                p.copy_(frozen)
        return self

    @torch.no_grad()
    def __exit__(self, *exc):
        if self.on:
            for p, saved in zip(self.slim.lora_parameters(), self.saved):
                p.copy_(saved)
            hp.route(self.slim, 'learned')


def kl_chunk(hs, ht, head, reverse):
    """Summed KL between the student's and the teacher's kept-row distributions: the teacher's
    from the student's (forward) or the student's from the teacher's (reverse)."""
    s = F.linear(hs, head).float().log_softmax(-1)
    with torch.no_grad():
        t = F.linear(ht, head).float().log_softmax(-1)
    return (s.exp() * (s - t)).sum() if reverse else (t.exp() * (t - s)).sum()


def batch_rows(rows, pad):
    """Right-padded inputs (B, L) of each row's prompt and answer but its last token, and the mask
    of the positions whose next token is one of the answer's first `count` tokens."""
    width = max(len(r['prompt']) + r['count'] - 1 for r in rows)
    x = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.bool)
    for b, r in enumerate(rows):
        seq = r['prompt'] + r['answer'][:r['count']]
        x[b, :len(seq) - 1] = torch.tensor(seq[:-1])
        mask[b, len(r['prompt']) - 1:len(seq) - 1] = True
    return x.to(mc.DEVICE), mask.to(mc.DEVICE)


def micro_step(model, slim, vocab, teacher, rows, reverse, chunk, scale):
    """Teacher and student passes over rows; backward of scale times the summed KL (reverse or
    forward) at their counted positions, which it returns. Padding sits after each row, so the
    causal passes need no mask."""
    x, mask = batch_rows(rows, vocab.stop[1])
    model.eval()
    with torch.no_grad(), teacher:
        slim.on = teacher.on
        ht = model.model(input_ids=x, use_cache=False).last_hidden_state[mask]
    slim.on = True
    model.train()
    try:
        hs = model.model(input_ids=x, use_cache=False).last_hidden_state[mask]
        kl = 0
        for c in range(0, len(hs), chunk):
            kl = kl + checkpoint(kl_chunk, hs[c:c + chunk], ht[c:c + chunk], vocab.head, reverse, use_reentrant=False)
        (kl * scale).backward()
    finally:
        slim.on = False
        model.eval()
    return float(kl.detach())


def pattern(n):
    """The student's rows: greedy, greedy, sampled, sampled, ..., so the round's alternating GSM8K
    and chat prompts get both decodings."""
    return [(j // 2) % 2 == 0 for j in range(n)]


def factor(step, done, cfg, since=0):
    """Learning-rate factor: linear warm-up over steps, then a cosine to the floor over the share
    done of the time budget. A run resumed at step since warms up again to the cosine, as its
    optimizer starts afresh."""
    if step < cfg['warmup_steps']:
        return (step + 1) / cfg['warmup_steps']
    cosine = cfg['lr_floor'] + (1 - cfg['lr_floor']) * 0.5 * (1 + math.cos(math.pi * min(1.0, done)))
    return min(1.0, (step - since + 1) / cfg['warmup_steps']) * cosine


def resume_point(cfg):
    """A stopped train stage's rounds and steps up to its last checkpoint, its halfway check if it
    made one, and the share of its time budget used by then, read back from its last learning rate
    (past the warm-up, the cosine)."""
    progress = json.loads(out('progress.json').read_text())
    every = cfg['checkpoint_every']
    kept = progress.get('checkpoint', len(progress['rounds']) // every * every)
    steps = [s for s in progress['steps'] if s['round'] < kept]
    if not kept or steps[-1]['step'] <= cfg['warmup_steps']:
        raise ValueError('Nothing to resume: no checkpoint past the warm-up')
    f = (steps[-1]['lr'] / cfg['lr'] - cfg['lr_floor']) / (1 - cfg['lr_floor'])
    return progress['rounds'][:kept], steps, progress['mid'], math.acos(min(1.0, max(-1.0, 2 * f - 1))) / math.pi


def kind_stats(rows, tok):
    """Share finished, share cut for a loop, mean length and GSM8K share right, of some answers."""
    if not rows:
        return None
    right = lambda r: (lambda a: a is not None and abs(a - r['p']['answer']) < 1e-6)(sb.final_number(tok.decode(r['tokens'])))
    gsm8k = [right(r) for r in rows if r['p']['kind'] == 'gsm8k']
    return {'rows': len(rows), 'finished': float(np.mean([r['stop'] is not None for r in rows])),
            'loops': float(np.mean([r['count'] < len(r['answer']) or r['loop_stopped'] for r in rows])),
            'mean_length': float(np.mean([len(r['tokens']) for r in rows])),
            'gsm8k_right': float(np.mean(gsm8k)) if gsm8k else None}


def start_state(cfg):
    """The adapter the student starts from: train.start's (a finished run's adapter.pt beside the
    carve's outputs, checked against that run's train.json), or None for the healing run's."""
    start = cfg.get('start')
    if start is None:
        return None
    path = mc.OUT / start['adapter']
    if sha(path) != json.loads((mc.OUT / start['record']).read_text())['adapter_out_sha256']:
        raise ValueError(f'{start["adapter"]} is not its run\'s adapter')
    return torch.load(path, weights_only=True, map_location='cpu')


def text_order(count, seed, used):
    """The healing sequences in the order this run takes them: the ones the healing run's
    permutation never reached first, then the ones it trained on."""
    order = np.random.default_rng(seed).permutation(count)
    return np.concatenate([order[used:], order[:used]])


def healing_text(heal_rules):
    """The healing run's sequences (ids of the 64K rows) and the order to take them in."""
    record = json.loads(sh.out('data.json').read_text())
    if sha(sh.out('data.npz')) != record['data_sha256']:
        raise ValueError('The healing data changed after the healing run')
    seqs = np.load(sh.out('data.npz'))['ids']
    cfg = heal_rules['train']
    used = json.loads(sh.out('heal.json').read_text())['steps'] * cfg['batch_sequences']
    return seqs, text_order(len(seqs), cfg['seed'], used)


def text_rows(seqs):
    """Training rows of healing text: every position after the first counts, toward the dense model."""
    return [{'prompt': [int(s[0])], 'answer': [int(t) for t in s[1:]], 'count': len(s) - 1, 'reverse': False,
             'text': True} for s in seqs]


def parts(rows):
    """A step's rows by loss: forward KL on the teacher's answers, forward KL from the dense model
    on healing text, reverse KL on the student's answers."""
    return {'forward': [r for r in rows if not r['reverse'] and not r.get('text')],
            'text': [r for r in rows if r.get('text')],
            'reverse': [r for r in rows if r['reverse']]}


def train_rows(rows, prompts, reverse, loop, greedy=None):
    """Training rows: prompt, answer (with its stop token) and how many answer tokens count; the
    student's own answers count only up to their first long repeat."""
    for j, (r, p) in enumerate(zip(rows, prompts, strict=True)):
        r.update(p=p, prompt=p['ids'], answer=r['tokens'] + ([r['stop']] if r['stop'] is not None else []),
                 reverse=reverse, greedy=bool(greedy and greedy[j]))
        r['count'] = loop_end(r['answer'], loop['window'], loop['run']) if reverse else len(r['answer'])
    return [r for r in rows if r['count'] > 0]


def stage_train(args):
    rules, heal_rules, docs = setup()
    cfg, rcfg = rules['train'], rules['rollout']
    t0 = time.time()
    rounds, steps, mid, done0 = resume_point(cfg) if args.resume else ([], [], None, 0.0)
    resumed = dict(rounds=len(rounds), step=rounds[-1]['steps'], done=done0,
                   checkpoint_sha256=sha(out('latest.pt'))) if args.resume else None
    # A resumed run has the rest of the time budget.
    stop_at = min(t0 + 3600 * cfg['max_hours'] * (1 - done0),
                  float(os.environ.get('MANIFOLD_JOB_DEADLINE', 'inf')) - 60 * cfg['reserve_minutes'])
    model, slim, vocab, _ = hp.load_healed(heal_rules, docs)
    tok = generation_tools(model, vocab)
    prompts = json.loads((ROOT / rules['prompts']['file']).read_text())['prompts']
    prompts = [dict(p, ids=vocab.short.encode(template(tok, p['question']))) for p in prompts]
    checks = check_prompts(rules, tok, vocab)
    tcfg = cfg.get('teacher', DENSE)
    teacher, dense = Teacher(slim, tcfg['kind']), Teacher(slim, 'dense')
    start = start_state(cfg)
    if start is not None:
        slim.load(start)
    if resumed:
        slim.load(torch.load(out('latest.pt'), weights_only=True, map_location='cpu'))
    text = cfg.get('text')
    if text:
        seqs, order = healing_text(heal_rules)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    params = slim.lora_parameters()
    optimizer = torch.optim.AdamW(params, lr=cfg['lr'], betas=tuple(cfg['betas']), weight_decay=0.0)
    n, k, micro = rcfg['per_round'], cfg['steps_per_round'], cfg['micro_batch']
    step, taken, checkpoint = 0, 0, len(rounds)
    cursor = cfg['start']['first_prompt'] if start is not None else 0
    if resumed:
        step, cursor = resumed['step'], rounds[-1]['prompts']
        taken = len(rounds) * text['per_round'] if text else 0
        kept = [line for line in out('rollouts.jsonl').open() if json.loads(line)['round'] < len(rounds)]
        out('rollouts.jsonl').write_text(''.join(kept))
        rollouts = out('rollouts.jsonl')
    else:
        rollouts = fresh('rollouts.jsonl')
    since = step
    begin = time.time()
    done = lambda: done0 + (1 - done0) * (time.time() - begin) / (stop_at - begin)
    with deadline((stop_at - time.time()) / 3600 + 0.25):
        while len(rounds) < (cfg['max_rounds'] or math.inf):
            longest = max((r['seconds'] for r in rounds), default=60 * cfg['first_round_minutes'])
            if time.time() + 1.2 * longest > stop_at or cursor + 2 * n > len(prompts):
                break
            if mid is None and done() >= cfg['mid_check']:
                name, mid = check_arm(model, slim, vocab, tok, checks, rules, 'mid', False, n)
                mid.update(round=len(rounds), step=step)
            t = time.time()
            asked, cursor = prompts[cursor:cursor + 2 * n], cursor + 2 * n
            seed = cfg['seed'] + len(rounds)
            greedy = pattern(n)
            with teacher:
                answers = generate(model, slim, vocab, [p['ids'] for p in asked[:n]], rcfg, teacher.on, [False] * n,
                                   rcfg['sampling'], seed)
            student = generate(model, slim, vocab, [p['ids'] for p in asked[n:]], rcfg, True, greedy,
                               rcfg['sampling'], seed, rcfg['loop'])
            generated = time.time() - t
            answers = train_rows(answers, asked[:n], False, rcfg['loop'])
            student = train_rows(student, asked[n:], True, rcfg['loop'], greedy)
            used = [r for r in answers if r['stop'] is not None or not tcfg['finished_only']]
            reading = []
            if text:
                reading = text_rows(seqs[order[np.arange(taken, taken + text['per_round']) % len(order)]])
                taken += text['per_round']
            with rollouts.open('a') as stream:
                for r in answers + student:
                    stream.write(json.dumps({'round': len(rounds), 'id': r['p']['id'], 'reverse': r['reverse'],
                                             'greedy': r['greedy'], 'tokens': r['tokens'], 'stop': r['stop'],
                                             'loop_stopped': r['loop_stopped'], 'count': r['count'],
                                             'trained': r['reverse'] or any(r is u for u in used)}) + '\n')
            for j in range(k):
                chosen = used[j::k] + student[j::k] + reading[j::k]
                positions = sum(r['count'] for r in chosen)
                if not positions:
                    continue
                lr = factor(step, done(), cfg, since)
                for group in optimizer.param_groups:
                    group['lr'] = cfg['lr'] * lr
                ts = time.time()
                while True:
                    sums = dict.fromkeys(parts([]), 0.0)
                    try:
                        for name, part in parts(chosen).items():
                            part = sorted(part, key=lambda r: len(r['prompt']) + r['count'])
                            for m in range(0, len(part), micro):
                                sums[name] += micro_step(model, slim, vocab, dense if name == 'text' else teacher,
                                                         part[m:m + micro], name == 'reverse', cfg['loss_chunk'],
                                                         1 / positions)
                        break
                    except torch.cuda.OutOfMemoryError:
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        if micro == 1:
                            raise
                        micro //= 2
                        log(f'Out of memory: micro-batch now {micro}')
                norm = float(torch.nn.utils.clip_grad_norm_(params, cfg['clip']))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                count = {name: sum(r['count'] for r in part) for name, part in parts(chosen).items()}
                steps.append({'step': step, 'round': len(rounds), 'positions': positions, 'lr': cfg['lr'] * lr,
                              'forward_kl': sums['forward'] / max(1, count['forward']),
                              'reverse_kl': sums['reverse'] / max(1, count['reverse']),
                              'text_kl': sums['text'] / count['text'] if count['text'] else None,
                              'grad_norm': norm, 'seconds': time.time() - ts})
            this = [s for s in steps if s['round'] == len(rounds)]
            mean = lambda name: (lambda v: float(np.mean(v)) if v else None)([s[name] for s in this if s[name] is not None])
            stats = {'round': len(rounds), 'steps': step, 'prompts': cursor,
                     'positions': {'teacher': sum(r['count'] for r in used), 'student': sum(r['count'] for r in student),
                                   'text': sum(r['count'] for r in reading)},
                     'teacher': kind_stats(answers, tok), 'teacher_trained': len(used),
                     'greedy': kind_stats([r for r in student if r['greedy']], tok),
                     'sampled': kind_stats([r for r in student if not r['greedy']], tok),
                     'forward_kl': mean('forward_kl'), 'reverse_kl': mean('reverse_kl'), 'text_kl': mean('text_kl'),
                     'generate_seconds': generated, 'seconds': time.time() - t,
                     'memory_gb': torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0}
            rounds.append(stats)
            share = lambda part, k: f'{stats[part][k]:.2f}' if stats[part] else '-'
            log(f'round {stats["round"]}: teacher finished {share("teacher", "finished")} ({len(used)} trained); '
                f'student greedy loops '
                f'{share("greedy", "loops")}, finished {share("greedy", "finished")}; sampled loops '
                f'{share("sampled", "loops")}, finished {share("sampled", "finished")}; forward KL '
                f'{stats["forward_kl"] or 0:.4f}, reverse KL {stats["reverse_kl"] or 0:.4f}' +
                (f', text KL {stats["text_kl"] or 0:.4f}' if text else '') + '; '
                f'{sum(stats["positions"].values())} positions; generate {generated:.0f} s, round {stats["seconds"]:.0f} s')
            if len(rounds) % cfg['checkpoint_every'] == 0:
                torch.save(slim.state(torch.bfloat16), out('latest.pt.tmp'))
                out('latest.pt.tmp').replace(out('latest.pt'))
                checkpoint = len(rounds)
            save({'rounds': rounds, 'steps': steps, 'mid': mid, 'checkpoint': checkpoint}, out('progress.json'))
    torch.save(slim.state(torch.bfloat16), fresh('adapter.pt'))
    save(dict(provenance(), prompts_sha256=sha(ROOT / rules['prompts']['file']), adapter_out_sha256=sha(out('adapter.pt')),
              start_adapter_sha256=sha(mc.OUT / cfg['start']['adapter']) if start is not None else None,
              resumed=resumed, text_sequences=taken, rounds=rounds, steps=steps, mid=mid, prompts_used=cursor,
              final_micro_batch=micro, seconds=time.time() - t0), fresh('train.json'))
    log(f'{len(rounds)} rounds, {step} steps, {sum(sum(r["positions"].values()) for r in rounds)} positions')


# ---------------------------------------------------------------- checks

def drift(model, slim, vocab, docs, heal_rules, name):
    rows = sh.evaluate(model, slim, vocab, sb.docs_of(docs, 'eval'))
    fresh(f'eval_{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    summary = sh.summarize(rows, heal_rules['data']['held_out'])
    log(f'{name}: KL {summary["kl"]:.3f} (in-mix {summary["in_mix"]["kl"]:.3f}, held-out '
        f'{summary["held_out"]["kl"]:.3f}), 64K bits per byte {summary["bpb_student"]:.4f}, '
        f'agreement {summary["agree"]:.3f}')
    return rows, summary


def stage_before(args):
    rules, heal_rules, docs = setup()
    cfg = rules['eval']
    t0 = time.time()
    model, slim, vocab, same = hp.load_healed(heal_rules, docs)
    tok = generation_tools(model, vocab)
    checks = check_prompts(rules, tok, vocab)
    last = json.loads(sh.out('heal.json').read_text())['milestones'][-1]
    with deadline(cfg['before_hours']):
        _, healed = drift(model, slim, vocab, docs, heal_rules, 'healed')
        check = {'kl': healed['kl'] - last['kl'], 'bpb': healed['bpb_student'] / last['bpb_student'] - 1}
        if abs(check['kl']) > cfg['check']['kl'] or abs(check['bpb']) > cfg['check']['bpb']:
            raise RuntimeError(f'The rebuilt student does not reproduce the healed one: {check}')
        padding = padding_drift(model, slim, [p['ids'] for p in checks[:cfg['padding_prompts']]])
        log(f'padding drift: student {padding["student"]:.2e}, dense {padding["dense"]:.2e}')
        if any(padding[k] > v for k, v in cfg['padding_limit'].items()):
            raise RuntimeError(f'Padding changes the outputs: {padding}')
        arms = dict(check_arm(model, slim, vocab, tok, checks, rules, arm, sample)
                    for arm in cfg.get('before_arms', REFERENCES) for sample in (False, True))
        start, started = start_state(rules['train']), None
        if start is not None:
            slim.load(start)
            _, started = drift(model, slim, vocab, docs, heal_rules, 'start')
            arms |= dict(check_arm(model, slim, vocab, tok, checks, rules, 'start', sample) for sample in (False, True))
    save(dict(provenance(), factors_identical=same, reproduces_healed=check, healed=healed, padding=padding,
              start=started, arms=arms, seconds=time.time() - t0), fresh('before.json'))


def stage_after(args):
    rules, heal_rules, docs = setup()
    cfg = rules['eval']
    record = json.loads(out('train.json').read_text())
    if sha(out('adapter.pt')) != record['adapter_out_sha256']:
        raise ValueError('adapter.pt is not the distillation\'s')
    t0 = time.time()
    model, slim, vocab, _ = hp.load_healed(heal_rules, docs)
    slim.adapters[True] = torch.load(out('adapter.pt'), weights_only=True, map_location='cpu')
    hp.heal(slim, True)
    tok = generation_tools(model, vocab)
    checks = check_prompts(rules, tok, vocab)
    with deadline(cfg['after_hours']):
        rows, distilled = drift(model, slim, vocab, docs, heal_rules, 'distilled')
        difference, from_start = kl_difference(rows, 'healed'), kl_difference(rows, 'start')
        arms = dict(check_arm(model, slim, vocab, tok, checks, rules, 'distilled', sample) for sample in (False, True))
    save(dict(provenance(), distilled=distilled, difference=difference, difference_from_start=from_start, arms=arms,
              seconds=time.time() - t0), fresh('after.json'))


def kl_difference(rows, name):
    """The distilled student's paired difference in benchmark KL from the before stage's eval_<name>.jsonl,
    if there is one."""
    if not out(f'eval_{name}.jsonl').exists():
        return None
    base = [json.loads(line) for line in out(f'eval_{name}.jsonl').open()]
    text = lambda rs: [r for r in rs if r['domain'] != 'copy']
    report = json.loads(mc.RULES.read_text())['report']
    diff, low, high = paired_interval([r['kl']['student'] for r in text(rows)], [r['kl']['student'] for r in text(base)],
                                      1.0, report['draws'], report['seed'], [r['domain'] for r in text(rows)])
    log(f'distilled - {name} KL: {diff:+.4f} ({low:+.4f} to {high:+.4f})')
    return {'difference': diff, 'low': low, 'high': high}


# ---------------------------------------------------------------- report

def fisher_less(a, n, b, m):
    """One-sided Fisher exact p: the chance that the first of two groups (n and m) has a or fewer
    of the a + b events."""
    k = a + b
    return sum(math.comb(k, x) * math.comb(n + m - k, n - x) for x in range(a + 1)) / math.comb(n + m, n)


def stage_report(args):
    rules, _, _ = setup()
    read = lambda name: json.loads(out(name).read_text()) if out(name).exists() else None
    arms = {path.stem[4:]: hp.generations([json.loads(line) for line in path.open()], rules['loop'])
            for path in sorted(out('.').glob('gen_*.jsonl'))}
    tests = {}
    for new, base in (('distilled_greedy', 'healed_greedy'), ('distilled_sampled', 'healed_sampled'),
                      ('mid_greedy', 'healed_greedy'), ('distilled_greedy', 'start_greedy'),
                      ('distilled_sampled', 'start_sampled'), ('mid_greedy', 'start_greedy')):
        if new in arms and base in arms:
            a, b = arms[new], arms[base]
            tests[f'{new}-{base}'] = {
                'loops': fisher_less(a['loops'], a['prompts'], b['loops'], b['prompts']),
                'unfinished': fisher_less(a['prompts'] - a['finished'], a['prompts'], b['prompts'] - b['finished'], b['prompts']),
                'wrong': fisher_less(a['prompts'] - a['correct'], a['prompts'], b['prompts'] - b['correct'], b['prompts'])}
    train = read('train.json') or read('progress.json')
    curve = None
    if train and train['rounds']:
        quarters = np.array_split(np.arange(len(train['rounds'])), 4)
        mean = lambda q, k: float(np.mean([train['rounds'][i][k]['loops'] for i in q if train['rounds'][i][k]]))
        curve = [{k: mean(q, k) for k in ('greedy', 'sampled')} | {'rounds': [int(q[0]), int(q[-1])]}
                 for q in quarters if len(q)]
    report = {'before': read('before.json'), 'after': read('after.json'), 'generate': arms, 'tests': tests,
              'on_policy_loops_by_quarter': curve}
    save(dict(provenance(), **report), fresh('report.json'))
    for name, g in arms.items():
        log(f'{name}: {g["correct"]}/{g["prompts"]} right, {g["finished"]} finished, {g["loops"]} loops, '
            f'finished median length {g["finished_median_length"]}')
    for name, t in tests.items():
        log(f'{name}: one-sided p loops {t["loops"]:.3f}, unfinished {t["unfinished"]:.3f}, wrong {t["wrong"]:.3f}')


def main():
    global RULES, FOLDER
    stages = {'prompts': stage_prompts, 'before': stage_before, 'train': stage_train, 'after': stage_after,
              'report': stage_report}
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=stages)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--resume', action='store_true', help='train: carry a stopped train stage on from latest.pt')
    ap.add_argument('--model', type=Path, default=mc.MODEL)
    ap.add_argument('--rules', type=Path, default=RULES)
    ap.add_argument('--heal-rules', type=Path, default=sh.RULES)
    ap.add_argument('--slim-rules', type=Path, default=sb.RULES)
    ap.add_argument('--carve-rules', type=Path, default=sb.CARVE_RULES)
    args = ap.parse_args()
    mc.DEVICE, mc.MODEL = args.device, args.model
    RULES, sh.RULES = args.rules.resolve(), args.heal_rules.resolve()
    sb.RULES, sb.CARVE_RULES = args.slim_rules.resolve(), args.carve_rules.resolve()
    FOLDER = json.loads(RULES.read_text()).get('folder', FOLDER)
    stages[args.stage](args)


if __name__ == '__main__':
    main()
