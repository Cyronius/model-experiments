"""Frozen baseline or final evaluation: corpus CE/KL per split and domain, and downstream tasks.

Downstream scoring is a fixed custom harness; do not compare its numbers with lm-eval tables.
- ARC-Easy, ARC-Challenge (validation) and the frozen MMLU subset: zero-shot full-choice
  continuations, mean log-probability per answer token.
- GSM8K test prefix: greedy chat-template generation with thinking off, scored on the last
  \\boxed{} (else the last number) against the reference answer.
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from canary import CanaryFailure, check as canary_check
from engine import counted, logit_loss, precision, stream
from prepare_data import last_boxed
from pruning import sha256
from run import load_model, save_json

SPLITS = ('validation', 'test', 'controller', 'test_in_domain')
MC_TASKS = ('arc_easy', 'arc_challenge', 'mmlu')
GSM8K_PROMPT = '{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.'


@torch.no_grad()
def corpus_metrics(model, teacher, rows, labels, names, device, amp):
    sums = {}
    for row, label in zip(rows, labels):
        ids = torch.tensor(row[None], device=device, dtype=torch.long)
        with precision(model, amp):
            h = stream(model, ids)[-1]
            target = h if model is teacher else stream(teacher, ids)[-1]
            ce, kl = logit_loss(model, teacher, h, target, ids)
        n = ids[:, 1:].numel()
        for key in ('all', names[int(label)]):
            s = sums.setdefault(key, [0.0, 0.0, 0])
            s[0] += float(ce) * n
            s[1] += float(kl) * n
            s[2] += n
    out = {k: dict(ce=c / n, perplexity=math.exp(min(c / n, 80)), teacher_kl=k_ / n, tokens=n)
           for k, (c, k_, n) in sums.items()}
    return dict(out.pop('all'), domains=out)


@torch.no_grad()
def multiple_choice(model, tokenizer, examples, device, amp):
    """Zero-shot full-choice continuations, mean log-probability per answer token.

    Prompt and answer are tokenized separately, then concatenated; no truncation.
    """
    rows = []
    for example in examples:
        prompt = tokenizer.encode(f"Question: {example['question']}\nAnswer:", add_special_tokens=False)
        scores = []
        for choice in example['choices']:
            continuation = tokenizer.encode(' ' + str(choice), add_special_tokens=False)
            if not continuation or not prompt:
                raise ValueError('Empty prompt or answer')
            ids = torch.tensor([prompt + continuation], dtype=torch.long, device=device)
            with precision(model, amp):
                h = stream(model, ids)[-1]
                selected = model.model.norm(h)[:, len(prompt)-1:-1]
                logits = model.lm_head(selected).float()
                score = -F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, len(prompt):].reshape(-1))
            scores.append(float(score))
        prediction = int(np.argmax(scores))
        rows.append(dict(id=example['id'], scores=scores, prediction=prediction,
                         answer=example['answer'], correct=prediction == example['answer']))
    return dict(harness='custom zero-shot answer-token-normalized log-likelihood',
                accuracy=sum(x['correct'] for x in rows) / len(rows), n=len(rows), examples=rows)


def gsm8k_answer(text):
    boxed = last_boxed(text)
    candidate = boxed if boxed is not None else (re.findall(r'-?\d[\d,]*\.?\d*', text) or [None])[-1]
    if candidate is None:
        return None
    candidate = candidate.replace(',', '').replace('$', '').strip().rstrip('.')
    try:
        value = float(candidate)
    except ValueError:
        return candidate
    if not math.isfinite(value):  # e.g. a damaged model's 400-digit number overflows to inf
        return candidate
    return str(int(value)) if value == int(value) else str(value)


@torch.no_grad()
def gsm8k(model, tokenizer, examples, device, amp, max_new_tokens):
    end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    rows = []
    for example in examples:
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': GSM8K_PROMPT.format(question=example['question'])}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = torch.tensor([tokenizer.encode(text, add_special_tokens=False)], device=device)
        with precision(model, amp):
            out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                 max_new_tokens=max_new_tokens, eos_token_id=end, pad_token_id=end)
        new = out[0, ids.shape[1]:].tolist()
        finished = end in new
        reply = tokenizer.decode(new[:new.index(end)] if finished else new, skip_special_tokens=True)
        predicted = gsm8k_answer(reply)
        rows.append(dict(id=example['id'], reply=reply, finished=finished, predicted=predicted,
                         answer=example['answer'], correct=predicted == example['answer']))
    return dict(harness=f'greedy, chat template, thinking off, {max_new_tokens} new tokens, last boxed else last number',
                accuracy=sum(x['correct'] for x in rows) / len(rows), finished=sum(x['finished'] for x in rows),
                n=len(rows), examples=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--teacher', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--sequences', type=int, default=32, help='Per corpus split (the whole frozen split by default)')
    ap.add_argument('--mc-examples', type=int, default=None, help='Per multiple-choice task; default all frozen')
    ap.add_argument('--gsm8k-examples', type=int, default=128)
    ap.add_argument('--gsm8k-tokens', type=int, default=1024)
    ap.add_argument('--canary', type=Path, help='CPU FP32 canary reference (required on a GPU)')
    ap.add_argument('--canary-gpu', type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    amp = str(args.device).startswith('cuda')
    if amp and args.canary is None:
        raise ValueError('GPU evaluations need --canary')
    torch.set_num_threads(2)
    dtype = torch.bfloat16 if amp else torch.float32
    data_meta = json.loads((args.data / 'manifest.json').read_text())
    for name, digest in data_meta['files_sha256'].items():
        if sha256(args.data / name) != digest:
            raise ValueError(f'Data hash mismatch: {name}')
    started = time.monotonic()
    teacher = load_model(args.teacher, args.device, dtype)
    canary = {}

    def run_canary(when):
        if args.canary is not None:
            try:
                canary[when] = canary_check(teacher, args.teacher, args.data, args.canary, args.device, amp, args.canary_gpu)
            except CanaryFailure as failure:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                save_json(args.output.with_suffix('.canary-failed.json'), dict(when=when, record=json.loads(str(failure))))
                raise SystemExit(3)

    run_canary('start')
    model = teacher if args.model.resolve() == args.teacher.resolve() else load_model(args.model, args.device, dtype)
    arrays = np.load(args.data / 'tokens.npz', allow_pickle=False)
    names = data_meta['domains']
    results = {}
    for split in SPLITS:
        metric, cost = counted(lambda: corpus_metrics(model, teacher, arrays[split][:args.sequences],
                                                      arrays[f'{split}_domain'][:args.sequences], names, args.device, amp))
        results[split] = dict(**metric, accounted_flops=cost)
        print(json.dumps({split: {k: v for k, v in metric.items() if k != 'domains'}}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    examples = [json.loads(line) for line in (args.data / 'downstream.jsonl').read_text().splitlines()]
    results['downstream'] = {}
    for task in MC_TASKS:
        chosen = [e for e in examples if e['task'] == task][:args.mc_examples]
        result, cost = counted(lambda: multiple_choice(model, tokenizer, chosen, args.device, amp))
        results['downstream'][task] = dict(**result, accounted_flops=cost)
        print(json.dumps({task: result['accuracy'], 'n': result['n']}), flush=True)
    if args.gsm8k_examples:
        chosen = [e for e in examples if e['task'] == 'gsm8k'][:args.gsm8k_examples]
        results['downstream']['gsm8k'] = gsm8k(model, tokenizer, chosen, args.device, amp, args.gsm8k_tokens)
        print(json.dumps({'gsm8k': results['downstream']['gsm8k']['accuracy']}), flush=True)
    run_canary('end')
    results['provenance'] = dict(model=str(args.model), teacher=str(args.teacher), canary=canary,
                                 data_manifest_sha256=sha256(args.data / 'manifest.json'),
                                 code_sha256=sha256(__file__), elapsed_seconds=time.monotonic() - started,
                                 arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                 model_sha256={p.name: sha256(p) for p in sorted(args.model.glob('*.safetensors'))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, results)


if __name__ == '__main__':
    main()
