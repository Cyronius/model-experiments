"""Full fine-tune of an LFM chat model on the documentation pairs (data/out/lm-doc-v1/train.jsonl).

The prompt is the zero-shot prompt of generate.py (its SYSTEM and user message, no examples, thinking
off); the loss is on the reference and the closing <|im_end|> only. About 4% of the training documents
are held out (by document, hash of the group) to pick the epoch; the 679 dev pairs are not touched.
fp32 weights and AdamW, bf16 autocast. Each epoch is saved (bf16, with the tokenizer and remote code)
under --out/epoch-N; --out/best points at the epoch with the lowest held-out loss. A rerun resumes after
the last saved epoch (fresh optimizer state; the LR schedule continues).

Run with target/manifold-rocm/env/bin/python and PYTHONPATH=data/out/lm-overlay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model_lm'))
from generate import DATA, messages  # noqa: E402


def held_out(group: str) -> bool:
    return int.from_bytes(hashlib.sha256(('lm-ft-heldout/' + group).encode()).digest()[:8], 'big') / 2**64 < 0.04


def encode(tok, row, max_len):
    prompt = tok.apply_chat_template(messages([], row['mode'], row['src']), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    p = tok(prompt, add_special_tokens=False)['input_ids']
    t = tok(row['ste'] + '<|im_end|>', add_special_tokens=False)['input_ids']
    ids = (p + t)[:max_len]
    labels = ([-100] * len(p) + t)[:max_len]
    return ids, labels


def batches(items, size, pad, shuffle, seed):
    order = list(range(len(items)))
    if shuffle:
        random.Random(seed).shuffle(order)
        # sort within chunks of 50 batches to cut padding, keep randomness across chunks
        chunk = size * 50
        order = [i for c in range(0, len(order), chunk)
                 for i in sorted(order[c:c + chunk], key=lambda i: len(items[i][0]))]
        groups = [order[b:b + size] for b in range(0, len(order), size)]
        random.Random(seed + 1).shuffle(groups)
    else:
        order.sort(key=lambda i: len(items[i][0]))
        groups = [order[b:b + size] for b in range(0, len(order), size)]
    for g in groups:
        n = max(len(items[i][0]) for i in g)
        ids = torch.full((len(g), n), pad)
        lab = torch.full((len(g), n), -100)
        att = torch.zeros((len(g), n), dtype=torch.long)
        for k, i in enumerate(g):
            a, b = items[i]
            ids[k, :len(a)] = torch.tensor(a)
            lab[k, :len(b)] = torch.tensor(b)
            att[k, :len(a)] = 1
        yield ids, lab, att


def loss_on(model, items, size, pad, device):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
        for ids, lab, att in batches(items, size, pad, False, 0):
            ids, lab, att = ids.to(device), lab.to(device), att.to(device)
            logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            tgt = lab[:, 1:]
            l = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                                  ignore_index=-100, reduction='sum')
            tot += l.item()
            n += (tgt != -100).sum().item()
    model.train()
    return tot / n


def save(model, tok, src: Path, dest: Path):
    tmp = dest.with_name(dest.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    model.to(torch.bfloat16).save_pretrained(tmp)
    model.to(torch.float32)
    tok.save_pretrained(tmp)
    for f in list(src.glob('*.py')) + list(src.glob('chat_template.jinja')) + [src / 'generation_config.json']:
        if f.exists() and not (tmp / f.name).exists():
            shutil.copy(f, tmp / f.name)
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--warmup', type=float, default=0.05)
    ap.add_argument('--max-len', type=int, default=512)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='smoke test: first N training rows')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = [json.loads(l) for l in (DATA / 'train.jsonl').open()]
    if args.limit:
        rows = rows[:args.limit]
    train = [encode(tok, r, args.max_len) for r in rows if not held_out(r['group'])]
    val = [encode(tok, r, args.max_len) for r in rows if held_out(r['group'])]
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = steps_per_epoch * args.epochs
    warm = max(1, int(total * args.warmup))

    start = 0
    log_path = args.out / 'log.jsonl'
    history = [json.loads(l) for l in log_path.open()] if log_path.exists() else []
    done = [h['epoch'] for h in history if (args.out / f"epoch-{h['epoch']}").exists()]
    load_from = args.model
    if done:
        start = max(done)
        load_from = args.out / f'epoch-{start}'
    model = AutoModelForCausalLM.from_pretrained(load_from, trust_remote_code=True, dtype=torch.float32)
    model.to(args.device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    for _ in range(start * steps_per_epoch):
        sched.step()
    meta = dict(model=str(args.model), train=len(train), heldout=len(val), epochs=args.epochs, lr=args.lr,
                batch=args.batch, steps_per_epoch=steps_per_epoch, max_len=args.max_len,
                truncated=sum(len(a) == args.max_len for a, _ in train))
    (args.out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(meta, flush=True)
    if not history:
        base = loss_on(model, val, args.batch, tok.pad_token_id, args.device)
        history.append(dict(epoch=0, heldout_loss=base))
        log_path.write_text(json.dumps(history[0]) + '\n')
        print('epoch 0 held-out loss', round(base, 4), flush=True)
    t0 = time.time()
    for epoch in range(start + 1, args.epochs + 1):
        run, n = 0.0, 0
        for step, (ids, lab, att) in enumerate(batches(train, args.batch, tok.pad_token_id, True, args.seed + epoch)):
            ids, lab, att = ids.to(args.device), lab.to(args.device), att.to(args.device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.device == 'cuda'):
                logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1),
                                                     ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at epoch {epoch} step {step} (GPU fault?)')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run += loss.item()
            n += 1
            if step % 50 == 0:
                print(f'epoch {epoch} step {step}/{steps_per_epoch} loss {run / n:.4f} '
                      f'lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s', flush=True)
                run, n = 0.0, 0
        vl = loss_on(model, val, args.batch, tok.pad_token_id, args.device)
        save(model, tok, args.model, args.out / f'epoch-{epoch}')
        history.append(dict(epoch=epoch, heldout_loss=vl, seconds=round(time.time() - t0)))
        with log_path.open('a') as f:
            f.write(json.dumps(history[-1]) + '\n')
        print(f'epoch {epoch} held-out loss {vl:.4f}', flush=True)
    best = min((h for h in history if h['epoch'] > 0), key=lambda h: h['heldout_loss'])
    link = args.out / 'best'
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(f"epoch-{best['epoch']}")
    (args.out / 'FT.done').write_text(json.dumps(best) + '\n')


if __name__ == '__main__':
    main()
