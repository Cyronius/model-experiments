"""Train the v2 decoder-only Llama titler (2026-10-17).

Stage `lm`: next-token training on raw prompts (plus --extra-text) so a from-scratch model learns English.
Stage `title`: <s>{prompt}<sep>{title, no Title Case}</s>, loss on title tokens only.

    python scripts/13_train_llama.py --stage lm --out model/v2-lm
    python scripts/13_train_llama.py --stage title --init model/v2-lm --out model/v2-title
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import LlamaConfig, LlamaForCausalLM

from common.validation import clean_title, decase_title

SHAPES = {
    "5L": dict(num_hidden_layers=5, intermediate_size=768, num_attention_heads=4, num_key_value_heads=4),
    "8L": dict(num_hidden_layers=8, intermediate_size=512, num_attention_heads=4, num_key_value_heads=2),
}
PAD, BOS, EOS, SEP = 0, 1, 2, 3
MAX_PROMPT = 256
MAX_TITLE = 40
LM_BLOCK = 512


def read(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def title_examples(tok: Tokenizer, rows: list[dict]) -> list[tuple[list[int], int]]:
    """(ids, number of leading ids that are context and carry no loss)."""
    prompts = tok.encode_batch([r["prompt"] for r in rows])
    titles = tok.encode_batch([decase_title(r["title"]) for r in rows])
    out = []
    for p, t in zip(prompts, titles):
        ctx = [BOS] + p.ids[:MAX_PROMPT] + [SEP]
        out.append((ctx + t.ids[:MAX_TITLE] + [EOS], len(ctx)))
    return out


def lm_examples(tok: Tokenizer, texts: list[str]) -> list[tuple[np.ndarray, int]]:
    """Packed LM_BLOCK-token blocks; uint16 arrays keep a billion-token stream in memory."""
    blocks, tail = [], np.zeros(0, dtype=np.uint16)
    for i in range(0, len(texts), 20_000):
        parts = [tail]
        for enc in tok.encode_batch(texts[i : i + 20_000]):
            parts.append(np.array([BOS] + enc.ids + [EOS], dtype=np.uint16))
        stream = np.concatenate(parts)
        usable = len(stream) // LM_BLOCK * LM_BLOCK
        blocks += [(stream[j : j + LM_BLOCK], 0) for j in range(0, usable, LM_BLOCK)]
        tail = stream[usable:]
    return blocks


def batches(examples, tokens_per_batch: int, shuffle: bool, seed: int):
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(order)
    # bucket by length inside windows so padding stays small without losing randomness
    for w in range(0, len(order), 2048):
        window = sorted(order[w : w + 2048], key=lambda i: len(examples[i][0]))
        chunks, cur, width = [], [], 0
        for i in window:
            width = max(width, len(examples[i][0]))
            if cur and width * (len(cur) + 1) > tokens_per_batch:
                chunks.append(cur)
                cur, width = [], len(examples[i][0])
            cur.append(i)
        if cur:
            chunks.append(cur)
        if shuffle:
            random.Random(seed + w).shuffle(chunks)
        for chunk in chunks:
            n = max(len(examples[i][0]) for i in chunk)
            ids = torch.full((len(chunk), n), PAD)
            labels = torch.full((len(chunk), n), -100)
            for r, i in enumerate(chunk):
                seq, ctx = examples[i]
                arr = torch.from_numpy(np.asarray(seq, dtype=np.int64))
                ids[r, : len(seq)] = arr
                labels[r, ctx : len(seq)] = arr[ctx:]
            yield ids, labels, (ids != PAD).long()


@torch.no_grad()
def evaluate(model, examples, tokens_per_batch, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for ids, labels, mask in batches(examples, tokens_per_batch, False, 0):
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits[:, :-1]
        target = labels[:, 1:].to(device)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100, reduction="sum")
        total += loss.item()
        count += (target != -100).sum().item()
    model.train()
    return total / max(count, 1)


@torch.no_grad()
def sample_titles(model, tok, rows, device) -> list[str]:
    model.eval()
    out = []
    for r in rows:
        ids = [BOS] + tok.encode(r["prompt"]).ids[:MAX_PROMPT] + [SEP]
        x = torch.tensor([ids], device=device)
        gen = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=MAX_TITLE, do_sample=False,
                             eos_token_id=EOS, pad_token_id=PAD)
        out.append(clean_title(tok.decode(gen[0, len(ids):].tolist(), skip_special_tokens=True)))
    model.train()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["lm", "title"], required=True)
    ap.add_argument("--data", default="data/v3")
    ap.add_argument("--tokenizer", default="model/tokenizer-v2")
    ap.add_argument("--shape", choices=list(SHAPES), default="5L")
    ap.add_argument("--init", help="checkpoint to continue from (title stage after lm)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default="data/01_combined.jsonl", help="lm stage: JSONL with a prompt field")
    ap.add_argument("--extra-text", nargs="*", default=[], help="lm stage: JSONL files with a text field")
    ap.add_argument("--lm-rows", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tokens-per-batch", type=int, default=32768)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(f"{args.tokenizer}/tokenizer.json")
    torch.manual_seed(0)

    val_rows = read(f"{args.data}/val.jsonl")
    val = title_examples(tok, val_rows)
    if args.stage == "title":
        train_rows = read(f"{args.data}/train.jsonl")[: args.limit_train]
        train = title_examples(tok, train_rows)
    else:
        held_out = {r["prompt"] for r in val_rows + read(f"{args.data}/test.jsonl")}
        texts = [r["prompt"][:3000] for r in read(args.raw) if r["prompt"] not in held_out][: args.lm_rows]
        for path in args.extra_text:
            texts += [r["text"] for r in read(path)]
        random.Random(0).shuffle(texts)
        train = lm_examples(tok, texts[: args.limit_train])
        val = lm_examples(tok, [r["prompt"] for r in val_rows])
    print(f"{args.stage}: {len(train):,} train sequences, {len(val):,} val sequences, device {device}")

    if args.init:
        model = LlamaForCausalLM.from_pretrained(args.init)
    else:
        cfg = LlamaConfig(vocab_size=tok.get_vocab_size(), hidden_size=256, max_position_embeddings=MAX_PROMPT + MAX_TITLE + 8 if args.stage == "title" else LM_BLOCK,
                          tie_word_embeddings=True, pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS, **SHAPES[args.shape])
        cfg.max_position_embeddings = max(cfg.max_position_embeddings, LM_BLOCK)
        model = LlamaForCausalLM(cfg)
    model.to(device).train()
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    steps_per_epoch = math.ceil(1.1 * sum(len(e[0]) for e in train) / args.tokens_per_batch)  # padding allowance
    total = args.max_steps or math.ceil(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    warmup = max(1, total // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast(device_type="cpu", enabled=False)

    step, start, seed = 0, time.time(), 0
    while step < total:
        for ids, labels, mask in batches(train, args.tokens_per_batch, True, seed):
            with amp:
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits[:, :-1]
            loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels[:, 1:].to(device).reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{total}  loss {loss.item():.3f}  lr {sched.get_last_lr()[0]:.2e}  {(time.time() - start) / step:.2f}s/step", flush=True)
            if step % args.eval_every == 0 or step == total:
                print(f"  val loss {evaluate(model, val, args.tokens_per_batch, device):.3f}", flush=True)
            if step >= total:
                break
        seed += 1

    out = Path(args.out)
    model.save_pretrained(out)
    Path(out / "tokenizer.json").write_bytes(Path(f"{args.tokenizer}/tokenizer.json").read_bytes())
    if args.stage == "title":
        for r, t in zip(val_rows[:8], sample_titles(model, tok, val_rows[:8], device)):
            print(f"  {t}   <-  {r['prompt'][:70]!r}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
