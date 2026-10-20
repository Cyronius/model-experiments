"""Phase-0 spike: instantiate the tiny T5, run a few CPU training steps on the
hand-made pairs, and save a loadable checkpoint. Validates config + tokenizer +
save format before any real compute is spent.

Usage:
    python scripts/spike_train.py --tokenizer model/tokenizer --pairs data/spike_pairs.jsonl --output model/titler-spike --steps 50
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import GenerationConfig, PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration


def load_pairs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    pairs = load_pairs(Path(args.pairs))

    config = T5Config(
        vocab_size=8192,
        d_model=256,
        d_kv=64,
        num_heads=4,
        d_ff=1024,
        num_layers=3,
        num_decoder_layers=3,
        feed_forward_proj="relu",
        dropout_rate=0.1,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        tie_word_embeddings=True,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = T5ForConditionalGeneration(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.2f}M")

    enc = tokenizer([p["prompt"] for p in pairs], truncation=True, max_length=512)
    dec = tokenizer([p["title"] for p in pairs], truncation=True, max_length=16)

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    n = len(pairs)
    for step in range(args.steps):
        idx = [(step * args.batch_size + i) % n for i in range(args.batch_size)]
        batch = tokenizer.pad(
            {"input_ids": [enc["input_ids"][i] for i in idx]}, return_tensors="pt"
        )
        labels = tokenizer.pad(
            {"input_ids": [dec["input_ids"][i] for i in idx]}, return_tensors="pt"
        )["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100

        out = model(**batch, labels=labels)
        out.loss.backward()
        optim.step()
        optim.zero_grad()
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step:3d}  loss {out.loss.item():.4f}")

    out_dir = Path(args.output)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    GenerationConfig(
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
        max_new_tokens=16,
        num_beams=4,
        no_repeat_ngram_size=2,
    ).save_pretrained(out_dir)

    # smoke-test generation from the fresh checkpoint
    model.eval()
    sample = pairs[0]["prompt"]
    ids = tokenizer(sample, return_tensors="pt")
    gen = model.generate(**ids, max_new_tokens=16, num_beams=2)
    print(f"sample generation (untrained-quality expected): {tokenizer.decode(gen[0], skip_special_tokens=True)!r}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
