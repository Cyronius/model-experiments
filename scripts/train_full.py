"""Full training run of the tiny T5 (standalone version of the Colab notebook,
for running on a GPU pod or any CUDA box).

    python train_full.py --data-dir <dir with train.jsonl/val.jsonl/tokenizer/> --out <output dir>

Architecture defaults to the v1.1 config (3+3, d256, relu). The v2 candidate
(.claude/plans/titler-v2-8x4-d320-gated.md):

    python train_full.py --data-dir <data> --out model/titler-v2-8x4 \
        --d-model 320 --num-heads 5 --enc-layers 8 --dec-layers 4 --ff-proj gated-gelu

The v1.3 candidate (.claude/plans/titler-v1.3-pretrained-init.md) fine-tunes a
pretrained T5 checkpoint instead of training from scratch -- architecture flags
are ignored (the checkpoint defines the architecture), and --lr/--epochs default
to fine-tuning-appropriate values (1e-4 / 5) unless overridden:

    python train_full.py --data-dir <data> --out model/titler-v13-tiny-pretrained \
        --pretrained google/t5-efficient-tiny
"""

import argparse
import json

import numpy as np
import torch
from datasets import Dataset
from rouge_score import rouge_scorer
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Config,
    T5ForConditionalGeneration,
)

MAX_INPUT_LEN = 512
# NOTE: 32, not 16 — with the 8k vocab, Title Case titles run a median of 20
# tokens; a 16-token cap truncated 75% of labels mid-word and taught the model
# to emit EOS mid-word (the titler-v1.0 bug).
MAX_TARGET_LEN = 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pretrained", default=None,
                    help="HF model id to fine-tune from (e.g. google/t5-efficient-tiny), instead of "
                         "training a fresh T5Config from scratch. Architecture flags below are ignored "
                         "when set -- the checkpoint defines the architecture.")
    ap.add_argument("--epochs", type=int, default=None, help="default: 10 from-scratch, 5 --pretrained")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="gradient accumulation steps (keep batch-size * grad-accum = 128 for v1-comparable dynamics)")
    ap.add_argument("--lr", type=float, default=None, help="default: 5e-4 from-scratch, 1e-4 --pretrained")
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="default 0.1; bumped to 0.2 in titler-v1.4 to counter the generic-collapse "
                         "mode-seeking found in v1.3 run 2 -- see titler-v1.4-collapse-fix-dual-arch.md")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=1024)
    ap.add_argument("--enc-layers", type=int, default=3)
    ap.add_argument("--dec-layers", type=int, default=3)
    ap.add_argument("--ff-proj", default="relu", choices=["relu", "gated-gelu"])
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint in <out>-checkpoints")
    args = ap.parse_args()

    # from-scratch (random init) needs an aggressive LR/schedule to learn everything from
    # nothing; fine-tuning a pretrained checkpoint at that LR risks catastrophic forgetting
    # before the data ever gets used -- see .claude/plans/titler-v1.3-pretrained-init.md
    args.lr = args.lr if args.lr is not None else (1e-4 if args.pretrained else 5e-4)
    args.epochs = args.epochs if args.epochs is not None else (5 if args.pretrained else 10)

    assert torch.cuda.is_available(), "needs a GPU"
    tokenizer = (
        AutoTokenizer.from_pretrained(args.pretrained) if args.pretrained
        else AutoTokenizer.from_pretrained(f"{args.data_dir}/tokenizer")
    )

    def load_jsonl(path):
        with open(path, encoding="utf-8") as f:
            return Dataset.from_list([json.loads(l) for l in f if l.strip()])

    train_ds = load_jsonl(f"{args.data_dir}/train.jsonl")
    val_ds = load_jsonl(f"{args.data_dir}/val.jsonl")

    def preprocess(batch):
        enc = tokenizer(batch["prompt"], truncation=True, max_length=MAX_INPUT_LEN)
        labels = tokenizer(batch["title"], truncation=True, max_length=MAX_TARGET_LEN)
        enc["labels"] = labels["input_ids"]
        return enc

    cols = train_ds.column_names
    train_tok = train_ds.map(preprocess, batched=True, remove_columns=cols)
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=cols)

    if args.pretrained:
        model = T5ForConditionalGeneration.from_pretrained(args.pretrained)
    else:
        config = T5Config(
            vocab_size=8192, d_model=args.d_model, d_kv=64, num_heads=args.num_heads, d_ff=args.d_ff,
            num_layers=args.enc_layers, num_decoder_layers=args.dec_layers, feed_forward_proj=args.ff_proj,
            dropout_rate=0.1, relative_attention_num_buckets=32, relative_attention_max_distance=128,
            tie_word_embeddings=True, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
        )
        model = T5ForConditionalGeneration(config)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, lr={args.lr}, "
          f"epochs={args.epochs}, label_smoothing={args.label_smoothing}")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        pred_txt = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_txt = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {"rougeL": float(np.mean([
            scorer.score(l, p)["rougeL"].fmeasure for l, p in zip(label_txt, pred_txt)
        ]))}

    targs = Seq2SeqTrainingArguments(
        output_dir=f"{args.out}-checkpoints",
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=2000,
        num_train_epochs=args.epochs, label_smoothing_factor=args.label_smoothing,
        # fp16 overflows on pretrained T5 checkpoints (grad_norm -> nan from step 1, every
        # step silently skipped by the loss scaler -- learned observation, not a guess: hit
        # this on titler-v1.3's first attempt). bf16 has fp32's exponent range and is T5's
        # own recommended training precision; the from-scratch path is unaffected (its
        # random-init weight scale doesn't trigger the overflow) so only switch for --pretrained.
        fp16=not args.pretrained, bf16=bool(args.pretrained),
        eval_strategy="steps", eval_steps=1000, save_steps=1000, save_total_limit=2,
        predict_with_generate=True, generation_max_length=MAX_TARGET_LEN,
        load_best_model_at_end=True, metric_for_best_model="rougeL",
        logging_steps=100, report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_tok, eval_dataset=val_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=args.resume or None)

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    GenerationConfig(
        decoder_start_token_id=model.config.decoder_start_token_id,
        pad_token_id=model.config.pad_token_id, eos_token_id=model.config.eos_token_id,
        max_new_tokens=MAX_TARGET_LEN, num_beams=4, no_repeat_ngram_size=2,
    ).save_pretrained(args.out)

    model.eval()
    for p in ["how do I center a div with flexbox", "write a poem about the ocean", "hi"]:
        ids = tokenizer(p, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=MAX_TARGET_LEN, num_beams=4)
        print(p, "->", tokenizer.decode(out[0], skip_special_tokens=True))
    print("saved to", args.out)


if __name__ == "__main__":
    main()
