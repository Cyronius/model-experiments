# Canvas editor (Model B), smoke test

The canvas editor of `PLAN.md` section 3.6, in a small first version to test whether
the canvas formulation learns documentation rewrites better than the tagger. It
shares the tokenizer, vocabulary, copy slots, checker and data with
`model_tagger/`. No weights or vocabulary artifacts are committed.

## Model

One bidirectional transformer (pre-LN) reads `[mode] [source pieces]` and a canvas of
`2n + 8` slots for a source of `n` pieces. Source piece `i` sits at slot `2i`, `NULL`
fills the gaps, and the reference is placed by the tokenizer's alignment
(`align_ids`, the `target_positions` of the ID export): kept and replaced words at
`2i`, insertions in the gaps, a run of insertions pushing the rest right. Canvas slot
`k` shares position `1 + k // 2` with its source piece, plus a parity and a segment
embedding. The canvas input uses the input vocabulary plus `MASK`, so rejected words
start in place; the output head is tied to the output-word embeddings and masked to
ordinary words, `NULL` and the request's active slots. A case head predicts lower,
title or upper case for each word.

Default size: width 256, six layers, four heads, FFN 1024, about 8M parameters (the
tagger: 27.9M). No dropout: on CPU it took 60% of a training step, and the corruption
already regularizes.

## Training (`train.py`)

Rows are re-encoded from their text with every detected span as a slot, as at
inference. For each row and each draw:

- with probability `--start-share` (0.2) the canvas is the inference start, the
  whole source canvas;
- otherwise a fraction `t ~ U(0.05, 1)` of slots is corrupted, each to `MASK` or to
  the source-aligned token with equal odds, and the other slots show the reference.

Cross-entropy is taken on the corrupted slots; slots whose reference equals the
source-aligned token weigh `--copy-weight` (0.4). Identity pairs (the reference
rewritten to itself) make up `--identity-share` (0.3) of each epoch, so clean input
is left alone. AdamW, LR 5e-4 with 300 warmup steps and cosine decay to 10%, batch
32, bf16 autocast.

## Inference (`canvas.rewrite`)

Start from the source canvas. Each pass predicts every unfrozen slot; a slot whose top
probability passes the pass threshold (0.95, 0.9, 0.85, 0.8, 0.75) is frozen with its
prediction, the others are set to `MASK` for the next pass, and a sixth pass accepts
every slot. The canvas is collapsed (`NULL` dropped, slots restored with their exact
counts); a slot count that does not match is a failed output. If nothing changed,
the source string is returned as is. The checker runs once at the end and is
reported, not used to re-mask.

Not yet as planned: 64/128/256 buckets, checker features for the canvas, re-masking
of violation spans, GGUF export.

## Smoke run (`run_smoke.py`)

Trains on the 6,897 documentation train pairs of `id-quality-v5` for 60 epochs, then
decodes the 679 documentation dev pairs of `id-doc-v5` (`decode.py`, the counts of
`model_tagger/decode_pairs.py` plus guard and checker counts) and decodes them again
with the fresh doc-only tagger (`doc-v5b-pipeline/probes/fresh-doc.pt`) for the same
counts. Output: `data/out/canvas-smoke-v1/`.

```bash
data/out/.venv/bin/python -m unittest discover -s model_canvas/tests
data/out/.venv/bin/python model_canvas/run_smoke.py
```

## Smoke result (2026-10-31)

`data/out/canvas-smoke-v1/`: 60 epochs, 18,480 updates, 68 minutes on CPU, 7.96M
parameters. Raw decode of the 679 documentation dev pairs, same counts for both:

| | Canvas editor (8M) | Doc-only tagger (27.9M) |
|---|---|---|
| Exact reference | 56 | 88 |
| Closer / farther than the source | 426 / 127 | 512 / 79 |
| Unchanged | 4 | 11 |
| Clean reference changed | 13 | 23 |
| No output | 47 (all slot counts) | 13 |
| Content guard flags the output | 622 | 643 |

- It overfits: dev loss is lowest at epoch 15 (1.32) and rises to 4.73 at epoch 60
  while train loss falls to 0.003. The last checkpoint is the one decoded.
- It does split sentences ("The host variable is of type `int`. The datum in the
  database is ..."), which the tagger breaks, but parallel filling also writes
  repeated or stray words ("Sets sets the the header", "object thread method the
  end").
- All 47 failed outputs copy a code or name slot the wrong number of times.
- The content guard flags over 90% of both models' doc rewrites, so it is not a
  usable metric for documentation.

Cheap next steps, in order: stop near epoch 15 (or select on a dev slice) and add
regularization; decode slots to their exact counts instead of failing; re-mask
repeated neighbours between passes. Then compare again before any size increase.

## Smoke v2: early stopping, slot fitting, repeats (2026-10-31)

Decoder changes (`canvas.rewrite`): each copy slot is fitted to its source count
after the last pass (`fit_slots`), the less confident of two adjacent frozen copies of
a word is masked again between passes (`unfreeze_repeats`), and adjacent repeats left
at the end are dropped. Training changes: 5% of the train groups held out, best
held-out loss saved as `best.pt`, input dropout 0.1, weight decay 0.05, 25 epochs.

| 679 doc dev pairs | v1, old decoder | v1, new decoder | v2 `best.pt` (epoch 14) | Doc-only tagger |
|---|---|---|---|---|
| Exact reference | 56 | 60 | 58 | 88 |
| Closer / farther | 426 / 127 | 467 / 133 | 470 / 159 | 512 / 79 |
| Clean reference changed | 13 | 18 | 29 | 23 |
| No output | 47 | 0 | 0 | 13 |

`data/out/canvas-smoke-v2/`, 7,200 updates, 28 minutes. The slot fit removes every
failed output; early stopping does not raise the exact count. Held-out loss does not
track decode quality: the quick decode of 128 dev pairs gets 13 exact at epoch 14
and 15 at epoch 25, with clean changes falling from 9 to 3. The errors are dropped
content words ("Sets" deleted, "need to be unpacked" -> "must be") and stray words
("At frontend must time send"). On 6,897 pairs, at 8M parameters, the canvas editor
does not beat the tagger. More parameters on the same pairs would overfit sooner;
pretraining on the local rule pairs (507,399 train) before the doc pairs is the
data-side test.

## Pretraining test (2026-10-31)

`run_pretrain.py`, `data/out/canvas-pretrain-v1/`. Stage 1: 2 epochs (15,080 updates,
47 minutes) on the non-documentation train rows of `id-quality-v5`, with
`drop_article` capped at 120,000 of 403,221 and the 59,498 pairs that delete an
article before a backtick literal skipped (168,883 pairs). Stage 2: 25 epochs on the
documentation pairs from the stage-1 end, LR 3e-4; `best.pt` is the epoch with the
most exact rewrites of the 447 held-out documentation train pairs (`--select decode`).
Control: stage 2 from scratch with the same selection.

| 679 doc dev pairs | Pretrain only | Pretrain + doc (epoch 18) | Doc only, decode-selected (epoch 11) | Smoke v2 | Doc-only tagger |
|---|---|---|---|---|---|
| Exact reference | 2 | 62 | 62 | 58 | 88 |
| Closer / farther | 282 / 320 | 485 / 134 | 479 / 113 | 470 / 159 | 512 / 79 |
| Clean reference changed | 279 | 10 | 32 | 29 | 23 |
| Checker clean | 542 | 600 | 554 | 583 | 586 |

Pretraining makes the editor converge in 3 epochs instead of 8, raises the
held-out exact count (44 vs 39), and leaves clean input alone (10 changed vs 32, now
better than the tagger), but the exact count on dev does not move (62 vs 62), and
the tagger stays ahead at 88. The pretrained model alone rewrites documentation
badly: the rule pairs are mostly article insertion. At 8M parameters with this data,
the canvas editor does not close the gap; per the decision rule of the test it is
shelved, and the doc effort goes back to the tagger.
