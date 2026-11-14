# Encoded document memory in a 4B language model

This experiment asks whether a model can recall facts from XDNA2 notes supplied
as continuous memory vectors. The document text is encoded offline; questions
then attend to those saved vectors inside the decoder. There is no query-time
retriever and no document text in the latent-condition decoder prompt.

The first pilot uses the released **LCLM 4x** encoder, MLP bridge, and Qwen3-4B
decoder. These weights already know how to communicate. No new training is
performed. This first experiment tests the memory interface and recall, rather
than whether a newly added layer on a stock frozen model can learn the interface.

## First result, 2026-10-25

The encoded-memory path recalled **13 of 14 note facts** after qualitative
review, both synthetic code phrases, and both changed phrases when only the
memory bank was replaced. No-memory and zero-memory controls recalled none of
the facts or canaries. The latent path recognized both absent facts, although
one of those responses elaborated until the output cap.

| Condition | Note facts | Synthetic phrases | Missing-fact abstention |
|---|---:|---:|---:|
| Encoded memory | 13/14 | 2/2 | 2/2 |
| Zeroed memory | 0/14 | 0/2 | 2/2 |
| No memory | 0/14 | 0/2 | 2/2 |
| Stock Qwen, full text | 12/14 | 2/2 | 1/2 |

The 1,929 source tokens became 483 vectors, width 2,560, occupying 2,472,960
bytes. Encoding took about 1.35 seconds on a Ryzen AI MAX+ 395 CPU with eight
PyTorch threads. This reduces decoder document positions by about 4x; the
vector file is larger than the original UTF-8 text. It is not 4x disk compression.

One latent response quoted `uint8` but denied that the recommendation was
specified; its regex match is rejected. The firmware response correctly said
the version was unspecified but missed the fixed regex wording; review accepts
its abstention. Both original regex results and review reasons are retained.

The LCLM decoder's full-text condition immediately generated an end token on
every question under this prompt. A direct, uncached forward pass confirmed
the same top end token with finite logits, so prefix-cache reuse does not
explain that diagnostic case. Stock-Qwen controls were added after that issue
was observed. The stock control also produced one empty answer, misspaced
`-Os` as `- Os`, and confused the recovery phrase with an absent override
password. Some responses hit the 40-token cap. These are pilot output-quality
limitations, not evidence that compression improves the underlying model.

See [complete results and raw answers](results/pilot/summary.md),
[protocol and scores](results/pilot/scored.json), and
[the full-text diagnostic](results/pilot/fulltext-diagnostic.json).

## Architecture

```text
six XDNA2 notes + synthetic controls
    -> trained Qwen3-Embedding-0.6B encoder (1024-token windows)
    -> mean pool groups of 4 hidden states
    -> trained RMSNorm / linear / GELU / linear bridge
    -> saved continuous vectors, width 2560

saved vectors + token embeddings of the question
    -> Qwen3-4B decoder, via inputs_embeds
    -> ordinary self-attention over document memory and question
    -> short answer
```

The bridge is a separate module. This version uses existing decoder attention
instead of adding a new cross-attention block. The decoder was also trained by
the LCLM authors; it is not an untouched stock Qwen model. Ordinary sentence
embeddings cannot be substituted for these vectors without learning an alignment.

## Research references

- [xRAG (2024)](https://arxiv.org/abs/2405.13792): a learned modality bridge maps
  dense document embeddings into a frozen language model's input space. This is
  a plausible match for the remembered paper; the original paper is unidentified.
- [End-to-End Context Compression at Scale (2026)](https://arxiv.org/abs/2606.09659):
  trained encoder / adapter / decoder systems with 4x, 8x, and 16x compression.
- [LCLM code](https://github.com/LeonLixyz/LCLM), pinned at
  `45de65087d910c7d6e5e478e0930a716c34d4d10`.
- [Pilot checkpoint](https://huggingface.co/latent-context/0.6b-4b-LCLM-4x), pinned
  at `d45629704bbc8de96d4b847c0098bf7ccebb3aa4`. About 18.54 GB downloaded in
  float32 storage; loaded as bfloat16 on CPU. Encoder plus decoder is about 4.6B
  parameters. The MLP weights are also supplied by the authors.

## Preregistered recall test

`cases.json` fixes questions and answer patterns before model inference:

- 14 factual questions from six complete, hash-recorded notes: memory size,
  error identifiers, compiler flags, conflicting stack observations, API traps,
  ROT13 timings, masked softmax, and context-switch timings.
- Two arbitrary code phrases appended as **synthetic test fixtures**, not facts
  about hardware. Success here is evidence of reading the supplied memory.
- Two absent facts, for which the expected behavior is `UNKNOWN`.

Five conditions use the **same decoder**, greedy decoding, independent question
histories, and a fixed 40-token output cap:

1. `no_memory`: question with no notes.
2. `full_text`: all note text, the positive control.
3. `latent`: saved encoded document memory.
4. `zero_memory`: same number of memory positions, all vectors zeroed.
5. `counterfactual`: only the two canary questions; new memory changes the two
   code phrases while preserving the real notes. Answers should follow the bank.

The final assistant prefix contains an empty closed `<think>` block. An earlier
format preflight without it spent the output budget on reasoning; that excluded
run is preserved under `results/prompt-preflight/`. The two subsequent format
preflights are also retained. Cases and answer patterns were fixed before all
inference. The stock controls were added afterward; qualitative review is
explicit in `review.json` rather than silently changing those patterns.

The source notes and canaries are encoded **before and independently of the
questions**. The scoring keys are not passed to the model. The encoded decoder
prefix is cached once per condition; its KV cache is cloned for each question.
No training examples, parameter updates, tools, or original-document fallback
are used. `questions.json` contains queries only; `gold.json` is read by the
scorer after inference. Each source file and memory artifact has a checksum.

Pilot acceptance: the full-text control must establish that these questions are
answerable; the latent condition should retain factual recall, recover both
canaries, and follow both changed canaries. Report abstention separately.
Zero/no-memory failures on arbitrary canaries are essential controls. This is
a hand-selected feasibility test, not a general RAG benchmark or a claim of
hardware truths across all stacks.

## Run

Use a CPU Python environment with `torch`, `transformers`, `huggingface-hub`,
`safetensors`, and `pytest`. Tested versions are recorded in each run's
`environment.json`; no system environment is modified. The initial pilot uses
`target/manifold-audit/env/bin/python` from the `model-experiments` root.

```bash
# From the model-experiments repository root:
PY=target/manifold-audit/env/bin/python
CODE=target/context-memory/LCLM
CKPT=models/context-memory/lclm-4x

git clone https://github.com/LeonLixyz/LCLM "$CODE"
git -C "$CODE" checkout 45de65087d910c7d6e5e478e0930a716c34d4d10
# Downloads public, free model weights. No hosted inference or paid API.
$PY context-memory/download.py --out "$CKPT"

$PY context-memory/run.py prepare --notes ../xdna2-notes --data context-memory/data
$PY context-memory/run.py encode --checkpoint "$CKPT" --upstream "$CODE" \
  --data context-memory/data --memories context-memory/memory
$PY context-memory/run.py evaluate --checkpoint "$CKPT" \
  --data context-memory/data --memories context-memory/memory \
  --output context-memory/results/pilot
$PY context-memory/run.py summarize --data context-memory/data \
  --memories context-memory/memory --output context-memory/results/pilot \
  --review context-memory/review.json
$PY -m pytest -q context-memory/test_memory.py
```

To reproduce the additional stock controls and the combined report:

```bash
$PY context-memory/download.py --model stock --out models/context-memory/qwen3-4b-stock
$PY context-memory/stock_control.py --checkpoint models/context-memory/qwen3-4b-stock \
  --data context-memory/data --output context-memory/results/stock-control
$PY context-memory/run.py summarize --data context-memory/data \
  --memories context-memory/memory --output context-memory/results/pilot \
  --review context-memory/review.json \
  --extra-answers context-memory/results/stock-control/answers.jsonl
```

`data/sources/` snapshots and `sources.json` make the chosen corpus reproducible.
`memory/*.safetensors` is the reusable encoded artifact; its manifest stores
shape, source hash, encoding time, model revision, and file hash. The result
directory contains all answers, automatic scores, timing, and environment.

To study actual model surgery next, freeze a stock Qwen3-4B, add a gated
cross-attention module to read encoder memory, and train the projections and
gate. Evaluate on documents and arbitrary facts held out from bridge training;
fitting a bridge on these same QA pairs would only show memorization. That is a
separate experiment from testing this released pretrained interface. Any paid
training requires a bounded cost proposal and explicit approval first.
