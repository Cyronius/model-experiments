# Cross-attention memory on a frozen stock Qwen3-4B

**Status (2026-11-06 23:32): negative result.** The 360-step main run finished
with no skipped steps (`../results/xattn-main/`). The new layers did not learn to
look up content.

| Held-out measure | memory | wrong memory | no memory (stock) | full text (stock) |
|---|---:|---:|---:|---:|
| SQuAD answerable F1 (primary) | 0.035 | 0.037 | 0.054 | 0.744 |
| Answerable questions abstained | 93% | 93% | 83% | 5% |
| Reconstruction NLL, FineWeb / SQuAD | 2.31 / 2.24 | 2.50 / 2.42 | 2.83 / 2.92 | — |
| Continuation NLL | 2.61 | 2.67 | 3.33 | 3.45 |
| XDNA facts / canaries | 0/14, 0/2 | — | 0/14, 0/2 | — |

- With memory, QA is no better than wrong memory and worse than the stock model.
  The answers it does give are almost always identical under wrong memory, so
  they come from the decoder's own knowledge.
- Most of the teacher-forced gain over no memory is content-independent (wrong
  memory gets it too). The content-dependent part is small: about 0.19 nats on
  reconstruction, up from 0.03 at step 95.
- Training probes (120 items) stayed between F1 0.00 and 0.04 throughout.
- The stock full-text continuation NLL (3.45) is not a usable upper bound: the
  untrained chat model does not continue raw text in the target format.

**First GPU attempt failed, and was restarted:**
- The first smoke run reached a non-finite gradient at step 3. Clipping turned it into NaN weights from step 4 on.
- The job was stopped before the main run used it. Records are in `../results/xattn-smoke-nan/`.
- A re-run of the same examples with length-sorted micro-batches stayed finite (gradient norms 29–37), recorded in `../results/xattn-batching-check/`.
- Two direct checks found nothing wrong:
  - ROCm attention masking matched a float32 reference;
  - padded and unpadded batches gave the same gradients.
- The cause is not identified. During that run the GPU was also being used, outside dibs, by another training job and by llama servers, and the machine was swapping 44 GB.
- Training now skips any step with a non-finite gradient and stops after three in a row.

## Question

The [LCLM pilot](../README.md) put encoded document vectors *into the token
sequence*. 1,929 note tokens became 483 positions that still occupy the context
window, the KV cache and RoPE slots, and it needed a decoder the LCLM authors
had fine-tuned. This experiment asks:

> Can new layers added to an **untouched** Qwen3-4B-Instruct-2507 learn to read
> encoded documents through cross-attention, so the memory uses **no context
> positions at all**?

## Design

```text
passage text ──> LCLM 0.6B encoder (frozen) ──> mean-pool every 4 tokens ──> 1024-wide vectors
                 (encoded once, offline; never sees questions)          │
                                                                        ▼
                                   memory adapter (RMSNorm → Linear → GELU → Linear, + slot embedding)
                                                                        │ 2560-wide memory bank
question tokens ──> Qwen3-4B (all 36 layers frozen) ─────────────────────┤
                    before layers 2, 6, 10, …, 34:                      │
                    h ← h + W_o · Attn(q = W_q·norm(h), k,v = W_k,W_v·norm(memory))
```

- **Decoder:** stock `Qwen/Qwen3-4B-Instruct-2507`, every weight frozen. Memory
  vectors are never placed in the sequence. Prompt positions are the same as
  with no notes at all.
- **Encoder:** LCLM's released 0.6B encoder, frozen. It is a trained compressor,
  so this run tests the *interface into a stock decoder*, not encoder training.
  Each passage of up to 1,024 tokens is encoded alone; a memory bank is a
  concatenation of passages.
- **New, trained parts (127.8M parameters, 3.5% of the decoder):**
  - nine cross-attention blocks: 16 query heads, 4 KV heads (GQA), head width 128, with q/k norms;
  - a shared memory adapter, initialized from LCLM's MLP bridge weights;
  - a slot-position embedding.
- **Identity at initialization:** the output projections start at zero, so
  before training the model is exactly the stock model (tested). With no memory
  set, the blocks are skipped, so the stock model is always available.
- **Memory cost:** each block caches 4 KV heads × 128 per memory slot, so about
  9.2k values per slot across nine blocks. In-context text costs 36 layers × 8
  KV heads × 128 × 2 per token, about 295k values per 4 tokens. The memory
  cache is about 32× smaller than the full-text cache, and it uses no positions.

## Training data (`data.py`, `rules.json`)

All public and free, pinned by revision:

| Kind | Source | Memory | Target (loss) | Train examples |
|---|---|---|---|---:|
| `qa` | SQuAD v2 train | gold paragraph + 0–11 distractor paragraphs from other articles, shuffled | short answer, or `UNKNOWN` for v2 unanswerables | 130,319 |
| `qa_removed` | SQuAD v2 train | 1–12 distractors, gold paragraph **removed** | `UNKNOWN` | 12,000 |
| `recon` | SQuAD paragraphs, FineWeb-Edu (128–512 tokens) | the passage | the passage, verbatim | 10,300 |
| `continue` | FineWeb-Edu | first 256–1,024 tokens | next 128–256 tokens | 4,300 |

- **Sampling mix (by example):** 55% `qa`, 7% `qa_removed`, 24% `recon`, 14% `continue`.
- **Loss:** each example's mean token loss, so a 5-token answer weighs as much as a 400-token passage.
- **Step size:** about 16k decoder tokens.
- **Optimizer and schedule:** AdamW at 2e-4, 100 warm-up steps, cosine decay to 10%.
- **Run length:** 360 steps, about 5.9M decoder tokens. At the measured 130–200 tokens per second on the shared GPU, that is 8–12 hours.
- **Cost:** no paid compute.

## Preregistered evaluation (`train.py evaluate`)

Held out, fixed before training:

1. **SQuAD v2 validation**, whose articles are disjoint from train: 400 answerable
   questions, 100 unanswerable, and 100 answerable with the gold paragraph removed.
   Each has 0–11 distractor paragraphs. Greedy decoding, 32-token cap, official
   normalization, EM/F1.
   - **Primary measure:** answerable F1 with memory.
   - **Conditions:**
     - `memory`
     - `no_memory`: blocks off, which is exactly the stock model.
     - `wrong_memory`: another item's bank, with no paragraph from the question's own article.
     - `full_text`: the stock model with the paragraphs in the prompt. This is the upper bound.
2. **Teacher-forced reconstruction and continuation:** token NLL and accuracy
   on 200 held-out passages and 100 continuations, with memory, without memory,
   and with wrong memory.
3. **The XDNA2 pilot** (`../cases.json`): 14 facts, 2 canaries and 2 absent facts,
   scored with the pilot's own patterns. The canaries are also tested after
   swapping in the counterfactual memory bank. These notes are technical and
   out of domain; the training data never contains them.

## Run

```bash
PY=../../target/manifold-rocm/env/bin/python
$PY data.py fetch && $PY data.py build
$PY data.py encode --device cpu          # about 3 h on a loaded CPU; minutes on the GPU
$PY -m pytest -q test_xattn.py
dibs run gpu --preemptible --requeue -- ./run_gpu.sh   # train (resumable) → evaluate
```

`run_gpu.sh` can be rerun safely at any point. Training checkpoints every 30
minutes. If another session queues for the GPU, training saves a checkpoint and
hands the GPU over (exit code 75), and resumes when the GPU comes back.

Outputs go to `../results/xattn-main/` (logs, probes, `eval/summary.json`).
Large files go to `../../target/context-memory/xattn/`: encoded memory (about
0.2 GB eval, 3.5 GB train) and checkpoints.

## Smoke checks done

- `test_xattn.py` (7 tests, tiny random Qwen3) checks that:
  - the output is identical at initialization;
  - batch rows are isolated from each other's memory;
  - memory adds no sequence positions;
  - gradients reach only the new modules;
  - cached and uncached generation agree.
- **CPU training smoke:** three steps on the eval bank ran, checkpointed and resumed cleanly.
  - At initialization the stock model answers `UNKNOWN` to every memory question: QA loss is about 11 nats per token and `qa_removed` loss about 0.
  - The new layers must therefore overcome a confident prior, not just fill a gap.
- **CPU evaluation smoke** (3 items): every stage ran.
  - Full text answered 2 of 3 exactly.
  - Memory, no memory and wrong memory all abstained, as expected before training.
