# Encoded memory: can a small Llama read retrieved knowledge vectors?

This experiment encodes documents into a bank of soft memory tokens, selects a
record for a question, and supplies its vectors to a frozen language model. It is
independent of `manifold-compression/`: there is no donor model, shrinking,
distillation, MoE conversion, or attempt to transfer a larger model's abilities.

**First result:** two 150-step CPU pilots did not demonstrate usable factual
memory. The corrected placement scored 15/16 with original text and 2/16 with
correct compressed memory, exactly matching deliberately wrong memory. Swapping
the memory changed none of the 16 answers. See [the corrected-placement report](results/smol360-reference/report.md)
and [the initial prefix report](results/smol360-pilot/report.md). Generator answers
on the reserved test and XDNA probes were not evaluated. These results reject
this small encoder/interface configuration at this budget, not encoded memory in
general.

The intended recipient is `meta-llama/Llama-3.2-1B-Instruct`. The local Hugging Face
credentials returned HTTP 403 for that gated checkpoint on 2026-10-23. The first
executable pilot therefore uses `HuggingFaceTB/SmolLM2-360M-Instruct`, which runs
through the standard `LlamaForCausalLM` implementation. Results for this checkpoint
are not results for Meta Llama 1B. A different checkpoint requires a newly trained
encoder; changing the model name does not make an existing memory bank portable.

## What is trained and what is stored

```text
Document text --> frozen recipient's contextual features --> trainable resampler
                                                                  |
                                                         16 memory vectors
                                                                  |
Question --> BM25 selector --> selected vectors --> frozen recipient --> answer
```

The offline feature extractor is the frozen recipient itself. A small shared
encoder uses positional embeddings and attention pooling to compress those
features to 16 vectors at the recipient's embedding width. The vectors are
inserted at a configured position in the chat template.
Only the resampler and bridge are trained, using answer-token cross entropy
through the frozen decoder. This is learned alignment, not a tensor reshaping
operation. Both feature extraction and answering use the same unmodified model.
The original pilot used slots after BOS (`--memory-site prefix`). Exact source
embeddings scored only 4/16 there, against 15/16 with ordinary reference text.
The corrected default puts slots inside the user message's reference section
(`--memory-site reference`); exact source embeddings score 15/16 there. An encoder
must be trained for its insertion site, which is stored in its checkpoint.

The encoder never receives a question, answer key, document ID, or learned
per-document embedding. Documents contain their facts, as intended. At evaluation
the encoder is frozen before new documents are converted into memory slots.
The generator's encoded-memory conditions receive the slots and the question;
they receive no source text. The source text remains in an untrained BM25 index
for selection and is used separately for the readable-text control. This is a
test of encoded **values**, not a claim of an entirely text-free retrieval index.

The source-token features cached in `work/` are intermediate artifacts, not the
compressed memory bank. `runs/<name>/memory-bank.pt` contains the compact values.
The small gate encodes only records needed by its controls (20 in the pilot).
The selector still searches the entire exported corpus, including distractors.
Use `--bank-scope all` to build every value offline for arbitrary later queries.

## Data and the first gate

`data.py` reads `C:/code/xdna2-notes/findings/` without modifying that repository.
It preserves evidence level, verification date, stack scope, source path, and a
hash of each original finding. It does not flatten disputed open questions into
asserted facts. Nine manually specified numeric probes have answer spans checked
against their source records; the same spans must survive tokenizer truncation.
These probes are a tiny recall test, not an XDNA reasoning benchmark.

For a first alignment test, synthetic records have random device identifiers and
two random attributes: alignment and queue capacity. They are explicitly marked
fictional and are not hardware advice. Each document has four questions. Splits
are by whole document, not question wording; each fact binding belongs to only
one split. Values can recur, but their association with unseen devices cannot be
learned from the training set. All real XDNA documents are outside training.

The local pilot uses 64 synthetic training documents (256 questions), 8 development
documents (32 questions), 16 test documents (64 questions), and all 347 real
findings as additional retrieval records. Its development evaluation takes 16
questions spread across the eight development documents. This limited sample
contains alignment questions in two phrasings; it does not cover queue capacity.
The first fit has a
fixed budget of 150 optimizer steps, with batch size 4 and seed 42.

Conditions:

| condition | generator receives | checks |
|---|---|---|
| `none` | question | prior and guessing |
| `text` | question + original record | ability to answer with readable evidence |
| `oracle` | question + correct encoded record | encoding and consumption |
| `retrieved` | question + BM25-selected encoded record | complete retrieval path |
| `wrong` | question + different encoded record | whether memory content matters |

Synthetic wrong memories have a different value for the queried attribute. Both
strict normalized exact match and a numeric metric are reported. The latter
accepts a unit after the number but rejects answers containing multiple numbers.
Comma grouping is normalized. Every prediction is saved for inspection.

The useful outcome is a held-out `oracle` improvement over both `none` and `wrong`,
followed by retention of that improvement with actual retrieval. A paired metric
reports recovery specifically on questions the plain recipient misses and the
text control answers correctly. It reports `null` when there are no eligible
questions. Retrieval recall is measured separately as exact source-document-ID
recall; another finding may contain valid evidence too. Lower training loss alone
does not establish that new facts can be encoded. A failed short run does not
disprove encoded memory; it rejects this configuration at this training budget.

## Run from the repository root

Install dependencies in the Python environment used for the experiment:

```powershell
python -m pip install -r encoded-memory/requirements.txt
python -m pytest encoded-memory/test_smoke.py -q
python encoded-memory/data.py --notes C:/code/xdna2-notes --train-docs 64 --dev-docs 8 --test-docs 16
```

Run the ungated local pilot (CPU uses float32 and six threads):

```powershell
python -u encoded-memory/run.py all --model HuggingFaceTB/SmolLM2-360M-Instruct --out encoded-memory/runs/smol360-pilot --steps 150 --eval-split dev --eval-per-domain 16 --memory-site prefix
```

The corrected placement uses a new run folder:

```powershell
python -u encoded-memory/run.py all --model HuggingFaceTB/SmolLM2-360M-Instruct --out encoded-memory/runs/smol360-reference --steps 150 --eval-split dev --eval-per-domain 16 --memory-site reference --threads 2
```

Only after the development gate passes, run the reserved generator test, including
all nine XDNA probes, without additional training. The first configuration did
not reach that stage:

```powershell
python -u encoded-memory/run.py evaluate --model HuggingFaceTB/SmolLM2-360M-Instruct --out encoded-memory/runs/smol360-pilot --eval-split test --bank-scope all
```

Check the memory insertion site without learned compression and export the
development result bundle:

```powershell
python encoded-memory/check_interface.py --run encoded-memory/runs/smol360-pilot
python encoded-memory/report.py --run encoded-memory/runs/smol360-pilot --out encoded-memory/results/smol360-pilot --split dev
python encoded-memory/report.py --run encoded-memory/runs/smol360-reference --out encoded-memory/results/smol360-reference --split dev
```

The interface control supplies the exact original source token embeddings at the
same position used by memory slots. It separates a placement problem from loss
introduced by the learned compressor. Original text in the ordinary user message
is a separate control.

Append `--local-files-only` after a model has been downloaded. Model files are
cached under the repository's ignored `models/encoded-memory-hf/`, not in another
project's cache. GPU runs can use `--device cuda --dtype bfloat16`. Once Meta
Llama access is available, substitute its model ID and use a new `--out` folder.
The default model ID is Meta Llama 1B; there is no silent checkpoint fallback.

`fit` trains only; `evaluate` loads a completed encoder; `all` runs both. Fitting
refuses to overwrite an existing encoder. Feature caches include source content,
model identity, hidden-state layer, and token cap. The recipient's full weights
are hashed before and after training; evaluation verifies the same fingerprint.
Prediction files are updated after each question so completed evaluation cases
are inspectable if a run is interrupted. Evaluation restarts from the beginning;
training does not currently resume a partially completed run.

## Files

| file | role |
|---|---|
| `data.py` | source ingestion, synthetic facts, checked XDNA questions, BM25, answer scoring |
| `memory.py` | trainable resampler, soft-memory insertion, answer loss, cached greedy decode |
| `run.py` | frozen-model loading, source feature cache, training, bank construction, evaluation |
| `report.py` | export the small result bundle without copying weights or feature caches |
| `check_interface.py` | exact source embeddings at the same memory insertion site |
| `test_smoke.py` | offline split/retrieval, padding, label masks, gradient, generation, scoring checks |
| `data/` | generated records and provenance manifest (ignored) |
| `work/` | frozen source features (ignored) |
| `runs/` | encoder checkpoints, memory banks, training logs and predictions (ignored) |
| `results/` | small checked-in reports and measurements |

## Related work

- [xRAG](https://arxiv.org/abs/2405.13792) trains a bridge from retrieval embeddings
  into a frozen language model. It is a precedent for a learned memory interface,
  not evidence that this much smaller model and training budget will succeed.
- [In-context Autoencoder](https://arxiv.org/abs/2307.06945) learns compressed
  memory slots. Its encoder training is more substantial than this small pilot.
