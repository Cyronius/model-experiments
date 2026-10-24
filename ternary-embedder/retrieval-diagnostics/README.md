# Retrieval diagnosis for 2a_384

This experiment measures ranking margins and the immediate inference effect of
quantization at fixed saved weights. It also trains the missing independent
150M-token float-only control on an isolated GPU; comparisons run locally on CPU.

## Results (2026-10-23)

The same two-layer student, trained for the same 17,134 updates, scores **44.44
in float versus 42.13 with the original float-then-ternary recipe**. MiniLM
scores 46.92. About half the observed gap is present without quantization; the
other half comes from the current ternary training recipe at this budget.
This does not establish an intrinsic quality ceiling for two layers or ternary.

| Model / training | SciFact nDCG@10 | NFCorpus nDCG@10 | Equal-task mean |
|---|---:|---:|---:|
| MiniLM-L6, reference | 62.386 | 31.453 | 46.920 |
| 2a_384, float-only, 150M tokens | 57.403 | 31.470 | 44.436 |
| 2a_384, original float/QAT, 150M tokens | 54.856 | 29.409 | 42.132 |
| QAT weights, activation quantization off | 54.602 | 29.380 | 41.991 |
| 2a_384, float-only, halfway checkpoint | 56.805 | 30.788 | 43.797 |
| Halfway checkpoint, quantized without adaptation | 34.042 | 16.696 | 25.369 |

Paired query-bootstrap 95% intervals, in mean nDCG points:

- MiniLM -> full-budget float student: **-2.484**, CI [-4.142, -0.765].
- Full-budget float -> original QAT recipe: **-2.304**, CI [-3.277, -1.320].
- Ternary weights alone -> ternary weights plus int8 activations: **+0.141**,
  CI [-0.044, +0.372]. No evidence here that int8 activation quantization
  explains the remaining loss after QAT.
- Doubling float training from about 75M to 150M tokens: **+0.639**,
  CI [+0.109, +1.156]. More vector distillation helped modestly.

These intervals measure query sampling uncertainty, not variation across
training seeds. The control uses seed 0, matching the original run. Before the
quantization switch, their training traces reproduce closely: through step
8,000 the maximum STSB-dev correlation difference is 0.000313, and maximum held
teacher-cosine difference is 0.000098. The matched-config check and these prefix
comparisons are recorded in `diagnosis.json`.

The immediate conversion is destructive (43.80 -> 25.37), and subsequent QAT
recovers most of that damage. The final unquantized QAT *master weights* score
only 1.40: they were optimized through the quantizer and are not a usable
float-only control. This is why independently training the float control was
necessary.

The vectors show concrete ranking failures. For NFCorpus query **"organotins"**,
both MiniLM and the full-budget float student put a judged-relevant paper first;
the QAT student puts its first judged-relevant document at rank 31. Two SciFact
PIN1/VPS9a queries have first-relevant ranks 9/11 in the full-budget float
student, versus 253/198 with QAT and 1/1 with MiniLM. These selected examples
illustrate failures; they are not a representative accuracy estimate.

The student has a larger common vector component than MiniLM, but its centered
effective dimension is similar (about 142-143 versus MiniLM's 135-153 on these
corpora). It has not simply collapsed into a single vector. A fixed, label-free
corpus-centering transform **hurts retrieval on both tasks**. Uniformly spreading
the vectors is not a sufficient fix; relevant/competitor orderings must improve.

This study isolates a cost of the **current quantized training recipe** and a
remaining float-student gap. It does not separate the latter into depth,
initialization, corpus, distillation objective and training budget. Nor does it
prove that a different QAT schedule could not recover the quantization gap.
The earlier claim that retrieval loss was "not ternary" was unsupported and is
superseded by this controlled result.

The isolated A5000 ran for 0.7141 hours at $0.27/hour, about **$0.193 compute**
(storage excluded). All 26 artifact files were downloaded and SHA256-verified
before deleting pod `njsvb6ugudq8qk`. GPU float-control MTEB scores are also
retained (STSBenchmark 82.4396; SciFact 57.403; NFCorpus 31.470); the table above
uses consistent local CPU scoring for every model.

## Controlled comparisons

The Grid 1 float checkpoint contains 8,567 float-training updates (about 75M
tokens). The final checkpoint contains 17,134 updates (about 150M tokens), the
second half with quantization-aware training (QAT). Comparing these checkpoints
directly does not isolate quantization because their training budgets differ.

Instead, evaluate each checkpoint with the same master weights in both float
and ternary-weight/int8-activation mode. The final checkpoint is also evaluated
with just weight quantization and just activation quantization. The latter
factorial comparisons locate the immediate arithmetic effect. Switching the
final QAT master weights back to float is **not** equivalent to independently
training a float-only model for 150M tokens; QAT can change the master weights
to compensate for quantization. These inference ablations alone do not provide
the missing training-recipe control or establish the best possible two-layer quality.
`float_control.sh` supplies that separate training-recipe control: the same
150M-token budget, initialization, seed, batch size, vector loss and learning-rate
schedule, with quantization left off throughout. Corpus text, pair and token
counts must match Grid 1/2 before training. It is a single-seed comparison on a
different GPU/runtime, rather than a complete precision study.

`diagnose.py` freezes each matrix's effective weights once for CPU evaluation.
The regression tests verify bit-exact equivalence with the original training
module's float and quantized forward paths, including six activation groups at
FFN width 1536. Neither pooling nor the float output projection is quantized.

## Evaluation protocol

- Complete cached MTEB test splits: SciFact (5,183 documents, 300 queries) and
  NFCorpus (3,633 documents, 323 queries).
- Original 128-token cap for every model, no query prompts, title plus one
  space plus body text, cosine similarity, complete-corpus exact search.
- MiniLM uses its pinned original revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` and mean pooling. Students use the
  original bge-small tokenizer and their saved pooling/output projection.
- nDCG@10 uses the raw relevance grade, matching TREC; a grade of 2 is gain 2,
  not gain 3. Self-ID matches are excluded. Equal scores break by descending
  document ID.
- Paired confidence intervals resample queries 4,000 times with seed 0.
  The two task scores are averaged with equal task weight, as in the grid.
- The reported margin is the best judged-positive score minus the strongest
  competitor's score. A competitor has no positive qrel. **Unjudged does not
  mean known irrelevant**; these margins describe the benchmark's judgments.
- Fixed-pair comparisons hold both document identities constant: the positive
  is MiniLM's highest-scoring judged-positive, and the competitor is the final
  quantized student's strongest document without a positive qrel. These are
  deliberately selected hard cases, not an unbiased sample of document pairs.

Protocol references: [MTEB 1.39.7 evaluator](https://github.com/embeddings-benchmark/mteb/blob/1.39.7/mteb/evaluation/evaluators/RetrievalEvaluator.py),
[MTEB qrels loader](https://github.com/embeddings-benchmark/mteb/blob/1.39.7/mteb/abstasks/AbsTaskRetrieval.py),
[TREC nDCG implementation](https://github.com/usnistgov/trec_eval/blob/master/m_ndcg_cut.c).

`verify_original.py` checks the diagnostic against the unchanged original
`StudentEncoder`, using its original character-length batching and batch 256.
SciFact vectors match bit for bit; NFCorpus has small batching-sensitive vector
differences but essentially identical retrieval scores. The current CPU scores
for the QAT model differ slightly from the archived GPU results (54.856 vs
54.067 on SciFact; 29.409 vs 29.583 on NFCorpus). Consequently all paired
conclusions use the same current CPU stack for both models, not a mixture of
archived GPU scores and fresh CPU scores. `environment.json` records that stack.

## Reproduce

The two datasets are read directly from existing Arrow caches without modifying
them. Model downloads and all new embedding caches stay inside this experiment.
The MiniLM model and bge-small tokenizer must first be downloaded to
`cache/huggingface`; their exact snapshot paths are constants in the script.
The original Grid 1 checkpoint files must remain in `tern-embed/pulls/grid1`.

```powershell
$env:OMP_NUM_THREADS = '6'
$env:MKL_NUM_THREADS = '6'
C:/Python311/python.exe ternary-embedder/retrieval-diagnostics/fetch_models.py
C:/Python311/python.exe -m unittest discover -s ternary-embedder/retrieval-diagnostics -p test_diagnose.py -v
C:/Python311/python.exe -u ternary-embedder/retrieval-diagnostics/diagnose.py
C:/Python311/python.exe ternary-embedder/retrieval-diagnostics/summarize.py
C:/Python311/python.exe ternary-embedder/retrieval-diagnostics/center_probe.py
```

`diagnosis.json` retains scores, paired query deltas, checkpoint and dataset
hashes, and fixed-pair margin counts. `geometry-and-examples.json` records corpus
geometry and concrete ranking failures. Ignored `outputs/` holds every query's
rank, nDCG and margin, plus all fixed-pair scores. Ignored `cache/` holds model
files and embeddings; cache identities include checkpoint, arithmetic mode,
model source, tokenizer and exact input text. `run_control.py` rents only its own
pod at a maximum $0.30/hour, watches a one-hour limit, verifies every downloaded
artifact by SHA256, and deletes the pod after successful collection. Training
artifacts and source/environment snapshots are retained under ignored
`float-control/`; `control-run.json` records cost and checkpoint provenance.

The original `/workspace` NFS volume stalled random memory-mapped teacher-vector
writes, leaving the GPU idle. The successful control uses container-local
`/tmp/tern-embed-float-control-work` for teacher and training arrays and copies
the completed teacher file back sequentially for provenance hashing. The final
teacher batch is the original 512; preliminary NFS/batch-2048 attempts were
abandoned before student training. This changes cache placement, not the
teacher function or student training recipe.
