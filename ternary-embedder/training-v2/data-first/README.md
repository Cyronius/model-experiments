# Data-first embedding training

The user chose more and better data on 2026-10-25. The next comparison uses
positive relevance pairs directly, without a teacher model, teacher vectors,
teacher ranking KL, or a frozen-vector anchor. Keep the existing two-layer
384-wide architecture so the data/objective change can be measured first.
The product plan retains ternary, int4 and int8 variants, with float as reference.
Prior distillation results remain historical; their scripts are not this plan.

## Data assembled locally

One million candidate pairs were fetched from pinned public TRAIN splits.
Each source contributes distinct canonical queries, rather than expanding the
same query over combinatorial negatives. Shuffle seed is 19, buffer 8192;
this is a bounded streaming sample, not a full-corpus uniform reservoir.
`sources.lock.json` records upstream revisions, fields and quotas.

| Source | Candidate pairs | Relationship |
|---|---:|---|
| MS MARCO | 250,000 | Search relevance |
| GooAQ | 200,000 | Question/answer, weak supervision |
| Natural Questions | 75,000 | Question/passage |
| SQuAD | 50,000 | Question/passage |
| Stack Exchange | 100,000 | Duplicate technical questions |
| S2ORC | 200,000 | Scientific title/abstract, weak supervision |
| Quora | 50,000 | Duplicate questions |
| NLI | 75,000 | Entailment only, auxiliary |

The pipeline normalizes Unicode/markup/whitespace, removes empty/short/pathological
pairs and identical texts, and verifies source hashes. NLI contradiction/neutral
labels and supplied mined negatives do not enter this first positive-pair set.
Mining and reviewing hard negatives is a separate data stage; no new negative
is labeled irrelevant solely because it was not the selected positive.

`audit.json` records 999,267 retained pairs: 979,295 train and 19,972 development.
595 rows overlap benchmark text under the actual tokenizer/128-token input;
65 duplicate visible pairs and 73 pairs becoming identical after tokenization
were also removed. The audit excludes text from SciFact, NFCorpus, ArguAna,
STSBenchmark, and reserved FiQA/SCIDOCS. It reads no relevance judgments.
Benchmark exclusion is conservative and does not guarantee exhaustive semantic
near-duplicate removal.

Train/dev assignment operates on connected components of query and passage
identities under the actual tokenizer, holding shared passages and transitively
related query groups together. There are zero shared model-visible texts across
the two splits. The largest component contains 2317 rows. This is a stronger
split than grouping query strings alone, but is still training-domain validation.

Long passages need special handling: 202,594 positives reach the 128-token cap,
including many NQ/SQuAD passages and scientific abstracts. The conservative
`capfit` subset keeps both full inputs strictly below the cap. This includes
excluding inputs exactly 128 tokens long, to guarantee no hidden truncation.
It is the proposed starting dataset; long pairs remain available for later
answer-preserving window extraction rather than blind prefix truncation.
`capfit-audit.json` records 796,672 complete-input pairs: **780,743 train and
15,929 development**. The source pool excluded 202,595 rows with a query or
positive at the cap; long pairs stay in the broader audited pool.

Raw text, local caches, train/dev JSONL and smoke weights stay gitignored under
`training-v2/work/data-first/`. Versioned files contain the plan, hashes and audits.
The million-pair set is an initial direct-supervision corpus, not a claim of
more total text exposure than the original 4.25M-text/vector-distillation run.
It is much broader than the recent 50K-triplet follow-up. Future scaling should
count unique pairs and actual tokens, and maintain these quality checks.

These checks establish provenance and eliminate specific data defects; they do
not establish that every weakly supervised pair is relevant or quality improved.

## Training preflight

`pair_train.py` performs direct query-to-passage contrastive training. It builds
single-source batches so negatives share a domain rather than being trivially
separated by source. Sources are sampled in proportion to retained row counts.
Known positive passages are marginalized, including other known positives from
outside the current batch. The checkpoint's float/ternary mode is preserved;
no implicit precision conversion occurs.

Six focused tests pass, including transitive split isolation, order-independent
group assignment and multiple-positive loss behavior. Eight-step local CPU
smokes passed with the independent float checkpoint and the existing ternary
checkpoint: finite losses/gradients and exact saved-checkpoint reload. Smoke
weights are disposable and are not promoted or evaluated as improved models.
The smoke uses 512 rows and batch 8; larger GPU memory/throughput is unmeasured.

Next training comparison: original versus direct pair supervision on the broader
cap-fit data, with unchanged shape, explicit token exposure and full-corpus
training-dev retrieval. Preserve the original ternary and independent float
checkpoints. Compare multiple training budgets before adding depth. Reserve
FiQA and SCIDOCS relevance scores for a frozen candidate; treat the already
observed retrieval tasks as development benchmarks. Int4/int8 derive from the
best float branch; ternary remains a separately adapted branch.

All preparation and smoke work was local. No further paid training is authorized.

## Reproduce

```bash
ternary-embedder/.venv/bin/python ternary-embedder/training-v2/data_first_fetch.py
ternary-embedder/.venv/bin/python ternary-embedder/training-v2/data_first_build.py
ternary-embedder/.venv/bin/python ternary-embedder/training-v2/data_first_capfit.py
ternary-embedder/.venv/bin/python -m unittest discover -s ternary-embedder/training-v2 -p test_data_first.py
```

The fetch reuses completed files only if the request and checksum match. Builds
refuse to overwrite a completed audit. The training runner does not allocate a
GPU or authorize spending. Its default device is local CPU.

## RunPod job prepared (2026-10-25)

The user requested RunPod training and periodic recovery copies. The
[concrete job and recovery policy](RUNPOD.md) are prepared; the user approved the $1.50/four-hour cap and the bounded job has completed. The training runner now saves
atomic resumable checkpoints every 500 updates or two minutes, and the local
controller attempts verified off-pod backups every minute. Interruption/resume
was tested to produce identical CPU model and optimizer state. Training covers
complete shuffled epochs rather than sampling rows repeatedly with replacement.


## Completed training

The approved RunPod job finished both branches and deleted the pod, at estimated total cost $0.1893. External core retrieval improved from 42.13 to 43.76 ternary and 44.44 to 45.47 float; q8 reaches 45.54. Neither target is cleared yet. [Complete results and model artifacts](RESULTS.md).
