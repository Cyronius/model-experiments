# Broader retrieval data: completed controlled training

Prepared **3,771,503 unique training pairs**, **27,155 validation rows**, and **48,409 complete new held rows**. The model stays at three layers, 384 dimensions and 17,241,984 parameters. The approved four-arm job finished at an estimated **$3.5242**, within its $10 ceiling. The pod was deleted and deletion verified by HTTP 404. Full trained checkpoints are verified on local storage. No candidate passed the wider development gate. See [results](RESULTS.md), [follow-up analysis](ANALYSIS.md), and [current state](active-work.json).

| Source | Retained training pairs |
|---|---:|
| sentence-transformers/gooaq | 1,292,150 |
| sentence-transformers/paq | 979,324 |
| sentence-transformers/msmarco-bm25 | 488,426 |
| sentence-transformers/s2orc | 331,317 |
| sentence-transformers/hotpotqa | 126,964 |
| HuggingFaceH4/stack-exchange-preferences | 116,106 |
| sentence-transformers/stackexchange-duplicates | 97,186 |
| sentence-transformers/natural-questions | 73,419 |
| sentence-transformers/all-nli | 71,188 |
| rajpurkar/squad | 65,627 |
| sentence-transformers/trivia-qa | 58,733 |
| sentence-transformers/quora-duplicates | 48,889 |
| sentence-transformers/squad | 19,758 |
| IFM/SFT-Reasoning | 2,416 |

Counts are globally deduplicated model-visible pairs, rather than raw downloaded rows or repeated epochs. The original clean corpus supplies retention examples. Broad QA adds new sources and recovers long positives only with original answer evidence. SQuAD uses the annotated answer occurrence; NQ/TriviaQA use original TRAIN answer aliases matched inside the original positive passage. Alias matching is weaker than independently verified relevance.

All new accepted inputs fit 128 tokens without blind truncation. Known benchmark texts and protected internal validation texts are excluded using exact tokenizer identities, normalized text and punctuation-normalized identities. Original source groups stay together; components containing known prior fine-tuning exposures stay in train. An independent final-file pass verified zero train/validation input overlap. These exclusions protect this continuation; they cannot undo benchmark-text exposure already present in its starting checkpoints. Historical pretraining and semantic near-duplicate exposure remain unknown.

K2-Horizon contributes a strict structured QA pilot, including reusable curated examples. The unstructured instruction-following sample was rejected because prompt/answer boundaries could not be established. The public LCQA examples validate the extraction path but remain a small separate pilot. A 5,000-passage generation queue and evidence validator are ready; no generation service was called.

The final cohort caps synthetic PAQ at 1M eligible pairs. Stack Exchange samples communities rather than letting the hundreds of StackOverflow shards dominate; only accepted or positively voted answers are retained, with original URLs and author attribution. Supplied mined negatives and lower-voted answers are not labeled irrelevant.

Yahoo Answers is excluded from this first mix after spot checks found nonresponsive joke answers. Generic Horizon refusals are filtered. New candidates receive pinned, checksum-verified encoding repair and safe conversion of serialized string lists into text; answer evidence must survive cleanup. Legacy retention strings are preserved unchanged. Source-level rejection and cleanup counts are recorded in the audit. These checks reduce known noise but do not certify every answer as relevant.

Two fresh reserved label sets are Touche2020 and TREC-COVID. Full qrels files remain unopened. A search preview exposed several Touche query1 judgments after training started; query1 is excluded from reserved confirmation. No training or candidate selection used that preview. FiQA, SCIDOCS and the previous 12-task subset are already observed development data. These checks do not establish full-MTEB superiority.

The completed comparison has four arms: ternary broad/repeated-old and float broad/repeated-old. Each starts from the exact final checkpoint of the previous campaign and sees one complete epoch with the same total pair count, seed 71, batch 1024, microbatch 128 and LR 1e-5. The repeat control preserves the old mixture proportions to rounding. Source composition and token lengths differ; this compares whole data recipes, not isolated novelty at identical source weights.

Best-checkpoint selection uses held-source macro nDCG with a 2-point per-source retention floor, including unchanged step 0. All four arms freeze before external development scoring. Reserved judgments open only for one frozen candidate that exceeds both references on every predeclared development retrieval task, STSBenchmark and the fixed 12-task mean. No training or selection follows reserved scoring.

CPU-only train/reload preflights passed for both precisions. The SHA-verified upload package is **1.64 GiB**. Smokes verify plumbing and recovery; they are not new quality results.

Approved resource: one RunPod Secure RTX 4090 24 GB, at least 32 GB host RAM, 80 GB temporary disk. Actual GPU rate is $0.89/hour; launch ceiling $0.90/hour. Expected 4–5.5 hours, about $3.61–$4.96 including temporary disk. **Approved total cap $10, maximum 10 hours**, with work stopped earlier and an independent deletion watchdog. No retained paid volumes. This is separate from the closed $9.58/$10 campaign.

Full model/optimizer/RNG/source-sampler checkpoints save every 100 steps or 120 seconds. Local SHA-verified backup attempts run every 60 seconds. Two latest complete resume states are retained. Every allocated pod must be deleted and GET404 verified before closing the job.

Review: [job plan](runpod-job.json), [data audit](data-audit.json), [preflight](preflight.json), [source locks](sources.lock.json), [extra source locks](extra-sources.lock.json), [PAQ cohorts](paq-cohorts.lock.json), [new evaluation texts](evaluation.lock.json), [quarantined samples](quarantine.json), [LCQA queue](lcqa-generation-queue.json).

Source terms are recorded with pinned revisions. H4 Stack Exchange attribution is preserved in work/broad-data/ready/attribution.jsonl. Several derivative cards omit license metadata; this preparation is not a blanket license grant or a commercial-release clearance.
