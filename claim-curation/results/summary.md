# XDNA2 claim-curation pilot

Local CPU inference; no new fine-tuning or paid compute.

The fixed dataset contains 12 passages: nine excerpts from five project notes and three synthetic controls. There are 33 named claim anchors and 10 separately supplied claim pairs. Annotations were written by the coding agent before benchmark inference; they are not independent human ground truth.

| Model | Valid extraction JSON | Named anchors matched | Conditions retained too | Valid citations/records | False claims on questions-only input | Pair labels correct | Capped outputs |
|---|---:|---:|---:|---:|---:|---:|---:|
| nuextract2b | 12/12 | 9/33 | 6/33 | 9/11 | 2 | 4/10 | 0 |
| nuextract3 | 12/12 | 30/33 | 27/33 | 66/66 | 0 | 8/10 | 0 |
| qwen4b | 10/12 | 26/33 | 17/33 | 26/35 | 0 | 10/10 | 0 |

A named anchor is a preregistered set of regex checks on the generated statement; source quotations are not counted as an answer. The conditions column additionally requires the specified qualifiers and exact source/condition fragments. These are diagnostic coverage checks, not semantic precision or full claim recall. A source quotation can be accurate while the generated statement misrepresents it. Every exported record stays proposed.

The table uses a post-inference matcher refinement: outcome words such as works/worked are bounded so that 'workers' cannot satisfy a successful-outcome anchor. Original frozen gold and original scores are retained in gold.json and extraction-scores.json; refined anchors, rationale and scores are separate files. This fixes a known false positive; it does not turn lexical scoring into semantic adjudication. See [qualitative review](../qualitative-review.md) for preserved-context and support failures.

The relation task receives supplied claims. It measures pair classification separately, not end-to-end discovery of contradictions from raw documents. Version/workload differences and unknown scope are deliberately included.

All records cite exact spans in retained input snapshots. Upstream paths and commits mentioned in project notes are preserved verbatim, but those upstream sources were not independently fetched or verified. These inputs are project notes/digests, not uniformly original experiment logs.

Prompts, examples, gold anchors and pair labels are fixed in prompt.json and data/. Only separate synthetic development fixtures were used for prompt-format calibration. NuExtract3 was added after the older 2B model returned empty lists on the first real passages. Models use their native chat formats with the same task instructions, templates and development examples.

## Per-model details

### nuextract2b

Generated 11 claim records; 2 records cover multiple named anchors and need atomicity review.

| Passage | Named anchors | Conditions and citations | Records | Schema valid |
|---|---:|---:|---:|---|
| core-dma | 0/3 | 0/3 | 0 | True |
| shim-wrap | 0/3 | 0/3 | 0 | True |
| granite-stack | 0/3 | 0/3 | 0 | True |
| softmax-stack | 0/3 | 0/3 | 0 | True |
| transfer-size | 0/4 | 0/4 | 0 | True |
| l1-budget | 0/3 | 0/3 | 0 | True |
| rot13 | 0/4 | 0/4 | 0 | True |
| softmax-mask | 2/2 | 0/2 | 1 | True |
| rounding | 3/3 | 2/3 | 3 | True |
| synthetic-negation | 2/3 | 2/3 | 2 | True |
| synthetic-questions | 0/0 | 0/0 | 2 | True |
| synthetic-version-scope | 2/2 | 2/2 | 3 | True |

| Pair | Expected | Actual | Correct |
|---|---|---|---|
| same-run-different-result | contradicts | contradicts | True |
| different-workload | different_scope | different_scope | True |
| different-versions | different_scope | different_scope | True |
| unit-equivalence | supports | different_scope | False |
| unknown-scope | insufficient_information | different_scope | False |
| same-run-timing | contradicts | different_scope | False |
| different-paths | different_scope | different_scope | True |
| partial-versus-universal | contradicts | different_scope | False |
| untested-not-failed | insufficient_information | contradicts | False |
| table-paraphrase | supports | different_scope | False |

Observed generation batches took 527.4 seconds for 1,072 output tokens. Runs may overlap and include prefill and padding; these timings are operational observations, not a controlled speed ranking.

[All raw outputs and rendered prompts](nuextract2b/raw.jsonl) · [All claim records](nuextract2b/claims.jsonl) · [Citation failures](nuextract2b/quarantine.jsonl) · [Execution configuration](nuextract2b/execution.json)

### nuextract3

Generated 66 claim records; 2 records cover multiple named anchors and need atomicity review.

| Passage | Named anchors | Conditions and citations | Records | Schema valid |
|---|---:|---:|---:|---|
| core-dma | 3/3 | 3/3 | 5 | True |
| shim-wrap | 3/3 | 3/3 | 13 | True |
| granite-stack | 2/3 | 2/3 | 3 | True |
| softmax-stack | 3/3 | 1/3 | 5 | True |
| transfer-size | 4/4 | 3/4 | 8 | True |
| l1-budget | 3/3 | 3/3 | 5 | True |
| rot13 | 3/4 | 3/4 | 9 | True |
| softmax-mask | 2/2 | 2/2 | 6 | True |
| rounding | 3/3 | 3/3 | 6 | True |
| synthetic-negation | 3/3 | 3/3 | 3 | True |
| synthetic-questions | 0/0 | 0/0 | 0 | True |
| synthetic-version-scope | 1/2 | 1/2 | 3 | True |

| Pair | Expected | Actual | Correct |
|---|---|---|---|
| same-run-different-result | contradicts | contradicts | True |
| different-workload | different_scope | different_scope | True |
| different-versions | different_scope | different_scope | True |
| unit-equivalence | supports | supports | True |
| unknown-scope | insufficient_information | contradicts | False |
| same-run-timing | contradicts | contradicts | True |
| different-paths | different_scope | different_scope | True |
| partial-versus-universal | contradicts | contradicts | True |
| untested-not-failed | insufficient_information | contradicts | False |
| table-paraphrase | supports | supports | True |

Observed generation batches took 782.5 seconds for 3,667 output tokens. Runs may overlap and include prefill and padding; these timings are operational observations, not a controlled speed ranking.

[All raw outputs and rendered prompts](nuextract3/raw.jsonl) · [All claim records](nuextract3/claims.jsonl) · [Citation failures](nuextract3/quarantine.jsonl) · [Execution configuration](nuextract3/execution.json)

### qwen4b

Generated 35 claim records; 5 records cover multiple named anchors and need atomicity review.

| Passage | Named anchors | Conditions and citations | Records | Schema valid |
|---|---:|---:|---:|---|
| core-dma | 3/3 | 3/3 | 5 | True |
| shim-wrap | 0/3 | 0/3 | 0 | False |
| granite-stack | 2/3 | 2/3 | 2 | True |
| softmax-stack | 3/3 | 0/3 | 2 | True |
| transfer-size | 4/4 | 0/4 | 5 | True |
| l1-budget | 2/3 | 1/3 | 6 | True |
| rot13 | 4/4 | 4/4 | 5 | True |
| softmax-mask | 0/2 | 0/2 | 0 | False |
| rounding | 3/3 | 2/3 | 4 | True |
| synthetic-negation | 3/3 | 3/3 | 3 | True |
| synthetic-questions | 0/0 | 0/0 | 0 | True |
| synthetic-version-scope | 2/2 | 2/2 | 3 | True |

| Pair | Expected | Actual | Correct |
|---|---|---|---|
| same-run-different-result | contradicts | contradicts | True |
| different-workload | different_scope | different_scope | True |
| different-versions | different_scope | different_scope | True |
| unit-equivalence | supports | supports | True |
| unknown-scope | insufficient_information | insufficient_information | True |
| same-run-timing | contradicts | contradicts | True |
| different-paths | different_scope | different_scope | True |
| partial-versus-universal | contradicts | contradicts | True |
| untested-not-failed | insufficient_information | insufficient_information | True |
| table-paraphrase | supports | supports | True |

Observed generation batches took 855.6 seconds for 3,321 output tokens. Runs may overlap and include prefill and padding; these timings are operational observations, not a controlled speed ranking.

[All raw outputs and rendered prompts](qwen4b/raw.jsonl) · [All claim records](qwen4b/claims.jsonl) · [Citation failures](qwen4b/quarantine.jsonl) · [Execution configuration](qwen4b/execution.json)
