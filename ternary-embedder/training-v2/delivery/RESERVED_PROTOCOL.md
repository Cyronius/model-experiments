# Reserved retrieval confirmation protocol

This protocol is written before opening the reserved judgments. It confirms a frozen development winner; it does not select another model.

## Frozen candidate and references

Candidate: `models/campaign-20261025/model-q8.pt`, SHA-256 `38e400de8c984ec7e7dd9aa33e2163096b8dc3725b1e89fdf416fda0f6117d70`.

References:

- `TaylorAI/bge-micro-v2`, revision `3edf6d7de0faa426b09780416fe61009f26ae589`.
- `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

Q8 qualifies for this confirmation because its development point estimates exceed both references on SciFact, NFCorpus, STSBenchmark, and the fixed 12-task mean. The physical artifact hash and supporting development-report hashes must be recorded in an activation receipt before extracting or reading judgments.

## Datasets and inference

Use the previously reserved **FiQA test** and **SCIDOCS test** judgments, with their full local corpora. Corpus/query texts were available for training-exclusion checks; their judgments were not used in this campaign. Public historical pretraining exposure cannot be ruled out.

Use the existing retrieval pipeline and common 128-token cap. Concatenate document title and text as in the development tests. Use mean pooling and unit normalization for the two reference models, and the candidate's existing embedding output. Do not add query instructions, query prefixes, reranking, filtering, or an experimental query head. Keep the existing deterministic tie and self-document handling.

Only quality is measured here. Run under the shared CPU reservation; do not report operational elapsed time as a latency benchmark. Cached vectors must be tied to model identity and corpus/query text hashes.

## Report and interpretation

Primary descriptive result: the equal-weight mean of the two datasets' nDCG@10 scores, in percentage points. Report each dataset separately as well. For candidate-minus-reference differences, report paired query-bootstrap intervals, with 5,000 resamples and a fixed recorded random seed. A combined interval resamples queries independently within each of these two fixed datasets and then averages their score differences. It does not quantify uncertainty over arbitrary domains, training seeds, or model-selection history.

Report whether the candidate exceeds each reference on each dataset and on the two-dataset mean. If intervals overlap zero, describe the difference as uncertain. If a reference wins, report it directly. Do not tune or select another checkpoint after opening these judgments, and do not describe this small confirmation suite as full-MTEB superiority.

After activation, mark these judgments as opened in the campaign's final report. They are no longer reserved for future development.
