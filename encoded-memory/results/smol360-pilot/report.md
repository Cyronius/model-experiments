# Encoded-memory pilot: dev gate

Recipient: `HuggingFaceTB/SmolLM2-360M-Instruct`. Encoder: 481,616 trainable parameters, 16 slots. Fixed training budget: 150 steps.

Memory insertion site: `prefix`.

Meta Llama 3.2 1B was inaccessible to the available credentials (HTTP 403). This run uses the ungated SmolLM2 checkpoint and the standard Llama implementation. It is independent of the model-shrinking experiment.

The recipient remained unchanged: **True**. Training used 64 fictional records and 256 questions. All XDNA records and the test device/value bindings were held out from encoder training.

| Domain | Condition | Numeric correct | Strict exact |
|---|---|---:|---:|
| synthetic | none | 0/16 (0.0%) | 0/16 |
| synthetic | oracle | 0/16 (0.0%) | 0/16 |
| synthetic | retrieved | 0/16 (0.0%) | 0/16 |
| synthetic | text | 15/16 (93.8%) | 8/16 |
| synthetic | wrong | 0/16 (0.0%) | 0/16 |

Numeric scoring accepts units but rejects multiple numbers. Strict exact match accepts only the normalized answer.

- synthetic: 15 questions were missed without memory and answered with original text. Oracle recovery: 0.0; retrieved recovery: 0.0.

## Interpretation

This configuration did not demonstrate usable encoding of new facts: correct encoded memories did not outperform both the no-memory and wrong-memory controls on the evaluated synthetic questions. A retriever cannot repair that encoding/reading failure.

Swapping correct for deliberately wrong memory changed the generated answer in 0/16 evaluated questions.

The document-dependent component of the stored vectors was 0.132% of their overall RMS. This measures variation across documents at each slot/coordinate; it does not measure how many facts the vectors contain. Memories were stored in fp16.

The first gate failed, so generator answers on the reserved test and the real XDNA probes were not evaluated. The sample contains 16 development questions across 8 independent fictional records, querying alignment. No configuration change or additional training was selected using reserved test answers.

A separate, untrained BM25 diagnostic inspected source-ID retrieval on the reserved questions: synthetic: 64/64; xdna: 5/9 matched the designated source. Other findings may repeat valid evidence. Neither the retriever nor encoder was tuned using that check.

## Source embeddings at prefix

Exact source token embeddings at `prefix` answered 4/16 correctly (4/16 strict exact match). This control uses the full token sequence without learned compression.

Prefix placement itself damages recall relative to ordinary reference text. This run cannot isolate the compressor from the placement problem.

## Source embeddings at reference

Exact source token embeddings at `reference` answered 15/16 correctly (8/16 strict exact match). This control uses the full token sequence without learned compression.

## Limits and artifacts

The evaluated synthetic set has 8 independent documents, with correlated question variants. This report includes 0 numeric XDNA probes from 0 findings. No confidence or generalization claim should be inferred from these small counts. The development and test facts are separate.

The bank contains 20 encoded records (614,400 bytes of fp16 values). The source text is also retained in the BM25 index. Fewer input tokens does not imply fewer storage bytes than plain text. Timing numbers are local CPU measurements under variable system load, not a speed benchmark.

`predictions-dev.jsonl` contains every answer and selected record. `summary-dev.json` contains the aggregates. `training-summary.json` records the frozen-recipient weight hash. `data-manifest.json` records source location and generated dataset hashes. Regenerate data with the recorded seed and document counts; source findings may evolve.

Encoder weights and the memory bank remain under the ignored local run folder. There was no donor model, cloud training, or modification of the XDNA source repository.
