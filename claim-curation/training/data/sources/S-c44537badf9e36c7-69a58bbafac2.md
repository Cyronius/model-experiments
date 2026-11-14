---
from: phlegm
date: 2026-10-25
evidence: measured
stack: Strix Halo Ryzen AI MAX+395; Fedora44 kernel7.2.11; amdxdna0.10.0; firmware1.1.2.65; XRT2.27; MLIR-AIE1.4.2; Peano21.0.0.2026080301+c9c5ecb7
---

# Compensated prefill passes model gates; preliminary engine improvement

Canonical source: /home/cyrus/code/phlegm/open_kernels/designs/gemm_q4_prefill/
compensated64/, with explicit experimental bfp16_hilo runtime input contract.
Full report and reproducible commands: phlegm/.plans/artlite-performance.md and
specs/open-engine/gguf-prefill-linux.md. Dirty development artifacts, no release
benchmark or parity claim. Same Llama3.2 1B Q4_1 GGUF SHA256
6e35ca7e357c6c7f55a34d683e538614cb3e34b1155d92731da75c1685136add.

Compensated model300 median cosine .9999477636194,argmax297/300,
loss2.803070816;8192 median .9999503172159,argmax8191/8192,
loss .0353892735892. Independent CPU plus sequential controls pass chosen
cosine/argmax/loss gates; real47 slice across4096 also passes. Artificial
zero-history failure retained, not repaired. Rotated64x64 image matches all
8192 logits and16 decode outputs byte-for-byte; resident bridge16 also exact.
32core .text5136B/frame64B/stack4096B/L1end60488B. All4 projection streams
rebuilt and three CDO payloads checked identical for shared context.

Preliminary same-engine paired trials:1024 valid2/3,760.161 to1076.689
prefill tokens/s,1.41642x;8192 valid1/3,578.617 to755.154,1.30510x.
Decode approximately unchanged (ratios .98963/.99053). Insufficient stable
pairs: not accepted engine improvement. Raw evidence:
phlegm/target/npu-dev/survey-native-vs-hilo64-idle/{raw.jsonl,summary.json}.
These measurements predate the new Dibs workflow; Dibs-isolated repeat pending.

Actual ArtLite prefill independently captured300 real positions from one
512-padded dispatch through its own tied Q6_K NPU head with output norm;
last row byte-equal to untouched forward, no CPU-head substitution.
Source artlite-project/artlite a07a7b8893cb4a835352669d237609ba017e8ab1,
MLIR-AIE1.4.5.dev56+g1a647f1/Peano22.0.0.2026102001+4f5df13d.
CPU median cosine .9995212360605257,min .9924211767146753,
argmax290/300,loss2.801516981795352. Own sequential median .9998918179979034,
argmax293/300,loss2.8006815596276406. Own-seq gate fails cosine; common
Phlegm-seq gate fails cosine and argmax. Loss remains close: not a general
quality ranking, but relevant to strict precision-matched speed comparisons.
Gate JSON SHA256s:
67908b78379dd686d6a153d056bc1a5edc741784d55a71887b5ae85b6e84dfd0 (ownseq),
906c68cce21bd8cdbd70472ef60f624cd4649c613cbb8a887085092e7a9b53dd (commonseq).
Located phlegm/target/npu-dev/quality/gate-artlite-prefill300-*.json.
