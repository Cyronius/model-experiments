# Data-first training results — 2026-10-25

**Retrieval improved; the bge-micro/MiniLM target is not yet met.** Both branches completed two full passes over 780,743 training pairs. The rented pod is deleted, deletion independently verified by HTTP 404. Total estimated GPU plus temporary-disk cost: **$0.1893** under the approved $1.50/four-hour cap. Elapsed paid time: 40.81 minutes.

## Matched external development results

128-token cap, full-corpus exact retrieval, CPU float32 reference arithmetic. Core mean equally weights SciFact and NFCorpus; it is not an overall MTEB score.

| Model | Previous core mean | New core mean | Change | SciFact | NFCorpus | STSBenchmark |
|---|---:|---:|---:|---:|---:|---:|
| float | 44.44 | 45.47 | +1.04 | 59.44 | 31.51 | 82.59 |
| ternary | 42.13 | 43.76 | +1.63 | 57.14 | 30.37 | 81.46 |
| q4 | 44.15 | 44.94 | +0.79 | 58.59 | 31.29 | 81.97 |
| q8 | 44.54 | 45.54 | +1.01 | 59.56 | 31.53 | 82.55 |
| bge_micro reference | — | 46.21 | — | 61.33 | 31.10 | 82.50 |
| minilm reference | — | 46.92 | — | 62.39 | 31.45 | 82.03 |

STSBenchmark is Spearman ×100 on 1379 pairs. Ternary STS is essentially flat (81.46 versus 81.58); float/q8 improve slightly. Q4/int8 quantize the six dense matrices per layer, with 64-weight groups and bf16 scales, plus the existing int8 activation simulation. Embedding tables, output projection, biases and norms remain float. Their scores are reference/simulation accuracy, not native low-bit speed measurements.

Paired query bootstrap 95% intervals for the change in core mean:
- float: +1.04, interval [+0.11, +1.95].
- ternary: +1.63, interval [+0.37, +2.90].
- q4: +0.79, interval [-0.13, +1.75].
- q8: +1.01, interval [+0.06, +1.96].

These intervals cover query sampling, not variation across training seeds. Q4's interval crosses zero, so its numerical gain is less certain. This is one seed and one training run.

The remaining core gap is concentrated in SciFact. Float/q8 slightly exceed both reference point estimates on NFCorpus, while trailing them on SciFact. ArguAna scores are 45.94 float and 43.77 ternary, versus 51.81 bge-micro and 48.76 MiniLM. This does not establish broad superiority. The reserved FiQA/SCIDOCS judgments remain unscored because no precision tier cleared both core retrieval gates. The additional 12-task MTEB run was not required after that gate failed.

## What the internal 89.25 score meant

It was held-source macro nDCG@10 ×100, not percentage accuracy or an external benchmark result. Each source ranks all its held queries against every unique positive passage from that source. This has many easy negatives: the old ternary model already scores 88.57 here while bge-micro scores 87.37, despite bge winning externally. MiniLM scores 92.37. Reference internal scores ran locally on CPU; training validation ran on GPU.

The selected float checkpoint is update **9000**, internal score 90.351 (initial 89.712). The selected ternary checkpoint is update **12206**, score 89.431 (initial 88.568). Both were chosen only by the predeclared rule: maximize equal-source nDCG while no source falls more than two points below its initialization. The initial checkpoint was eligible as a null control. Both selections were frozen after recovery/deletion and before external scoring. No milestone was substituted after seeing external quality.

## Training and recovery

Direct multi-positive contrastive supervision, no teacher model/target vectors/ranking KL or frozen-vector anchor. Two layers, width 384, FFN1536, 384 output, **15,467,520 parameters**. Both branches trained 12,206 updates at batch 128, with a folded final singleton where needed, for 1,561,486 pair presentations each. Source batches are shuffled and interleaved, with complete coverage each epoch. LR1e-5 with cosine decay, temperature0.05, seed29, BF16 CUDA autocast. Initialization remains the independent float150 checkpoint for float and the original QAT checkpoint for ternary.

Full resumable checkpoints were saved every 500 updates or two minutes. A local controller attempted verified backups every minute, pinning immutable checkpoint links and verifying SHA256 before publishing local copies. A real SSH/SCP timeout occurred late in training; the last verified copies survived, retries recovered later checkpoints and all final artifacts. Both final model reload checks passed. Local CPU interruption/resume testing had already reproduced weights, optimizer and RNG state exactly.

All run logs, best/final/milestone weights, and the last two resumable states per branch are under ignored `training-v2/work/data-first-run/backups/`. Full input hashes and source snapshots are retained in the prepared package and manifests. Credentials remained local. Only one pod was allocated; failed allocation calls created none. The controller used REST v2 plus an explicit official PyTorch image to handle the available CUDA13.2 host. The pod was Secure Cloud RTX A5000 at $0.27/hour.

## NPU and model artifacts

The selected ternary model scores **43.7009** core mean with actual NPU vectors, versus 43.7584 CPU. Minimum fixture cosine agreement is 0.999559. Warm short-text cap128/batch1 median is 1.287 ms; cap32/batch4 median is 1.687 ms for all four texts. This is a microbenchmark with observed scheduling outliers, not a full-corpus throughput comparison.

Experimental model files and hashes are in [`models/data-first/manifest.json`](../../models/data-first/manifest.json). Float and ternary use the existing `Student.load`; only the ternary checkpoint is supported by the current NPU encoder. Q4/q8 are packed reference exports loaded by `export_pair_precision.load_reference`; their reference vectors match the corresponding simulation bit for bit. A native q4/q8 runtime remains future work. The existing default checkpoint was not changed.

Reproduction/evidence: `results/frozen.json`, `results/summary.json`, `results/*-training.json`, `results/npu-quality.json`, `results/npu-fixtures-and-speed.json`, `results/resources.json`; scripts in the parent training-v2 directory.

The data-first change helped this run. The float-to-ternary core gap narrowed from 2.31 to 1.72 points. That is an observed recipe comparison, not an intrinsic precision limit. The next data investigation should emphasize harder scientific claim/evidence and same-topic retrieval negatives with false-negative checks, rather than merely expanding easy QA pairs. Any further paid job needs separate approval.
