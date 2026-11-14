# Approved Qwen-teacher pilot — 2026-10-24

**The candidate failed acceptance and is not promoted.** Keep the current
Grid 1 checkpoint as the default. This pilot has not met the goal of beating
bge-micro-v2 and MiniLM-L6-v2.

## External quality, frozen checkpoint

All scores use the existing evaluator and a 128-token cap. Retrieval ranks
all documents for all judged test queries. STSBenchmark uses its 1379 test pairs.

| Metric | Current model, CPU | New candidate, CPU |
|---|---:|---:|
| SciFact nDCG@10 | 54.85 | 53.45 |
| NFCorpus nDCG@10 | 29.40 | 28.23 |
| Equal-task mean retrieval | 42.13 | 40.84 |
| STSBenchmark Spearman ×100 | 81.58 | 81.35 |

The frozen candidate is `continued-8B-qat/best.pt`, update 500, SHA256
`e796c96824c9e7da8c69eac49bd8fbf685479b4423499ab645424af0e60146fa`.
It was selected before these tests, using only the MS MARCO training-source
query-group holdout. No other pilot checkpoint was searched on the acceptance
benchmarks. Full results and paired query intervals are in `candidate-results.json`.

The NPU implementation passes the new checkpoint's CPU-vector checks, with
minimum cosine 0.999795 (cap 32) and 0.999876 (cap 128). Warm short-text latency
is 1.33 ms at batch 1 / cap 128. This is a microbenchmark, not a fresh full-corpus
NPU quality evaluation. The CPU quality failure makes promotion inappropriate.
The CLI accepts `--checkpoint` for explicit experimental use; its default is
still the original model.

## What the training holdout showed

The pilot used 4096 public MS MARCO train triplets, with 204 complete query
groups reserved for validation. The metric is MRR among each fixed validation
batch's positive and negative candidates, not full-corpus MS MARCO MRR.

| Model / route | Best fully quantized holdout MRR |
|---|---:|
| Current Grid 1 QAT model | 0.7289 |
| Float control, before conversion (reference) | 0.7657 |
| 0.6B teacher, float start → gradual QAT | 0.7340 |
| 8B teacher, float start → gradual QAT | 0.7638 |
| 8B teacher, float start → abrupt QAT | 0.7499 |
| 8B teacher, continue existing QAT directly | **0.8123** |

The initial 400-update runs exposed insufficient quantization adaptation.
Within the same authorized pod and deadline, the three arms were extended to
4000 updates with a restarted QAT learning-rate schedule, and a direct-QAT
continuation control was added. The direct continuation avoided throwing away
the existing quantized function and gave the strongest training-holdout signal.
Its selected update 500 saw about 2.59M real training tokens.

Those improvements **did not transfer** to SciFact, NFCorpus or STSBenchmark.
This implicates the current recipe's generalization; it does not prove an
architecture ceiling or that an 8B teacher cannot help. Narrow data, preservation
strength, query-instruction asymmetry and ranking-objective design remain
confounded. The 8B-versus-0.6B comparison is controlled in the gradual arms;
there was no 0.6B direct-QAT continuation arm, so the direct-continuation gain
cannot be attributed solely to increasing teacher size.

## Hardware, recovery and cost

Qwen 8B encoded 12,257 unique training texts / 749,445 real tokens in 314.8 s
on the A5000. Peak allocated CUDA memory was 15.27 GB. Teacher and student
occupied the GPU in separate processes. The deployed student stays at
15,467,520 parameters, two layers, width 384 and 384-dimensional vectors.

The first pod failed during package setup before training. It was deleted
and the same approved job retried within the original deadline. GPU compute
for both attempts is estimated at **$0.132**; 60 GB of ephemeral disk adds a
small charge (roughly $0.004 at the published [$0.10/GB/month rate](https://www.runpod.io/pricing)). This remains below the approved
$0.50 cap. Both pod IDs returned HTTP 404 after deletion; no paid volume remains.
The job used about 30 minutes of its one-hour wall limit.

Recovered and SHA256-verified: 38 bundle artifacts and six data/teacher-target
files. Five checkpoints were retained: the best full-QAT checkpoint from each
long arm and the 8B gradual arm's float checkpoint. Redundant intermediate/final
snapshots were inventoried by hash and excluded from transfer. Logs, configs,
source, dataset identities and environment versions are retained. Summary:
`pilot-summary.json`; detailed run bundle: `runs/approved-pilot/` (gitignored).

## Follow-up proposal — subsequently approved

`run_balanced_round2.sh` starts from the original QAT checkpoint and tests:

1. 50,000 training triplets: 50% MS MARCO, 25% NLI, 25% duplicate questions.
2. Prompted 8B targets, original anchor weight 0.1.
3. The same targets with anchor weight 1.0 to preserve the original geometry.
4. Plain-query 8B targets with anchor 1.0, testing symmetric input formatting.

The last three items are three matched 4000-update training arms; all use the
same data and architecture. Passage vectors are reused exactly between teacher
formats. Selection stays on the training-source holdout, with per-source metrics.
Previously inspected acceptance tasks remain development benchmarks. Reserve
an additional unseen retrieval task (ArguAna) to assess later generalization;
do not select the next checkpoint using its labels.

Public-source schemas, passage-vector reuse, mixed-source teacher caching,
finite-gradient QAT continuation and exact checkpoint reload have been smoke-
tested locally with the 0.6B teacher. The 8B cache and training use the GPU path
already exercised in this pilot. Separate approval was subsequently received; the bounded round-2 allocation has finished and its pod is deleted. The frozen candidate failed core retrieval acceptance. See [round-2 status](round2/README.md).

Proposed separate authorization: one A5000 at at most $0.30/hour, at most two
hours, expected approximately $0.45–$0.65 including temporary disk, **$0.75 total
cap**. The controller's `balanced-round2` profile has a distinct state file,
105-minute work deadline and 115-minute independent deletion watchdog, with
retry margin before the two-hour limit. Approval of the completed pilot is
not authorization for this new job.
