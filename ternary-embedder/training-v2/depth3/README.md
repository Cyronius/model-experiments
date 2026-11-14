# Controlled third-layer experiment — running in the $10 campaign

The user requested an additional layer after the data-first run. Test depth
without attributing extra training alone to the new architecture.

## Comparison

Four arms, each with two additional complete passes over the same 780,743
cap-fit relevance pairs (15,929 held-out rows):

| Precision | Control | Grown model |
|---|---|---|
| Float | Current selected two-layer float, continued | Same parent plus a third layer |
| Ternary | Current selected two-layer ternary, continued | Same parent plus a third layer |

Width384, FFN1536, 12 heads, output384, 128-token limit. Two layers have
15,467,520 parameters; three have **17,241,984** (+11.5%). Transformer work
increases approximately 50%; whole-encoder latency must be measured.

Use the same direct positive-pair loss, batch128, LR1e-5 cosine schedule,
temperature0.05, seed29, BF16 CUDA autocast, source batches and two-epoch budget
as the preceding run. Every arm restarts AdamW, so the optimizer treatment is
matched. No teacher model, target vectors, or distillation loss. Preserve the
ternary/q4/q8 family; derive q4/q8 locally from frozen float branches.

`initializations.json` records all input hashes. Float parent is the preceding
run's update9000 checkpoint; ternary parent is update12206. For each precision,
the two-layer control and grown model share the same parent weights.
The extra block is an independent clone of the last trained block.

## Growth and selection

A training-only interpolation enables the third block over the first 500
updates: `x + alpha * (block(x) - x)`, alpha0→1. At alpha0, vectors exactly match
the two-layer parent. At alpha1, the block returns its ordinary output directly.
The final checkpoint is a standard three-layer Student with no inference gate
and no additional gate parameters.

The initial validation score for the grown branch therefore measures the
common two-layer parent function. Partial-ramp files are resumable training
states; they must not be used with the ordinary inference loader. Best/milestone
inference exports are saved only once alpha reaches1. Metadata records the ramp
position so interruptions during growth can resume correctly.

At initialization, every1500 updates and completion, rank each held source
against its full held-positive document corpus. Select maximum equal-source
nDCG@10 after alpha1, subject to every source remaining within two points of the
common parent. Two-layer controls include initialization as a null control.
If no fully grown checkpoint passes the guard, evaluate the predetermined final
three-layer model as an unsuccessful diagnostic and mark the guard failure;
do not search external scores for a different checkpoint.

Freeze all choices before external scores. Compare three versus two layers
within this same run, then report the earlier parents, bge-micro and MiniLM.
Report SciFact/NFCorpus/ArguAna and STSBenchmark. The existing conditional
reservation of FiQA/SCIDOCS remains: use them only when a frozen candidate
clears both core retrieval references. No broad MTEB superiority claim follows
from a two-task average. Repeat promising results with another seed later.

The current NPU graph supports two layers. This experiment first measures
quality and CPU latency. A three-layer NPU graph is separate implementation
work if the quality gain warrants it; the existing 1.3ms NPU latency must not
be assigned to the new architecture without measuring it.

## Recovery and proposed resources

Use atomic resumable checkpoints every500 updates or120 seconds. Attempt local
SHA256-verified copies every60 seconds. Recovery includes model, AdamW state,
sampling epoch permutations/cursors, LR position, CPU/CUDA RNG and growth ramp.
Keep two local recovery generations and validation milestone weights. To reduce
backup bandwidth, transfer only the current pinned resumable state each time;
the previous verified local generation remains available. Failed transfers do
not replace verified files.

Separate work/state/backups under `training-v2/work/depth3-run/`; the preceding
run's artifacts, approval and state remain untouched. An independent watchdog
receives the depth3 profile so it uses this job's budget and pod ID.

The user explicitly authorized a **$10 cumulative experimental campaign**, covering
this comparison and follow-up experiments. The initial A5000 allocation failed
without creating a pod. The active resource is one Secure Cloud RTX4090 24GB at
**$0.89/hour**, accepted maximum$0.90/hour. Expected0.75–1.5 hours/$0.70–$1.40;
this job reserves **$2.50**, with a2.5-hour hard maximum including temporary disk.
Work stops at130 minutes, independent deletion watchdog at140 minutes, leaving
another ten minutes for deletion retries. No retained paid volume. The campaign
ledger counts the full outstanding reservation until final deletion/cost recovery.

## Preflight and run entry points

Growth endpoint and gradient tests pass for float/ternary. An actual384-wide
ternary eight-step smoke interrupted at step4 (alpha0.8) resumes to bit-exact
weights, optimizer and RNG at step8. Its standard Student reload at alpha1 passes.
A float smoke exercises validation and excludes partial-ramp inference exports.
Sampler, atomic backup and profile-isolation tests pass. These are local CPU
pipeline checks, not training quality or CUDA throughput results.

```bash
ternary-embedder/.venv/bin/python ternary-embedder/training-v2/pair_runpod.py prepare --profile depth3
ternary-embedder/.venv/bin/python ternary-embedder/training-v2/pair_runpod.py status --profile depth3
```

The prepared controller refuses launch without a new matching job-specific
approval record. `run_depth3.sh` and the job specification in `runpod-job.json`
record the concrete workload. Prepared grown checkpoints are initialization
assets, not finished inference models.
