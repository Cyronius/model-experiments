# Larger dense-model projection screen

Status: **complete** as `mc-scale-9b5ca68c9da9`; the pod is deleted. All three
models and all nine primary arms finished. Larger scale helped at 75% retention,
but neither large architecture reached parity and all half-size arms failed.
See the [results and interpretation](summary.md). This was a projection-only
experiment, with no healing or MLP-neuron pruning.

## Question and comparisons

Does the corrected shared residual-space projection retain more of the original
model's behavior at larger scale? Run stock, approximately 75%-retained, and
approximately 50%-retained text models. Count parameters, including embeddings,
the output head and dense output-norm matrix, at the same storage precision.
Keep all original layers, attention heads, head dimensions and MLP neurons.

| Source | Text parameters | 75% target: width / parameters | 50% target: width / parameters |
|---|---:|---:|---:|
| Qwen3.5-2B control | 1,881,825,088 | 1520 / 1,399,093,040 (74.35%) | 1008 / 927,454,512 (49.28%) |
| Qwen3.5-27B | 26,895,998,464 | 3824 / 20,103,074,288 (74.74%) | 2544 / 13,371,419,888 (49.72%) |
| Mistral Small 24B Instruct 2501 | 23,572,403,200 | 3824 / 17,620,261,616 (74.75%) | 2544 / 11,719,009,776 (49.71%) |

Widths are rounded down to multiples of 16, so actual retention is reported
rather than calling these exactly 75% and 50%. Vision and MTP modules are outside
the text-model count. Revisions and exact counts are in `*-plan.json`.

Qwen27 is tested within the Qwen3.5 family; Mistral is the different-architecture
check. Qwen2 and Qwen27 also differ in depth, head counts and embedding tying,
so this is evidence about these checkpoints, not an isolated universal scaling
law. The same shared-basis/output-matrix formula is used for every model.

## Frozen data and decisions

`data.json` contains the exact conversations and task checks, selected before
candidate outputs. Source: pinned UltraChat train_sft and test_sft, with the
same conversations for every model and no first-prompt overlap across splits.

- Calibration: 141 complete conversations, 65,968 Qwen tokens and 89,157 Mistral
  tokens. Mistral's native system message explains much of the token difference.
- Held-out distribution check: 32 complete conversations, 16 evenly spaced
  final-assistant positions per conversation, including EOS. Full-distribution
  KL, target NLL and top-1 agreement are reported separately from task behavior.
- Generation: 64 mechanical/rubric tasks plus eight long-form prompts requiring
  human review. Includes the previous 16 pilot cases and new arithmetic, literal
  copying, JSON, sorting, Python and multi-turn recall cases.
- Greedy decoding, thinking off, no repetition penalty; 512-token cap for the
  64 short cases and 1,024 for the eight long cases. All raw answers are retained.
- Loop flags use the earlier four-gram diagnostic; they are not infallible and
  must be reviewed against stock responses. Long-form answers are not given an
  automatic correctness label.
- Mistral's regex correction is enabled, and its template's date is frozen to
  2026-10-24. Default system messages are preserved. Every model's encoded-data
  digest is checked before loading weights.

A screening pass requires retaining at least 95% of cases the original model
passed, no increase in loop flags, and at most one fewer completed answer. New
successes do not cancel out lost stock successes. Passing means broader
evaluation is warranted; 72 prompts cannot establish parity. Lower KL alone is
never a pass. The Qwen scale comparison also reports retention on the subset of
tasks both stock models answered correctly.

## Numerical and execution checks

1. Load one original model in BF16, requiring matched text weights and the
   preflight parameter count. Record any intentionally excluded vision/MTP keys.
2. Compute the shared, uncentered PCA with equal covariance trace weight per
   residual site, then measure RMS corrections on the same calibration data.
3. Use the full orthogonal basis in FP32. Require maximum logit error below
   0.005 and maximum per-token KL below 1e-6, including cached-inference checks.
   Qwen uses its FP32-capable reference delta kernels on both sides of this gate.
4. Check full-width BF16 rounding separately against the original BF16 teacher:
   mean KL <= 0.02 and top-1 agreement >= 95%. A failed precision gate blocks
   interpretation of compressed arms.
5. Reload the original separately for each compressed arm; never compress an
   already compressed model. Projection matmuls run in FP32 chunks; resulting
   weights and output matrix use BF16. Preserve native rotary-buffer precision.
6. Cached-inference checks precede candidate generation. Numerical or loading
   failures abort the paid worker, rather than silently changing the experiment.

The GPU run retains basis, scales, projected configs, metrics, generations,
software versions and source revisions. These reconstruct the unhealed weights;
full checkpoint copies are not transferred during the short GPU lease. Tiny
models verify adapter save/reload and stock export separately.

## Approved bounded paid job

- One Secure Cloud H200 SXM, 141 GB VRAM, GPU rate ceiling **$5.29/hour**.
- 200 GB temporary container disk, about **$0.0278/hour** at $0.10/GB/month.
- Expected planning allowance: 60–90 minutes, approximately **$5.32–$7.98**.
  GPU runtime has not yet been measured; this is a bounded screen, not a promise
  that all three models finish within the allowance.
- **$10 total cap**, with a 90-minute automatic termination deadline starting
  before the create request. No automatic extension or persistent volume.
- Download/install time counts. Run Qwen2 first to detect CUDA/kernel problems
  cheaply, then Qwen27, then Mistral24.
- A separate local watchdog is started before creation. It terminates only the
  newly created, uniquely named pod on failure or deadline, and retries until
  the API confirms deletion. The main supervisor also requests termination in
  `finally`. Artifact copying never postpones termination.
- The RunPod API key stays local in its existing private credential file. It is
  neither uploaded nor included in the bundle. A temporary SSH key is used.

Prices were checked against the RunPod read-only API on 2026-10-24; see
`gpu-quotes.json`. Public references:
[GPU pricing](https://www.runpod.io/pricing),
[storage pricing](https://docs.runpod.io/pods/storage/types).

## Exploratory diagnostics added during execution

The frozen primary data, arms and screening criteria were not changed. After
observing punctuation loops, the summary script also counts runs of at least
eight identical nonempty lines. This post-hoc diagnostic is reported separately
and applied to stock and all compressed arms; it does not replace the original
word-based loop flag.

A four-prompt Qwen2 batch-versus-single check investigated a stock arithmetic
error. A separate output-only check compares the retained-space output matrix
with an exact gain-folded untied readout, using unchanged teacher activations
from the first two held-out conversations. Neither diagnostic is a healing run
or a new primary experimental arm. They are small diagnostic samples, not
generation-quality evidence for an alternative student. Their concurrent use
of the same GPU means recorded durations are not a throughput benchmark.

The output-only diagnostic found a larger readout effect for Mistral. A further
post-hoc Mistral 75% ablation therefore folds the original gain into its untied
head exactly, keeps an identity output matrix to match the primary parameter
count, and evaluates all 512 held-out positions plus the 64 frozen short tasks.
It does not include long-form generation. This ablation is reported separately;
it is not substituted into the primary architecture/scale comparison.

`manual-review.json` in the job directory classifies the lost stock successes
of both large 75% arms into clear content errors, formatting/length failures,
and an ambiguous list-order case. This review was performed by the assistant
after seeing outputs, without blinding. Strict check counts include benign
punctuation differences and must not be described as general model accuracy.

Raw `compare()` records contain `numerical_pass`, which uses the strict FP32
rotation tolerance. For BF16 cache checks the actual gate is maximum token KL
below 0.05 for stock, and below max(0.05, five times stock) for a projected arm.
For the full-width BF16 rotation the gate is mean KL <= 0.02 and top-1 agreement
>= 95%. A false strict FP32 flag inside a BF16 record is not itself a failed
BF16 execution gate.

## Reproduction

From the repository root, the default command only builds the reviewed bundle
and prints the proposal. It does not create or resume any paid resource:

```bash
target/manifold-audit/env/bin/python manifold-compression/run_scale_pod.py
```

Only after explicit approval of this exact bounded job:

```bash
target/manifold-audit/env/bin/python manifold-compression/run_scale_pod.py --execute --approved-budget-usd 10
```

The `--execute` switch is an operator control, not a substitute for the user's
approval. The user-supplied AGENTS.md requires the rate, duration/cost and cap to
be presented before paid action; the earlier quick-test allowance is not used
as standing authorization.

Results sync to a uniquely named subdirectory here. Summarize a completed run:

```bash
target/manifold-audit/env/bin/python manifold-compression/summarize_scale_probe.py manifold-compression/results/scale-probe-20261024/mc-scale-JOB_ID
```
