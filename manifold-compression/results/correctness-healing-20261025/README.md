# Correctness-focused healing pilot — 32 updates complete

This experiment repairs the strongest saved 75% projection, Qwen3.5-27B
at residual width 3824 (20,103,074,288 text parameters, 74.74% of the source).
It preserves the projection and the corrected full final-normalization matrix.
The objective is supervised correctness on verified targets and improvement on
held-out tasks. The stock model is a reference; its output distribution is not
a training target.

**Final status:** all 32 planned updates completed. Development selected update
32, which scored **117/128 (91.4%)** on the fresh short-answer test, versus
103/128 at update 16, 92/128 before repair and 121/128 (94.5%) for stock.
Long-form loop flags fell from 3/6 to 1/6. Both pods are deleted; estimated
combined cost is **$11.93**, within the approved $15 cap. The selected adapter
reload is exact, and all 52 continuation artifacts verified. The newest
reconstructible checkpoint is
[update-32 adapter-best](mc-heal-resume-12789a78a60b/healing-extended/adapter-best/).

The first job, `mc-heal-4d3614e4c060`, completed 16 of the planned 32 updates,
stopping to reserve final-evaluation time rather than because of regression.
Pod `do2mevn0r9ac89` was deleted after 86.05 minutes, within the approved
90-minute/$10 limit. Estimated compute plus disk cost is **$7.63**; provider
billing was only partially posted at the final check. All 55 artifact hashes
verified, and the selected adapter's reload check reproduced logits exactly.
See [completion audit](mc-heal-4d3614e4c060/completion-audit.json),
[raw job results](mc-heal-4d3614e4c060/), and [authorization](authorization.json).

## First-segment results

| Arm | Development correct / 64 | Fresh test correct / 128 | Fresh test loop flags / 134 | Old strict screen / 64 |
|---|---:|---:|---:|---:|
| Stock | 60 | 121 | 0 | 63 |
| Full-width projection | 60 | 121 | 0 | 63 |
| 75% projection, before healing | 43 | 92 | 3 | 41 |
| 75% projection, selected update 16 | 49 | 103 | 3 | 55 |

The selected checkpoint saw 3,017 training examples, 52,805 assistant tokens
and 1,053,473 input tokens. Development correctness at updates 0/4/8/16 was
43/43/40/49. The frozen development rule selected update 16 before final tests.
The fresh-test improvement is 8.6 percentage points, with an 18-answer gap to
stock remaining. This small screen does not establish general parity.

The full-width arm lost **zero** stock-correct answers on development, fresh
test and the old screen, and introduced no loop flags. This is an observed
zero failure count on these prompts, not a universal guarantee. On 512 numerical
validation positions, mean KL was 0.000529 and top-1 agreement was 99.21875%
(four disagreements). Full-width reconstruction is therefore not bit-exact
under BF16. These generation controls cover Qwen27, not Mistral24.

Fresh-test paired results contain 12 gains and one loss. Manual review classified
them as nine value/order repairs, one completed correct answer, two presentation
repairs and one new sorting error. A supplemental syntax-tolerant content review
scores unhealed/healed as 93/103 instead of 92/103. It accepts whole Python
literals and unambiguous numeric sequences; it does not alter the primary score
or checkpoint selection. See [paired comparisons](mc-heal-4d3614e4c060/healing/comparison.json),
[manual review](selected16-manual-review.json), and [syntax review](syntax-review.json).

**Looping is not fixed.** The same Wi-Fi, interval-code and photosynthesis
prompts trigger flags before and after healing. Wi-Fi repetition worsens from
an ended answer to the 1,024-token limit; photosynthesis still loops to that
limit. The interval function remains invalid Python. A post-hoc executable
audit passes six cases for stock and full-width, while both 75% outputs fail
syntax parsing. The narrow short-answer training set has not demonstrated
long-form repair. See [interval audit](interval-audit.json).

The three flags occur among **six long-form prompts**. The 128 scored tasks
are short-answer checks, so 103/128 must not be interpreted as general long-form
reliability. A post-hoc family breakdown makes the remaining gaps clearer:

| Fresh-test family (16 cases each) | Stock | Before healing | Update 16 |
|---|---:|---:|---:|
| Multiplication | 16 | 0 | 3 |
| Sorting | 16 | 11 | 12 |
| Python execution | 16 | 11 | 16 |
| Unit conversion | 16 | 16 | 16 |
| JSON instructions | 16 | 16 | 16 |
| Grounded extraction | 16 | 16 | 16 |
| Conversation memory | 16 | 16 | 16 |
| Withheld Horizon code functions | 9 | 6 | 8 |

See [family review](family-review-step16.json). These small family samples
describe this screen; they are not general benchmark estimates.

The selected adapter is [adapter-best](mc-heal-4d3614e4c060/healing/adapter-best/).
The [last adapter](mc-heal-4d3614e4c060/healing/adapter-last/) and optimizer/RNG
state permit continuation from update 16. They are reconstruction artifacts,
not a standalone merged Hugging Face model.

## Completed continuation and final comparison

The original pod completed and deleted itself before the proposed budget
extension was approved. Its in-place extension is no longer available.
A replacement-pod job restored update 16 and completed the remaining 16
updates, reusing the completed controls and preserving the frozen data,
optimizer, objective, development selection and regression gates.

| Arm | Development correct / 64 | Fresh test correct / 128 | Fresh long-form flags / 6 | Old strict screen / 64 |
|---|---:|---:|---:|---:|
| Stock | 60 | 121 | 0 | 63 |
| Full-width projection | 60 | 121 | 0 | 63 |
| 75%, unhealed | 43 | 92 | 3 | 41 |
| 75%, update 16 | 49 | 103 | 3 | 55 |
| **75%, update 32** | **58** | **117** | **1** | **60** |

Update 32 has zero development loop flags and all 66 development responses
ended. On the fresh test, 133/134 responses ended, with no unparseable scored
answers. Relative to update 16, all 14 fresh-test gains are substantive
value/order repairs: nine multiplication, four sorting and one withheld code
output. No previously correct fresh-test answer was lost. Relative to unhealed,
there are 25 gains and zero losses; two gains are presentation repairs and one
is a previously unfinished correct answer, so the whole 25 should not be called
pure arithmetic/reasoning improvement.

Final family scores are 12/16 multiplication, 16/16 sorting, 16/16 Python
execution, 16/16 each for units/JSON/grounding/memory, and 9/16 withheld Horizon
code functions. Stock scores 16/16 for multiplication and 9/16 for withheld
code. Paired comparison with stock loses four multiplication answers and one
code answer while gaining one code answer. Equal family totals do not imply
identical answers. See [family review](mc-heal-resume-12789a78a60b/family-review.json)
and [paired comparisons](mc-heal-resume-12789a78a60b/healing-extended/comparison.json).

The narrow syntax supplement leaves the final score at 117/128; it raises the
unhealed score to 93/128. The old strict screen has six gains and one apparent
loss relative to update 16: the aliasing answer is substantively correct but
uses different wording from the grader's required `same list`/`same object`
phrases. Its preregistered 60/64 score remains unchanged. See
[syntax review](mc-heal-resume-12789a78a60b/syntax-review.json) and
[manual review](mc-heal-resume-12789a78a60b/manual-review.json).

Long-form behavior improved but remains unreliable. Wi-Fi now finishes and
falls below the loop threshold, while retaining duplicated advice. The
interval-merging response no longer triggers a loop flag but still mixes Python
and JavaScript and fails the executable audit's syntax check. Photosynthesis
has a substantially better opening, then repeats sections and reaches the
1,024-token limit. Automatic flags alone understate these remaining defects.
See [executable interval audit](mc-heal-resume-12789a78a60b/interval-audit.json).

The user requested a 4,096-token rerun to distinguish truncation from persistent
repetition. This inference-only diagnostic is prepared for the selected update-32
adapter: eight prompts flagged anywhere in this pilot, with twelve companion
prompts preserving the original six batches and their padding. Model weights,
prompts and decoding remain fixed. Completion, repetition and exact agreement
with each original answer's prefix will be reported separately. A higher token
limit alone is not assumed to fix or confirm repetition. The original scores
above remain unchanged. See the [request](../loop-retry-20261025/loop-retry.json),
[validation](../loop-retry-20261025/validation.json), and
[new job proposal](../loop-retry-20261025/job-proposal.json). The user approved
the separate $3.50 cap. Job `mc-loop-52581c3d31d1`, pod `ujoj3g8p7avcon`,
completed within its 35-minute limit and the pod is deleted. See
[authorization](../loop-retry-20261025/authorization.json).

**4k diagnostic result:** seven of the eight previously flagged prompts ended
naturally. Binary search finished at **1,285 tokens**, so its 1,024-token limit
was restrictive. Photosynthesis reached **4,096 tokens without EOS**, repeatedly
emitting “Photosynthesis is a mix-up”; its repeated-4-gram fraction was 0.9797.
Allowing more output did not resolve that repetition. Finishing an answer is
separate from producing correct content: the completed binary-search answer
still contains invalid Python, conflates binary search with binary search
trees/B-trees, and gives incorrect example indexing. Syntax checks also fail
for the interval, transaction-grouping and CSV functions in this diagnostic.
See [manual review](../loop-retry-20261025/mc-loop-52581c3d31d1/manual-review.json).

| Previously flagged prompt | Same-pod original-limit tokens | 4k-limit tokens | Outcome |
|---|---:|---:|---|
| Pattern generation | 14 | 14 | Ended identically |
| Group transactions | 674 | 674 | Ended identically |
| Wi-Fi troubleshooting | 679 | 679 | Ended identically |
| Merge intervals | 748 | 748 | Ended identically |
| Photosynthesis | 1,024, capped | 4,096, capped | Repetition persists |
| Parse CSV | 899 | 899 | Ended identically |
| Lost-key story | 359 | 359 | Ended identically |
| Binary search | 1,024, capped | 1,285, ended | Truncation removed; content still incorrect |

Historical saved long-form outputs were not reproduced verbatim after model
reconstruction. To isolate the limit effect, the same pod also reran the
original limits on all six batches, within the unchanged deadline and budget.
All **20 same-pod original-limit text outputs exactly matched prefixes of their
4k counterparts**. Only four of the twenty 4k outputs matched historical saved
prefixes (one of the eight target prompts). The historical mismatch's cause
has not been diagnosed;
this diagnostic does not revalidate the earlier 117/128 score or establish
bit-exact reconstruction of the whole model. The previous exact reload check
covered adapters on an already reconstructed model.

See [4k outputs](../loop-retry-20261025/mc-loop-52581c3d31d1/loop-retry/generations.json)
and [same-pod comparisons](../loop-retry-20261025/mc-loop-52581c3d31d1/original-limit-control/result.json).
The launcher synchronized and checksum-verified the artifacts before deletion.
The independent audit also verified all **46 artifacts** and confirmed pod
absence through the API. Duration was **25.26 minutes**, including the extra
same-pod controls; estimated cost was **$2.24**, within the approved $3.50 cap.
Billing had not posted at the check; the zero posted amount is not a zero-cost
job. Dibs holds are released. See
[completion audit](../loop-retry-20261025/mc-loop-52581c3d31d1/completion-audit.json).

Use a larger allowance when testing long answers to avoid confusing truncation
with failure to finish, but continue measuring repetition and content correctness
separately. Before relying on a reconstructed export, investigate full-model
reconstruction reproducibility with fixed-input logits and parameter hashes;
adapter-only reload equality does not establish that broader property.

The full 32 updates cover exactly one pass over 5,964 examples: 105,287 answer
tokens and 2,106,214 input tokens. This is evidence that correctness-focused
healing can recover much of this fixed projection's performance on the pilot.
It is not evidence of general parity. The final test has now been inspected;
future experimental choices should be judged on a new held-out set. The data,
update schedule and selection rule were frozen before this continuation.

Keep the selected checkpoint and current projection. The next useful work is
broader independent evaluation and verified training targets for long-form
generation, executable code generation and the remaining arithmetic errors.
These results do not establish a need to restart projection design or move to
a larger model. No additional paid job is authorized by this completed run.

The concrete replacement proposal is one Secure Cloud H200 SXM at at most
$5.29/hour plus about $0.0278/hour for 200 GB ephemeral disk, expected 50–65
minutes, hard limit 75 minutes. The combined cap is **$15**, reserving $8 for
the first run and at most $7 for this continuation. Planned maximum new usage
is $6.65; estimated combined usage is about $12.06–$13.39 at the expected
duration. No persistent volume or automatic replacement is allowed.
See [replacement proposal](restart-proposal.json).

The user approved this specific replacement job and $15 combined cap.
Pod `m15lynbrinlrdu`, job `mc-heal-resume-12789a78a60b`, completed and was deleted
after 48.56 minutes. Its estimated cost is $4.30, bringing the combined estimate
to $11.93. At the final check, the first run's posted bill was $7.6262 and the
continuation's bill had not posted; a zero posted amount is not a zero-cost run.
See [authorization](restart-authorization.json),
[completion audit](mc-heal-resume-12789a78a60b/completion-audit.json), and
[lifecycle record](mc-heal-resume-12789a78a60b/job-lifecycle.json).
All 52 continuation artifact hashes verified, including the selected/last
adapters and optimizer. Both pods are absent and this job's Dibs holds released.
Continuation code restores adapter, Adam state and RNG; its tiny-model test
verifies that the next update exactly matches uninterrupted training.

## Data and objective

The [official Horizon data navigator](https://github.com/ifm-ai/DATA_NAVIGATOR)
points to the public IFM datasets. This pilot uses pinned samples from
[Math-Reasoning](https://huggingface.co/datasets/IFM/Math-Reasoning) and
[Code-Reasoning](https://huggingface.co/datasets/IFM/Code-Reasoning).
The inspected general SFT subset is excluded because its factual labels were
not independently verified. All original reasoning traces are excluded.

| Training data | Examples | Assistant tokens, including EOS |
|---|---:|---:|
| Independently recomputed math | 1,525 | 10,541 |
| Code I/O agreeing with executed references | 4,439 | 94,746 |
| Total | 5,964 | 105,287 |

There are 2,106,214 input tokens including prompts. Every response is complete
within 2,048 tokens; none is clipped to fit. Source revisions, source row-group
coordinates, example hashes, token counts and splits are in
[preparation.json](preparation.json). The exact pilot is [data.json](data.json),
SHA-256 `4983f874e1f47d35071c7bc683a3f86285860b7c8e8fdcfe60eb54c109c4b3c2`.

Code runs in a local namespace sandbox with no network, home, project files or
inherited environment, with CPU, memory and output limits. Environment-dependent
and set-order-dependent programs are excluded. The final audit re-executed all
4,439 training and 24 held-out code examples and recomputed all 1,525 math
targets: zero failures. See [verification-audit.json](verification-audit.json).
Reference-program agreement does not prove every natural-language specification
is correct. This deliberately narrow dataset is a first repair diagnostic, not
a general instruction tuning corpus.

Training uses assistant-token cross-entropy, including EOS, with zero teacher
KL. LoRA rank 16 / alpha 32 trains 106,442,752 FP32 adapter parameters on body
linear layers; embeddings, vocabulary head, norms, base weights and projection
remain frozen. Base weights stay BF16, without weight quantization. Updates use
AdamW at 3e-5, four-update warmup, betas (0.9, 0.95), no weight decay, gradient
norm clipping at 1, and non-reentrant activation checkpointing. Each update
accumulates whole examples until 4,096 answer tokens or 65,536 input tokens
have been reached; the final example can exceed that boundary. Loss is divided
by the actual number of assistant tokens in the update.

## Controls, selection and stopping

The four arms are stock, full-width projected, 75% before healing, and the
selected 75% checkpoint after healing. Full-width retains all 5,120 residual
coordinates; its extra normalization matrix means it is not exactly 100% of
the stock parameter count.

All arms use identical native chat formatting, thinking disabled, greedy
decoding, no repetition penalty, and the same generation limits. Before repair,
the full-width BF16 numerical control must have mean KL <= 0.02 and top-1
agreement >= 95% on the previous 512 validation positions. KL is used only to
detect a broken full-width reconstruction. Each control also checks cached vs
uncached inference. A zero-initialized LoRA adapter must leave logits unchanged.

| Evaluation | Scored tasks | Additional long-form prompts | Purpose |
|---|---:|---:|---|
| Development | 64 | 2 | Checkpoint selection and early stopping |
| Final test | 128 | 6 | Before/after correctness comparison |
| Previous scale screen | 64 | 8 | Continuity and full-width generation control |

Fresh tasks cover arithmetic, sorting, Python execution, unit conversion, JSON
instructions, grounded extraction, conversation memory and withheld Horizon
code functions. Train/development/test code function groups are disjoint;
fresh development and test prompts have no exact overlap. This does not rule
out pretraining contamination or shared templates. The small synthetic screen
cannot establish broad capability parity.

Correctness and formatting are scored separately. Correct numbers inside a
clear final answer, JSON in a code fence, or a parcel code with explanatory
text can receive correctness credit while failing the strict format check.
Unparseable answers are separately flagged for review. The old 72-prompt
screen retains its original checks, which mix content and formatting; it is
not the primary selection score. Word n-gram and repeated-line loop flags,
EOS completion and full raw generations are saved. Long-form content needs
manual review; automatic loop checks do not establish its quality.

The run attempts at most 32 updates, measuring development results after
updates 4, 8, 16 and 32. It selects the highest correctness score among
checkpoints with no more loops than the unhealed model and at most one fewer
ended response. Ties favor fewer loops, more completions, then the earlier
checkpoint. The zero-update adapter is an eligible fallback.

Training stops after a measured loss of at least 4/64 correct answers, at least
two additional loop flags, or when final-evaluation time must be reserved.
It may therefore complete fewer than 32 updates. Final test outcomes never
select a checkpoint. Paired reporting identifies gained and lost answers, so
an unchanged aggregate cannot hide regressions. A development improvement alone
is not a successful general repair.

## Saved artifacts and validation

The pod saved intermediate, selected-best and last adapters, the last
optimizer and RNG state, source revision, exact projection recipe, executed
code, frozen dataset, environment, metrics and every generated answer. The exact
adapter reload check passed. The launcher synchronized artifacts during the job
and verified their hashes before deletion.
Termination is never delayed to finish copying.

The paid job does not transfer a full 40-GB merged model. The saved adapter plus
pinned source and projection recipe can reconstruct it. The `export` subcommand
of `correctness_heal.py` verifies source/recipe identity, merges adapters, and
writes a self-contained Hugging Face directory. BF16 merging can introduce
rounding differences, so the resulting large export must be evaluated before
being treated as equivalent to the unmerged adapter model.

Local validation covers label shifts and EOS, frozen dataset hashes and group
separation, target verification, correctness-vs-format scoring, frozen-base
training, chunked loss/gradient agreement, exact adapter reload, tiny-model
merged HF reload, paired regressions, dry-run isolation and budget/deletion
guards. A BF16 CPU smoke test of both Qwen attention types reduced answer CE
from about 5.58 to 5.39 over three updates. This validates the training path,
not model quality. The GPU runtime backward smoke and the full CUDA training
path both completed successfully in this first segment.

## Original approved paid job

| Item | Proposal |
|---|---|
| Resource | One Secure Cloud on-demand NVIDIA H200 SXM, 141 GB VRAM |
| GPU rate ceiling | $5.29/hour, confirmed by read-only API quote |
| Storage | 200 GB temporary container disk, about $0.0278/hour; no persistent volume |
| Expected time | 60–90 minutes, including setup, download and evaluation |
| Hard lifetime | 90 minutes from create intent |
| Planned usage at 90 minutes | About $7.98 |
| Requested budget cap | $10 total, including a cleanup/billing margin |

See [job-proposal.json](job-proposal.json) and [price-quote.json](price-quote.json).
The rate is checked again immediately before allocation; a higher quoted rate
aborts. The local watchdog is started before the sole create request and
survives the main launcher. It deletes only this job's pod on completion,
failure or deadline, and retries until deletion is confirmed. Timing/billing
and API outages can affect realized charges; the planned maximum leaves about
$2 below the requested cap. There is no automatic continuation or second pod.

This machine has no GPU. Local preparation and the small training smoke test
are practical here; the 27B controlled experiment needs the proposed GPU to
finish promptly.

## Reproduction and launch

From the repository root, using the existing local audit environment:

```bash
target/manifold-audit/env/bin/python -m pytest -q manifold-compression/test_correctness_healing.py manifold-compression/test_correctness_pod.py manifold-compression/test_scale_pod.py
target/manifold-audit/env/bin/python manifold-compression/run_correctness_pod.py --quote
```

The second command creates the bundle and reads the current price; it does not
allocate a resource. After explicit approval of this exact job:

```bash
dibs run gpu network --label "Approved Qwen27 correctness healing" --eta 90m -- target/manifold-audit/env/bin/python manifold-compression/run_correctness_pod.py --execute --approved-budget-usd 10
```

Credentials remain local at the user-specified private path. They are absent
from the bundle, command-line arguments, remote environment and artifacts.
The Dibs instruction was supplied after this job started; its existing launcher
was registered with a process-bound `gpu network` hold for the remaining run.

For raw-data reproduction, extract `prepared_source_manifest.raw_manifest`
from `preparation.json` and pass it to `horizon_fetch.py --replay-manifest`.
The recorded source files and row-group coordinates are authoritative; seed-only
resampling is not a promise to reproduce this already captured sample. Process
the raw sample with `horizon_data.py`, create a new pilot directory with
`correctness_prepare.py --out NEW_PATH`, and run `correctness_verify.py` on it.
Do not overwrite the frozen data used by a launched experiment.
