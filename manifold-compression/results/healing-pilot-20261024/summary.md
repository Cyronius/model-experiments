# Healing pilot results

All four fixed-budget training runs completed locally. Every saved FP32 checkpoint reloaded with identical logits on the reload test. RunPod spending: **$0**.

## Held-out token evaluation

| Model | Initialization | KL before | KL after | Target NLL before → after | EOS NLL before → after |
|---|---|---:|---:|---:|---:|
| Qwen3.5-2B | corrected | 0.7122 | 0.8202 | 1.9904 → 1.9469 | 0.6909 → 0.6773 |
| Qwen3.5-2B | legacy | 0.8016 | 0.8361 | 2.0708 → 1.9581 | 0.9695 → 0.7296 |
| SmolLM2-1.7B | corrected | 0.5717 | 0.4195 | 1.7455 → 1.5977 | 0.8317 → 0.8652 |
| SmolLM2-1.7B | legacy | 0.9185 | 0.5097 | 2.0797 → 1.6798 | 0.2291 → 1.7983 |

Lower is better for KL and NLL. These are 12 held-out complete conversations, with 1,806 supervised assistant positions for Qwen and 1,847 for SmolLM2. Qwen teacher agreement worsened after this short recipe; SmolLM2 improved. These results do not isolate architecture from pretraining, tokenizer, or optimization differences.

## Qwen diagnostic checks

The increased held-out KL triggered checks for implementation artifacts. FP32 versus CPU BF16 autocast changed predictions by only about 0.00006–0.00064 mean KL on six sampled conversations, much less than the observed drift. Batched padded versus individual inference agreed: mean KL 3.03e-13 in FP32 and zero in the BF16 diagnostic. Thus those checks do not explain the regression.

A small diagnostic subset also showed substantial forgetting of early short training examples: one code/list-format example rose from 0.076 to 5.928 KL at sampled positions. This is consistent with an unstable or insufficiently representative tiny-batch repair recipe. It is not proof that Qwen cannot be healed. The pilot contains only about 8,000 supervised tokens per arm, one pass, and no learning-rate sweep.

The training data, loss, updates, and evaluation settings were held fixed. A grading correction was made after reviewing the stock baseline: “Yes” and “Teal” are accepted for prompts that did not require lowercase. The correction applies uniformly to every arm; exact code copying stays case-sensitive. Original scores are retained where changed, and the amendment is recorded in `protocol.json`.

## Full-response results

| Model | State | Checks passed | Completed | Repetition flags |
|---|---|---:|---:|---:|
| Qwen3.5 | Stock | 13/16 | 16/16 | 0/16 |
| Qwen3.5 | Legacy initialization | 10/16 | 15/16 | 1/16 |
| Qwen3.5 | Legacy + healing | 4/16 | 14/16 | 2/16 |
| Qwen3.5 | Corrected initialization | 10/16 | 16/16 | 1/16 |
| Qwen3.5 | Corrected + healing | 7/16 | 13/16 | 4/16 |
| Qwen3.5 | Recovered 500M-token student | 4/16 | 15/16 | 2/16 |
| SmolLM2 / Llama | Stock | 7/16 | 16/16 | 0/16 |
| SmolLM2 / Llama | Legacy initialization | 4/16 | 16/16 | 1/16 |
| SmolLM2 / Llama | Legacy + healing | 1/16 | 14/16 | 5/16 |
| SmolLM2 / Llama | Corrected initialization | 5/16 | 16/16 | 2/16 |
| SmolLM2 / Llama | Corrected + healing | 6/16 | 16/16 | 0/16 |

These are mechanical instruction/format/rubric checks, not a general accuracy score. Some failures are formatting only: the healed legacy SmolLM2 answers 29 × 17 correctly but adds prose despite the number-only instruction. Other failures are substantive: the healed Qwen model turns an exact code-copy request into repeating HTML-like text, and both healed SmolLM2 arms still give an incorrect explanation/output for Python alias assignment. A word-limit check can pass an inaccurate or repetitive answer. The raw answers and separate repetition metric are essential.

The corrected SmolLM2 arm removed the two repetition flags observed before healing and improved from 5 to 6 checks, versus 7 for stock. It restored exact copying, the simple logic answer, and color recall, but regressed on the summary and hash-table checks. This is a promising initialization/recovery result on a small suite, not parity.

Both Qwen healing arms regressed. Corrected Qwen fell from 10 to 7 checks, gained three additional repetition flags, and hit the cap on copying, list editing, and the poem. Its shorter dog-adoption answer passes the length check while still repeating itself. The legacy Qwen arm fell from 10 to 4 checks.

The legacy SmolLM2 arm is the clearest counterexample to using KL alone: KL improved from 0.9185 to 0.5097 while checks fell from 4 to 1 and repetition flags rose from 1 to 5. The corrected arm finished with better KL and much better generated behavior than that control.

The recovered 500M-token Qwen student has KL 0.4326 on this validation set, better than either fresh Qwen initialization, yet passes only 4 checks and repeats on sorting and interview advice. Its vocabulary IDs, ending token, generation prompts, and validation token sequences match the source tokenizer exactly. This is not a tokenizer-format mismatch. Its parameter count and training history differ, so it is a reference rather than a matched arm.

## Decision

**Keep the full output-matrix correction. Do not promote either Qwen healing checkpoint or scale up this exact repair recipe.** SmolLM2 supports keeping the correction, but neither architecture has demonstrated parity.

The next controlled repair should stabilize optimization: use a smaller learning rate and a larger effective batch, with a generation regression gate before extending the token budget. Teacher-written complete responses would also avoid the competing style of the small external-response corpus. These are proposed changes, not conclusions established by this pilot; separate ablations should determine their effects. The observed forgetting and tiny batch make this a more direct next diagnostic than guessing that a larger source model or more PCA calibration points will solve the problem.

Future validation needs more held-out tasks and sampled decoding in addition to this greedy stress test. One seed, 16 generation cases, and about 8,000 supervised tokens cannot establish broad parity, a population loop rate, or architectural uncompressibility.

## Saved artifacts and cost

All four final HF checkpoints are saved with their dense adapter and tokenizer. They are experimental; the Qwen runs failed the behavioral comparison:

- `models/manifold-compression/healing-pilot-qwen35-corrected/`
- `models/manifold-compression/healing-pilot-qwen35-legacy/`
- `models/manifold-compression/healing-pilot-smollm2-corrected/`
- `models/manifold-compression/healing-pilot-smollm2-legacy/`

Each arm completed exactly 32 updates and 64 supervised endings. Qwen used 7,870 assistant tokens per arm, and SmolLM2 used 8,072. Each checkpoint reload matched its pre-save logits exactly on the reload test. The 26-test suite passed; the scoring amendment also passed its six-test subset.

Each training run took roughly four to five minutes while the matched arms ran concurrently. CPU generation dominated the total elapsed time, especially when a response ran to 512 tokens. No RunPod compute was allocated and no new paid resources were created: **$0 spent**.

Completed generation results can be reused when the checkpoint file metadata, data hash, generation length, and batch size match; `--force-generation` overrides reuse. All raw outputs and any original casing-only scores are retained.

See [experiment protocol and reproduction](README.md), `protocol.json`, all raw generation JSON files, and the precision/padding diagnostics.
