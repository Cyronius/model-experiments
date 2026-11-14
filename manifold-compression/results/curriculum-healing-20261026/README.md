# Expanded correctness curriculum: Qwen3.5-2B continuation

Status: complete locally; **$0 cloud spend**. The expanded curriculum produces modest closed-answer gains, but code regresses and science remains unreliable. Preserve the parent and selected adapters as separate experimental checkpoints; this is not a general replacement or stock-parity result.

## Outcome

| Evaluation | Parent (16 total updates) | Selected (24 total updates) | Stock |
|---|---:|---:|---:|
| Original closed tasks | 43/56 | **45/56** | 51/56 |
| New closed tasks | 12/38 | **16/38** | 19/38 |
| New arithmetic word problems | 3/24 | **4/24** | 8/24 |
| New retention tasks | 9/14 | **12/14** | 11/14 |
| New programs passing all supplied tests | 1/4 | **0/4** | 4/4 |
| Original repetition flags | 2/62 | **0/62** | 1/62 |
| New repetition flags | 1/44 | **1/44** | 0/44 |

The original closed set gains two answers and loses none. The new closed set gains six and loses two; its dot-product program regresses from correct scalar summation to a list of products. All four selected original programs also fail. The parent's one original I/O pass did not satisfy the required binary-search algorithm, so neither compressed checkpoint meets a full original program specification.

Both original science explanations remain false. Photosynthesis now finishes but describes plants digesting their DNA to make cells; the spoon answer confuses temperature with heat transfer. No automatic repetition flags does not imply factual correctness or absence of semantic repetition. Both new science explanations also fail. Stock explains the lightning/thunder mechanism correctly and identifies mammal traits for whales, though its fish-temperature generalization has exceptions (see stock manual review).

Development closed correctness is 8/23 → 10/23 → 10/23 at total updates 16, 24, 32; program correctness stays 1/2. The frozen tie rule selects **24 total updates**. The selected adapter received **32,975 additional answer tokens**; the complete 16-update continuation consumed 65,988. Final-32 is retained but was not selected or evaluated on the full final test.

## Diagnostics and interpretation

Six within-taught-topic science applications were fixed before the first logged continuation update. Parent and selected each give two sound core answers (DNA sequence/backbone and frictional energy conversion), plus an unexplained correct “No” on molecular motion. Both fail charge/current; the selected catalyst answer has a correct equilibrium conclusion with a false reversal explanation. Final-32 loses both sound core answers and adds false explanations. These are manual core-concept judgments, not a claim that every wording/detail of the two accepted answers is complete.

For eight identical held-out arithmetic operand sets, plain calculation accuracy is **1/8 parent, 1/8 selected, 3/8 final-32**. The matched word problems score **0/8 parent and 2/8 selected**. Removing story wording does not restore arithmetic in the selected model. This small exploratory diagnostic is neither an independent test nor a basis for replacing the selected checkpoint.

The evidence supports limited task repair, not broad intelligence recovery. Completing more updates of this recipe does not uniformly improve abilities. It does not establish whether the limiting factor is rank-16 adaptation, optimization, the global projection, or model scale. Without a matched old-curriculum continuation, breadth itself is not causally isolated.

The next useful experiment is a controlled small-model comparison with the same verified data and fresh evaluation: current projection with greater healing capacity, versus a less destructive projection at matched size and healing budget. Include an uncompressed training control to check whether the curriculum/objective can teach the target skills at all. Compare correctness and lost abilities, not only training loss or stock-logit similarity. These are proposed experiments, not additional jobs launched here. More expensive scale tests should wait for a clear small-model signal.

## Verification

Five tests pass. Adapter reload and cached/full-forward numerical gates pass. A fresh reconstruction from the pinned source and projection reproduces six selected answers, termination states and token counts exactly, including both original explanations. The uniform numeric-content audit leaves checkpoint selection unchanged; raw grades remain saved. All source/training-code hashes remain unchanged; supporting audit/report scripts are recorded separately. No standalone HF export or upload was performed.

The projection and original weights remain frozen. This stage continues the same rank-16 body LoRA parameters and restores the parent AdamW optimizer state. It adds 16 updates of at least 4096 answer tokens each, with no teacher-logit or KL objective. Learning rate stays 0.00003. Training uses CPU BF16 autocast with FP32 stored parameters; evaluation uses FP32, four threads, greedy decoding, no thinking or repetition suppression, and the prior fixed batch convention.

## Frozen data and selection

The pool contains:

- 800 exact-arithmetic examples across addition/subtraction, multiplication, division, fractions, percentages, linear equations, remainders, and multi-step calculations. Half request a brief calculation; half request only the answer. A separately parsed rational-expression oracle checks every target.
- 12 authored algorithm families checked against multiple independent input/output oracles in the existing networkless sandbox. The 94 training rows provide different examples for these same 12 algorithms, not 94 independent algorithms.
- 256 additional Horizon reference programs, disjoint by source problem group from the first healing pool. Their supplied I/O is re-executed in the sandbox. This is weaker validation than the independent multi-input checks above and is not proof of the full natural-language specification. Prompts include exact function signatures.
- 24 additional original science explanations checked against 12 OpenStax chapters, plus the previous eight science targets. The additional explanations cover distinct aspects of atoms, transport, nucleic acids, enzymes, cell division, evolution, bonding, energy, momentum, gases, ecosystems, and current/resistance. Related explanations from a chapter are not independent knowledge domains.
- 560 replay examples from the earlier structured-task pool to help retain prior gains.

The scheduled answer-token mix is 35% arithmetic, 15% independently tested code, 20% Horizon code, 20% science, and 10% replay. The schedule contains 65,988 answer tokens and 1,562 unique rows; repeated presentations are recorded explicitly in `protocol.json`. Not every pool row is scheduled: in particular, this run schedules 100 of the 256 new Horizon functions. The full pool is retained for auditing.

Development has 16 arithmetic word problems, seven retention tasks, two programs, and two science explanations. Fresh test has 24 arithmetic word problems, 14 retention tasks, four programs, and two science explanations. Their prompts are disjoint from both stages' training prompts. Explicit multiplication operand pairs are checked across training and evaluation. The old unit-conversion generator had only 99 possible questions, so this stage's new retention questions use a larger numeric range.

Select among stage steps 0, 8, and 16 using development closed-answer plus program correctness. Reject a candidate with more combined word/line loop flags or more than one lost completion relative to the parent. Ties favor fewer loops, more completions, then the earlier checkpoint. Science receives a separate manual factual review and does not select the checkpoint. The parent remains eligible. The original 62-case benchmark is evaluated after selection and does not choose the checkpoint.

## Limits

Before any trained-checkpoint generation was inspected, the stock baseline exposed a numeric-grading bug: the previous parser could select an intermediate equals sign and miss an explicit correct numeric final line. `numeric-audit-amendment.json` records a uniform correction, implemented and tested in `curriculum_numeric_audit.py`. Standalone final numeric values, including simple LaTeX fractions, mixed numbers, and valid thousands separators, take precedence; otherwise the previous explicit-answer parser is used. Extraction never consults the expected value. Original grades remain available. Final checkpoint selection uses the corrected development content scores with the same completion/repetition safeguards; if this changes the provisional runner's choice, the audited winner must be reloaded and evaluated. No training targets or training code change.

The uncompressed source is also evaluated on the frozen new test, solely as a reference. Its outputs do not enter training or checkpoint selection. Six supplementary within-topic science applications were fixed before the first logged continuation update. They test application of taught concepts rather than unseen-topic generalization and are not used for selection.

After the halfway development comparison, an exploratory arithmetic-form diagnostic was also fixed: the first held-out test item from each of the eight arithmetic families is asked as a plain calculation with the same operands. This compares symbolic calculation with the original word-problem form. It is not a new independent test set and does not choose checkpoints. Its timing and scope are recorded in `math-form-cases.json`.

This is an additional-training experiment with a changed curriculum. Without a matched continuation on the old curriculum, any gain cannot be assigned solely to greater breadth. Arithmetic evaluation changes wording and operands but shares task families with training. Science remains a small, resampled set. The original benchmark has been inspected repeatedly, although its questions and direct target answers remain out of this training pool. These results cannot establish broad intelligence parity or larger-model transfer.

Generation caps remain 256 tokens for closed answers, 512 for code, and 1024 for explanations. Caps, repetition, syntax, required interfaces, algorithm requirements, and factual correctness are separate checks. Passing I/O examples is not proof of algorithm correctness; the original binary-search requirement still needs manual review.

## Sources and artifacts

The added science text is original summary prose checked against OpenStax, with per-example URLs in `data.json` and `../../curriculum_science.py`. Sources include [passive transport](https://openstax.org/books/biology-2e/pages/5-2-passive-transport), [enzymes](https://openstax.org/books/biology-2e/pages/6-5-enzymes), [covalent bonding](https://openstax.org/books/chemistry-2e/pages/7-2-covalent-bonding), and [Ohm's law](https://openstax.org/books/physics/pages/19-1-ohms-law). The complete source list is retained alongside each target.

Scripts: `../../curriculum_healing.py`, `../../curriculum_tasks.py`, `../../curriculum_science.py`, and `../../test_curriculum_healing.py`. The frozen protocol, source/adapter/optimizer hashes, code hashes, and test output are saved here. `result.json` is written only when the full training and evaluation run completes. Adapters and optimizer states are saved at stages 8 and 16; `selected` contains the development-selected adapter.

From the repository root, the run command is:

```bash
dibs run cpu --label "Expanded correctness curriculum" --eta 2h -- target/manifold-audit/env/bin/python -u manifold-compression/curriculum_healing.py run
```

Preparation refuses to overwrite frozen data. Training refuses to overwrite an existing training log; interrupted jobs require explicit checkpoint recovery. These are local recipe-plus-adapter artifacts, not a standalone Hugging Face checkpoint or upload.
