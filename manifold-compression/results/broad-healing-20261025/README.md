# Broad correctness-healing pilot: Qwen3.5-2B

Status: complete. All compute local; **$0 cloud spend**. Update 16 selected using development data only.

Starts from the 64k broad-calibrated shared projection, width 1520, with 1,399,093,040 frozen parameters. Rank-16 body LoRA adds 15,247,872 trainable parameters (1,414,340,912 total before merging). Projection, norms, embedding, output head, and original body weights stay frozen. The initial zero adapter reproduces the projected model exactly.

The objective is cross-entropy on checked assistant answers, including EOS. There is no stock-model KL or teacher-logit matching. Training uses CPU BF16 autocast with FP32 stored parameters; generation uses FP32, greedy decoding, no thinking or repetition suppression, and the previous four-example batch convention.

## Results

| Measure | Unhealed broad projection | Selected healing checkpoint | Stock reference |
|---|---:|---:|---:|
| Original closed-task content correctness | 24/56 | **43/56** | 51/56 |
| Fresh closed-task correctness | 18/42 | **32/42** | Not run |
| Development closed-task correctness | 8/21 | **14/21** | Not run |
| Original programs meeting all requirements | 0/4 | 0/4 | Not compared here |
| Fresh programs passing supplied tests | 0/4 | 0/4 | Not run |
| Original science explanations factually correct | 0/2 | 0/2 | Not compared here |
| Fresh science explanations factually correct | 0/2 | 0/2 | Not run |

There are **19 gains and no losses** on the original closed tasks, and **14 gains and no losses** on the fresh closed tasks. Stock's 51/56 uses the same content convention for sorting: unambiguous integer strings receive ordering credit, with type compliance separate. No stock fresh-test run was performed.

Development improves 8 → 12 → 14/21 at steps 0 → 8 → 16. Word-loop flags decline 2 → 1 → 0; all 25 development answers end at every checkpoint. The frozen rule selects step 16. On the fresh test, all 48 answers end, compared with 47 before; combined word/line loop flags fall 3 → 0. On the original benchmark, 61/62 answers end both before and after, and both science answers still trigger repetition flags. The capped answer moves from the spoon explanation to word-count code. A cap is not automatically a loop.

| Original task family | Before | After | Stock content score |
|---|---:|---:|---:|
| Arithmetic | 0/8 | 1/8 | 6/8 |
| Sorting | 5/8 | 6/8 | 8/8 |
| Python output prediction | 0/8 | 5/8 | 5/8 |
| Unit conversion | 3/8 | 7/8 | 8/8 |
| JSON construction | 0/8 | 8/8 | 8/8 |
| Grounded extraction | 8/8 | 8/8 | 8/8 |
| Conversation memory | 8/8 | 8/8 | 8/8 |

Fresh arithmetic remains **0/6**, so the one original arithmetic success does not establish recovered arithmetic capability. Fresh Python output prediction is 2/6. Other fresh families reach 6/6 each.

**Programming:** Original I/O scoring reports 1/4 before and after, but the passing search routine uses linear membership/index lookup rather than binary search. It therefore fails the full requirement. The repaired development chunking routine has the right algorithm but names its input `a` instead of `values`; it fails the keyword interface. A separately labelled, post-hoc positional-call audit credits that one partial recovery and does not change selection. No fresh program passes even that diagnostic. All execution stays in the networkless sandbox.

**Science:** The repaired model still claims a vacuum is solid, ice floats because it is denser than water, and gravity is absent inside the spacecraft. The shadow answer partially recovers the role of solar angle but adds a false explanation. Photosynthesis/respiration remain confused, and the spoon answer attributes sensation to rigidity/softness while explicitly denying heat transfer. Both original explanations still repeat. Cleaner endings and fluent prose are therefore not evidence of recovered scientific reasoning. See `selected-manual-review.json` for individual judgments.

**Interpretation:** Correctness healing of this fixed broad projection is useful: the closed-task gains are much larger than the one-answer gain in the earlier calibration-only expansion. It has not reached stock parity or restored reliable programming/science. The next priority is a richer, independently checked healing curriculum covering arithmetic, algorithms, and scientific explanations, with new held-out problem families. Repeating the eight science targets for longer would not test broader knowledge coverage. This experiment does not compare against an equally trained narrow-calibration projection or establish larger-model transfer.

## Verification and retained artifacts

- Three tests pass: frozen data/mask/split invariants, checkpoint-selection safeguards, and the distinction between argument renaming and algorithm repair.
- A tiny hybrid-Qwen backward test passes with the current projection: zero adapter is exact, loss decreases, and frozen weights remain unchanged.
- Real-model zero-adapter equality, saved-adapter reload equality, and cached/uncached numerical checks pass. Cache top-1 agreement is 1.0; mean KL is approximately 1.86e-11 on the checked positions.
- An independent process verifies pinned source hashes, reconstructs the projection, and loads the selected adapter. **All six generated answers match exactly**, including both complete legacy explanations, with original batch composition preserved. This is a six-case reconstruction check, not an exhaustive reproducibility guarantee.
- The completed schedule uses **66,308 assistant tokens across 1,361 presentations of 983 unique examples**. The selected adapter has 15,247,872 parameters. The final training loss is 0.863; checkpoint selection uses generated correctness, not this loss.
- `selected/adapter.safetensors` is the selected local adapter; `step-8` and `step-16` also retain optimizer states. `reproducibility.json` pins source files and code; `artifact-manifest.json` records artifact hashes. There is no new standalone Hugging Face checkpoint or upload.

## Frozen experiment

- 16 updates, at least 4096 assistant tokens each; evaluate steps 0, 8, and 16.
- AdamW, learning rate 0.00003, two-step warmup, betas 0.9/0.95, no weight decay, gradient clipping at 1.
- Complete sequences of at most 1024 tokens. Long targets are rejected rather than truncated.
- 192 distinct Horizon reference programs; each re-executed in the existing networkless bwrap sandbox against its supplied I/O. This verifies that example, not the complete natural-language specification.
- 320 synthetic worked multiplication targets, 560 structured/short-answer targets, eight original source-checked science explanations in the pool. The frozen schedule uses 983 unique rows; it resamples examples, especially the small science bank.
- Target-token mixture approximately 40% programs, 25% worked math, 20% science, 15% short tasks. Full counts and scheduled token totals: `protocol.json`.

Development contains 21 closed tasks, two programs and two explanations. Fresh test contains 42 closed tasks, four programs and two explanations. The previous 62-case benchmark remains separate. Development checkpoint selection uses closed-task plus program correctness, rejecting checkpoints with more loop flags or more than one lost completion. Ties favor fewer loops, more completions, then the earlier checkpoint. Explanations are reviewed separately and do not select checkpoints. The unhealed checkpoint can win.

Program evaluation uses execution tests. Passing I/O does not establish an explicitly requested algorithm: the original binary-search task also requires manual algorithm review. Reaching a generation cap alone is not a loop. Explanations require factual review rather than keyword scoring.

## Scope limits

This is a bounded repair pilot, not a broad intelligence benchmark. Fresh short-answer cases share task templates with training. Worked math is restricted to multiplication. Eight science topics are not broad science coverage. Exact prompts and held-out multiplication operand pairs are excluded from training; no claim is made about absence from stock pretraining or earlier projection calibration. The original photosynthesis/respiration and metal-spoon prompts are not trained. There is no matched narrow-calibration healing arm, so this trial cannot isolate the interaction between calibration breadth and healing.

Science source URLs are recorded per example in `data.json`; original texts live in `../../broad_healing_science.py`. Sources include NASA's [seasons explanation](https://spaceplace.nasa.gov/seasons/en/), [Moon phases](https://science.nasa.gov/moon/moon-phases/), and [weather/climate distinction](https://science.nasa.gov/climate-change/faq/whats-the-difference-between-weather-and-climate/), plus the linked USGS and NASA pages for the other topics.

## Reproduction and artifacts

From the repository root, with the existing CPU environment:

```bash
dibs run cpu --label "Local broad correctness healing" --eta 2h -- target/manifold-audit/env/bin/python -u manifold-compression/broad_healing.py run
```

`prepare` freezes the data and refuses to overwrite it. `run` refuses to overwrite an existing training log; an interrupted run requires explicit checkpoint recovery, not blind rerunning. `training.json` records every completed update. `step-8` and `step-16` retain adapter and optimizer states. `selection.json` records development selection; `selected` stores the chosen adapter. `test-0.json`, `test-selected.json`, and `legacy-selected.json` contain answers, grading, EOS/cap and loop information. `result.json` exists only when the complete run has finished.

Import `load_broad_healing.load()` with `manifold-compression` on `sys.path`. It reconstructs the exact projection from local source weights and loads the selected adapter. This is a local recipe-plus-adapter artifact, not a standalone Hugging Face checkpoint or an upload.
