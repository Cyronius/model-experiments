# Plan: a tiny iterative editor that rewrites English into ASD-STE100

*Draft, October 2026. Guideline targets, not gates: under 100 MB on disk, under 500 ms per sentence on a laptop CPU, the XDNA2 NPU on Strix Point as the preferred deployment target.*

This is the design record for the Simplified Technical English (STE) rewriter. Read it before touching anything under `ste/`. The other plan in this repo, the heal pipeline, is referenced the same way from `heal/README.md`.

---

## 0. What we are building

**Input:** English technical text, one sentence or one paragraph at a time, plus two side inputs: a mode flag (`PROC` for procedures, `DESC` for descriptive text) and a customer glossary.

**Output:** an STE-compliant rewrite, a token-aligned diff against the input, and a list of flags the model could not resolve.

**Shape:** a small bidirectional transformer that iteratively edits a fixed-length canvas. The canvas starts as a copy of the source. Every pass runs under a hard lexicon mask, so the model cannot emit an unapproved word. A deterministic rule checker runs between passes. It feeds violations back into the next pass and decides when to stop.

**What it is not:** an autoregressive LLM, a general simplifier, or a replacement for a technical author. The output is a draft with a diff. A human signs off.

---

## 1. Decisions already made, and why

| decision | choice | why |
|---|---|---|
| where the vocabulary constraint lives | output side only | The encoder must read any English a writer types, including unapproved words it has to recognize. The decoder vocab is the STE lexicon plus slots, so the constraint is one logit mask. |
| generation scheme | non-autoregressive iterative refinement over a canvas | Most of the output is a copy of the input. Parallel decoding makes latency a function of passes, not length. Fixed shapes suit NPUs. This is Mask-Predict with a source-initialized canvas, which the masked-diffusion literature later formalized. |
| canvas init | source tokens with NULL padding, not all-mask | Starting from all masks throws away the 70 percent of tokens that are already right. |
| exit decider | deterministic checker, plus "nothing changed", plus a hard cap | Checking STE is mostly lookups. Fixing it is not. Rules verify, the model generates. |
| checker in the loop | per-token violation features fed into each pass | Turns blind refinement into guided repair. Cheap: the features are lookups. |
| domain terms | slotted at the edges, never in the weights | STE rules 1.5 and 1.12 already define Technical Names and Technical Verbs as customer-maintained lists. A new domain is a new glossary file. |
| unknown words | protect, copy, flag | In aviation, unchanged and flagged beats confidently wrong. |
| graph | static shapes, boring ops, loop on the host | Every NPU toolchain handles MatMul, LayerNorm, GELU, Softmax, Gather. Several choke on RoPE, SwiGLU, RMSNorm, or dynamic shapes. |
| NPU target | AMD XDNA2 on Strix Point, through ONNX Runtime and the Vitis AI execution provider | It is the machine on the desk. The official toolchain is Windows-only as of Ryzen AI 1.7; Linux has the kernel driver and IRON for custom kernels but no production execution provider. See 3.8. |
| quantization | W8A8 as the shipping target, BF16 as the fallback on XDNA2; ternary as a custom-kernel experiment | The Ryzen AI compiler accepts INT8 and BF16, nothing lower. Ternary packs to INT8 on the standard path and only pays off through a hand-written IRON kernel. At 20M to 40M parameters the model fits in cache on the CPU, so bandwidth is not the bottleneck there. |
| teacher | three teachers for three jobs: the local Qwen MoE for volume, Claude Sonnet 5.5 for the hard cases, Claude Opus 5.5 as judge | The local box is bandwidth-bound at about 120 GB/s, so only MoE teachers decode at a usable rate. The checker filters compliance, so the volume teacher can be modest. Meaning fidelity on the hard rewrites and the judge cannot. See 3.4. |
| build order | tagger baseline before the canvas model | A GECToR-style tagger is the simplest thing that can work. It shares the lexicon, checker, slots, and data. It sets the metric floor and is the fallback if small-scale diffusion disappoints. |
| pretrained weights | none | Our vocab is word-level and ours. A short MLM pretrain on technical English with our tokenizer is cheap if we want it. |

---

## 2. Spec facts the design depends on

Source: ASD-STE100 Issue 9, January 2025, from asd-ste100.org. The spec is free on request. The owner has confirmed its terms permit embedding the dictionary in this product. Rule numbers below follow Issue 9; confirm each against the PDF when encoding, since Issue 9 reworded 31 of the 53 rules.

- **Two parts.** Part 1 is 53 writing rules in nine sections. Part 2 is the dictionary: about 900 approved words, each with exactly one approved meaning and one approved part of speech, plus a larger list of unapproved words with approved alternatives.
- **Technical Names (rule 1.5) and Technical Verbs (rule 1.12).** Words outside the dictionary are allowed when they name parts, tools, materials, locations, processes, and similar categories the spec lists. Companies keep their own lists. This is our extension mechanism.
- **Sentence length.** Procedures: at most 20 words. Descriptive text: at most 25 words.
- **Paragraphs.** At most six sentences. One topic per paragraph.
- **Noun clusters.** At most three words.
- **Verbs.** Only the infinitive, imperative, simple present, simple past, future, and past participle as adjective. No -ing forms except inside Technical Names. No perfect tenses. Active voice, with narrow exceptions in descriptive text.
- **Procedures.** Imperative mood. One instruction per sentence. Warnings and cautions come before the step.
- **Articles and "that".** Use them. Do not drop them to save words.
- **Spec examples.** Every rule ships with non-STE and STE example pairs. They become the checker's unit tests and the first seed of the gold set.

**Third-party encodings: oracle, not source.** A PyPI package, `biz-dfch-asdste100vocab`, implements an Issue 9 dictionary in JSONL under AGPL-3.0, unaffiliated with ASD. Our right to the dictionary content comes from ASD's terms, not from this package, and whether its author had the right to publish it changes nothing about what we may take from it. What the author does own is the schema, the categorization, the notes, and any examples they wrote. Copying the file takes that layer with it, and AGPL attaches to it. An AGPL item in the bill of materials is a procurement problem for aerospace customers whether or not anyone would ever sue.

So: build our own encoding from the PDF, then use their file read-only as a test oracle. Diff word lists and statuses against ours, and resolve every disagreement by looking at the PDF, never by copying their row. Nothing from their file is committed. Also ask ASD whether a machine-readable dictionary ships with Issue 9. If it does, this paragraph is moot.

---

## 3. Components

Everything lives under `ste/`. Each component has its own README once it exists.

### 3.0 The development machine

A Strix Point laptop: Ryzen AI 9 HX 370-class, 12 Zen 5 cores, Radeon 890M iGPU, XDNA2 NPU rated at 50 TOPS INT8, and 96 GB of LPDDR5X on a 128-bit bus. For the teacher, the number that matters is the memory bus: about 120 GB/s theoretical, 80 to 100 GB/s in practice. Decode speed for any model on this box is roughly that bandwidth divided by the bytes touched per token. For the student, the number that matters is the NPU, covered in 3.8.

Three consequences:

- **MoE teachers only.** A dense 27B model at Q4 touches about 15 GB per token and decodes at 5 to 8 tok/s. A 27B-A2.8B MoE touches about 1.6 GB and decodes several times faster.
- **The 96 GB is what makes the big MoE possible.** Qwen3.8-Flash-Next-131B-A6B at 65 GB sits in RAM with room to spare. It is the quality ceiling available locally, at roughly half the decode rate of the 27B-A2.8B.
- **The NPU does not help the teacher.** Ryzen AI's LLM path targets small dense models on Windows and does not run a custom MoE GGUF. The iGPU speeds up prefill through Vulkan but shares the same memory bus, so it does not change decode speed.

### 3.1 Lexicon and tokenizer, `ste/lexicon/`

**Lexicon.** One JSONL row per dictionary entry.

```
{"lemma": "ensure", "pos": "verb", "status": "unapproved",
 "alternatives": ["make sure"], "inflections": ["ensure", "ensures", "ensured", "ensuring"],
 "examples": {"ste": [...], "non_ste": [...]}, "source": "Issue 9, dictionary"}
```

Approved entries carry `approved_meaning`. Unapproved entries carry `alternatives`. Approved words that are approved in one part of speech and unapproved in another get two rows.

**Build.** Text-extract the dictionary from the PDF, parse into rows, then review every row by hand against the page. Budget two to three weeks. This is the one step with no shortcut, and every other component depends on it being right.

**Tokenizer.** Word-level, lowercase, with a case feature.

| vocab region | contents | rough size |
|---|---|---|
| `V_out` | every inflected form of every approved word; function words; punctuation; digit tokens; slot tokens `TN_1..TN_32`, `TV_1..TV_16`, `NUM_1..NUM_16`, `PROTECT_1..PROTECT_16`; control tokens `NULL`, `PAD`, `SEP`, `PROC`, `DESC` | 3k to 4k |
| `V_in` | `V_out` plus every inflected form of every unapproved word, plus a 4k to 8k BPE fallback for everything else | 12k to 16k |

The output head only ever scores `V_out`. The lexicon mask is a `[|V_out|]` additive bias, built once per request from the base lexicon plus the glossary's company-approved words.

**Numbers never change.** Numbers, units, and part numbers go through `NUM` slots and are copied back verbatim.

**Case.** The detokenizer restores sentence-initial capitals and glossary casing. Slots carry their original surface form.

### 3.2 Deterministic checker, `ste/checker/`

**Interface.** `check(tokens, mode, glossary) -> [Violation(rule_id, span, severity, suggestion)]`.

**A useful property.** Every approved word has exactly one approved part of speech. So on the *output* side, where the vocab is closed, part of speech is a lookup. The checker at inference time needs no tagger. A tagger is only needed on noisy *input* text, which only matters offline for data generation and diagnostics.

**Rule tiers.**

| tier | what | how |
|---|---|---|
| A, pure lookup | unapproved word; approved word used in an unapproved part of speech; words per sentence; sentences per paragraph; noun cluster length; -ing form outside a Technical Name; perfect tense ("has/have/had" plus participle); passive ("be" plus participle); more than one imperative per sentence in `PROC` mode | lexicon lookups, counting, and short pattern rules over the closed output vocab |
| B, heuristic | missing article before a singular count noun; "that" dropped after reporting verbs; warning placed after its step | pattern rules with known false-positive rates; reported with lower severity |
| C, not checkable | meaning preserved; the right alternative chosen; the sentence actually reads well | handled by evaluation, never by the checker |

**Five jobs for one checker.** Evaluation metric. Exit decision. Feedback features into the model. Filter for synthetic targets. The inverse of the noiser.

**Tests.** The spec's own example pairs. Every non-STE example must raise at least the rule it illustrates. Every STE example must raise nothing in tier A.

### 3.3 Glossary and slotting, `ste/glossary/`

**Glossary format.** One JSON file per customer or domain.

```
{"technical_names": [{"surface": ["bleed air valve", "bleed-air valve"], "lemma": "bleed air valve", "category": "part"}],
 "technical_verbs": [{"lemma": "ream", "forms": {"base": "ream", "s": "reams", "ed": "reamed"}}],
 "forced_substitutions": [{"from": "a/c", "to": "aircraft"}],
 "company_approved": ["avionics"]}
```

**Slotter.** Runs before the model and after it.

1. Apply forced substitutions. Deterministic, before anything else.
2. Longest-match glossary terms. Replace each with `TN_i` or `TV_i`, record the surface form in a side table.
3. Heuristics for unglossed terms: part-number and alphanumeric patterns, units, acronyms, capitalized multiword runs. These become `TN` slots too.
4. Numbers become `NUM_i`.
5. Any remaining word that is in neither the lexicon nor the glossary becomes `PROTECT_i` and raises a flag.
6. After decoding, restore every slot from the side table. Technical verbs get inflected by a small rule table with exceptions. STE only allows base, -s, -ed, infinitive, and imperative, so the table is short.

**Training with slots.** Slot 70 percent of technical names in the training data and leave 30 percent raw, so the model tolerates detector misses.

**One glossary, two consumers.** The checker reads the same file, so it never flags a Technical Name as an unapproved word.

**Allowed versus preferred.** What a customer *allows* is a mask bit and a glossary entry, changed at runtime. What a customer *prefers* is learned behavior. Forced substitutions cover most preference asks deterministically. If a customer needs the model to behave differently, phase 6 adds a small LoRA adapter trained on their manuals over the frozen base. A few hundred kilobytes per customer.

### 3.4 Data pipeline, `ste/data/`

There is no public parallel corpus of plain English to STE. Real STE manuals are proprietary. We build the data.

**Seed clean STE.**
- The spec's example sentences.
- Teacher-generated STE. Prompt a teacher model with the rules and a topic, generate a paragraph, keep it only if the checker reports zero tier A violations. Topics come from public aviation and maintenance text: FAA handbooks, NASA technical reports, public maintenance training material. The teacher writes new sentences about those topics; it does not copy them.
- Which teacher does which job is in the teacher plan below.

**Teacher plan.** Three jobs, three teachers. The checker filters compliance on every slice, so what we pay for in a teacher is meaning fidelity and natural phrasing, not rule knowledge.

| job | teacher | why |
|---|---|---|
| volume: seed STE generation and LLM back-translation, 25M to 30M output tokens, easy tasks | local Qwen3.6-27B-A2.8B Q4_K_M with MTP, behind `llama-server` | Free, already tuned for this box, and good enough for paraphrase when the checker is downstream. Keep the teacher prompt short: a 300-token rule summary plus the one rule being targeted, not all 53 rules, so prefill stays cheap. |
| hard cases: passive-to-active, sentence splitting, 100k pairs with retries | Claude Sonnet 5.5 through the Batch API, with the rules prefix cached | A weak rewriter passes the checker and quietly changes meaning. This slice needs a strong one. Small enough to buy. |
| judge and gold set: meaning preservation on the gold set, release gates, drafting the 1,000 gold rewrites for human verification | Claude Opus 5.5 at release gates; Sonnet 5.5 for routine iteration rounds | The judge must be stronger than the teacher it grades and from a different lineage than the volume teacher. Human verification sits on top for the gold set. |

If the Sonnet slice grades weaker than expected on the gold set, promote it to Opus 5.5. It costs tens of dollars more.

**Local throughput, measured on day one.** Expect roughly 20 to 35 tok/s single stream for the 27B-A2.8B and 60 to 100 tok/s aggregate with 8 to 16 parallel slots. Run `llama-server` with parallel slots, continuous batching, Vulkan offload for prefill, and prompt-cache reuse so the shared rule prefix is not re-prefilled per request. Size the context as slots times the per-slot budget, about 1k tokens with the short prompt.

**Time and money, rough.** Prices are first-party API rates as of October 2026.

| slice | output tokens | local box at 80 tok/s aggregate | API alternative |
|---|---|---|---|
| seed STE, 500k sentences with rejection sampling | about 10M | about 1.5 days | Claude Haiku 5.5 Batch, about $30 |
| LLM back-translation, 400k sentences | about 16M | about 2.5 days | Claude Haiku 5.5 Batch, about $50 |
| hard-case distillation, 150k requests with retries | about 9M | not recommended locally | Claude Sonnet 5.5 Batch with cached prefix, about $100; about $200 without Batch |
| judge, one routine round of 3k items | about 0.6M | no | Sonnet 5.5, $10 to $15; Opus 5.5 at a release gate, about $30 |

A rented GPU is the other way to do the volume slices: a few hours of one H100-class card running vLLM with the 35B-A3B base at high batch clears 30M tokens for $10 to $20. Start local because the pipeline has to work locally anyway, then move the bulk run to Haiku Batch or a rented card when the pipeline is proven and a week of laptop time is the bottleneck.

**Noise the clean STE into messy English.** Two noisers, mixed about half and half.

| noiser | operations | cost |
|---|---|---|
| rule-based, `noise_rules.py` | reverse the dictionary: swap an approved word for one of its unapproved synonyms; merge adjacent sentences with "and", "which", "while"; nominalize verbs into -ing forms; drop articles; insert "prior to", "in order to", "ensure that", "it is necessary to"; convert to perfect tense; convert imperative to "should be" passive | free, millions of pairs |
| LLM back-translation, `noise_llm.py` | teacher rewrites clean STE as "how an engineer would normally write this", in several styles: verbose, passive, bureaucratic, terse-with-jargon | about 40 tokens per sentence |

The reversed dictionary is the key trick. The spec's unapproved-word list, read backwards, is a labeled noise model for exactly the errors this task fixes.

**Hard-case distillation, `distill_hard.py`.** For the rules a noiser cannot invert cleanly, mainly passive-to-active and sentence splitting, run the teacher on messy source with the rules in its prompt and the checker in a reject-and-resample loop, up to four tries. Keep only checker-clean outputs. These are the gold for the structural rewrites.

**Volume and cost, rough.**

| slice | size | cost driver |
|---|---|---|
| rule-noised sentence pairs | 2M to 4M | CPU only |
| LLM-noised sentence pairs | 300k to 500k | about 20M teacher tokens |
| hard-case distilled pairs | 100k | about 15M teacher tokens with retries |
| paragraph pairs, 2 to 6 sentences | 200k | assembled from the above |

Reuse `heal/gen_teacher.py`'s shard writer and `bench/run_bench.py`'s `llama-server` client. Neither needs much change. API calls go through the official Python SDK with the Batch API and prompt caching on the rules prefix.

**Gold set.** 1,000 sentences and 200 paragraphs. Source text is real messy technical English; the STE rewrite is hand-authored or hand-verified. Tagged by rule category so per-rule accuracy is visible. Never generated by the training teacher. This set is the only thing that tells us the truth, so it gets built in phase 2 and grows every phase.

### 3.5 Model A: iterative tagger baseline, `ste/model_tagger/`

GECToR-style. An encoder with a per-token tag head.

| | |
|---|---|
| encoder | 6 layers, width 384, 6 heads, FFN 1536, learned absolute positions, LayerNorm, GELU |
| parameters | about 15M including embeddings |
| tags | `KEEP`, `DELETE`, `REPLACE_w` and `APPEND_w` for every `w` in `V_out`, plus about 20 transforms: verb form changes, case changes, merge-with-next |
| inference | apply tags, re-run, stop when every tag is `KEEP` or the checker is clean or the cap of 5 passes is hit |
| conditioning | mode token prepended; checker violation features added to token embeddings |

Why first: it is the simplest thing that can work, it exercises every shared component, and it sets the metric floor. If it hits the targets, we may be done.

### 3.6 Model B: canvas editor, `ste/model_canvas/`

A single bidirectional transformer over `[mode] [source tokens] [SEP] [canvas]`. Same encoder size as Model A to start.

**Buckets.** Total length 64, 128, or 256. One compiled graph per bucket. A 20-word sentence with its canvas fits in 64. A six-sentence paragraph fits in 256.

**Canvas layout.** Length is twice the source length, rounded up into the bucket. Source tokens sit on even slots, `NULL` on odd slots. Insertions fill the nearest free `NULL` to the right. Deletions become `NULL`. Target placement is precomputed by token-level Levenshtein alignment. A sample whose insertions do not fit moves up a bucket; one that does not fit in 256 is dropped.

**Training objective.** Masked-diffusion style, absorbing state. For each sample, draw a corruption rate `t` uniform in `[0, 1]`. Corrupt the target canvas by replacing a `t` fraction of positions with either the `MASK` token or the source-aligned token, half and half. Predict the full target canvas with cross-entropy on every position. Checker features for the corrupted canvas are computed on the fly; they are lookups, so this is cheap.

**Inference loop, on the host.**

```
canvas = source_interleaved_with_NULL
frozen = {}
for k in range(MAX_PASSES):            # MAX_PASSES = 6
    logits = graph(mode, source, canvas, lexicon_bias, checker_features(canvas))
    pred, conf = argmax_and_prob(logits)
    canvas[~frozen] = pred[~frozen]
    frozen |= conf > tau[k]             # tau schedule descends: 0.95, 0.9, 0.85, ...
    violations = check(collapse(canvas), mode, glossary)
    if not violations and (canvas unchanged this pass or all frozen):
        break
    unfreeze positions inside violation spans
return collapse(canvas), diff, flags
```

`collapse` drops `NULL` tokens and restores slots. If the cap is hit with violations remaining, return the pass with the fewest violations and attach them as flags.

**Why this beats the tagger, if it does.** Splits, reorders, and multi-token replacements are one canvas rewrite instead of a chain of dependent tags.

### 3.7 Exit policy

Exit when all three hold: tier A violations are zero; the canvas did not change in the last pass, or every position is frozen; the pass count is under the cap. Otherwise run another pass. A learned "another pass will help" head is a phase 6 option, only if the measured pass count is too high.

### 3.8 Runtime, `ste/runtime/`

**Export.** PyTorch to ONNX, one static graph per bucket.

| graph inputs | shape | notes |
|---|---|---|
| token ids | `[L]` | source plus canvas, padded |
| attention bias | `[L, L]` | additive; pads masked |
| checker feature ids | `[L]` | zero where clean |
| lexicon bias | `[|V_out|]` | additive; minus infinity for disallowed words |
| graph output | `[L_canvas, |V_out|]` | logits |

Ops allowed in the graph: MatMul, Add, Mul, LayerNorm, Gelu, Softmax, Gather, Transpose, Reshape. Nothing else. Position embeddings are baked constants since lengths are fixed. Everything that is dynamic, freezing, re-masking, slot restore, and the checker, runs on the host.

**Backends, in order.**

1. **ONNX Runtime CPU, INT8, on the Strix Point box.** The reference latency numbers come from here.
2. **XDNA2 through Ryzen AI Software**, 1.7.x as of this writing: ONNX Runtime with the Vitis AI execution provider. The compiler accepts INT8 and BF16. AMD documents BF16 as the transformer flow and INT8 for CNNs, and INT8 transformers work when the graph stays inside the supported op subset. Try INT8 first through Quark static quantization and fall back to BF16 if the INT8 compile rejects ops. Export at ONNX opset 17. Unsupported subgraphs fall back to the CPU silently, so read the partition report and demand zero fallback nodes. Pre-compile for the deployment build of the provider. Windows only as of Ryzen AI 1.7.
3. **Linux on the same machine.** The amdxdna kernel driver, mainline since 6.14, plus XRT gives device access, and IRON gives custom kernels. There is no production ONNX execution provider for Ryzen NPUs on Linux as of this writing. Verify against AMD's current release notes before planning around it.
4. **Later, same graph:** Intel through OpenVINO, Qualcomm QNN, Core ML.

**XDNA2 budget.** A 30M-parameter INT8 model streams 30 MB of weights per pass and needs about 10 GFLOP, which is 0.2 ms at the rated 50 TOPS. Launch and DMA overhead will dominate, so expect a few milliseconds per pass, and a whole sentence under 50 ms. Measure; do not trust this paragraph.

**De-risk early.** In the first week of phase 3, export a random-weight model of the final shape through the Vitis AI provider on the Strix Point box. Fix op-support and partition problems before training a real model into them.

**Custom kernels.** Two separate questions. On the CPU: only if ONNX Runtime INT8 misses the budget, which it almost certainly will not. On XDNA2: IRON is the path. Python describes the dataflow across the AIE2P tile array, C++ with vector intrinsics describes each tile's kernel, and Peano compiles it. Phoenix-era kernels do not run on Strix; build for the NPU2 targets from the mlir-aie Strix examples. This is also where a ternary experiment could actually pay. XDNA2 is a dataflow array fed by DMA from shared DDR, so a hand-written ternary GEMM that streams 2-bit weights and expands them in tile memory cuts weight traffic per pass by about 4x against INT8. Treat it as phase 5b research, not the plan of record.

### 3.9 Quantization

| phase | precision | criterion |
|---|---|---|
| 1 to 3 | bf16 training, fp32 eval | none; establish quality |
| 5 | quantization-aware training, W8A8, per-channel weights, per-tensor activations, about 10 percent extra steps; Quark static quantization for the Vitis AI provider | compliance within 0.5 points of bf16; if the INT8 compile fails on transformer ops, ship BF16 on the NPU and INT8 on the CPU |
| 5b, optional | ternary weights, 8-bit activations, trained from scratch per the BitNet b1.58 recipe; embeddings and output head stay INT8; deployed through a custom IRON kernel on XDNA2 or a lookup-table kernel on the CPU | within 1 point of bf16 and measurably faster on the target, else drop it |

**Rough budget for the W8A8 model.**

| | estimate |
|---|---|
| parameters | 15M to 40M |
| on disk, model plus lexicon | 20MB to 45MB |
| compute per pass, 128-token bucket | 4 to 10 GFLOP |
| passes before exit | 2 to 4 typical |
| CPU latency per sentence, 4 threads | 50ms to 300ms |
| XDNA2 latency per pass | a few ms, overhead-dominated; see 3.8 |

These are back-of-envelope numbers. The latency table in `ste/eval/` replaces them the day we have a model.

---

## 4. Evaluation, `ste/eval/`

| metric | how | target |
|---|---|---|
| compliance | tier A violations per 100 output sentences, from the checker | at most 2 |
| meaning preservation | a judge model from a different family than the teacher, five-point scale, plus two-way entailment; 200 sentences human-checked per release | mean at least 4.5; at most 1 percent rated "meaning changed" |
| over-editing | token edit rate on input that is already compliant | at most 3 percent |
| slot fidelity | every Technical Name, number, and protected word restored exactly | 100 percent, hard gate |
| flags | share of sentences returned with an unresolved flag | reported, not targeted; drive it down over time |
| passes | mean and p95 pass count | reported |
| latency | p50 and p95 per sentence and per six-sentence paragraph, reference laptop CPU, 4 threads | p95 sentence under 500ms; stretch 150ms |
| size | on disk including lexicon | under 100MB; stretch 40MB |

**Ablations to run once.** Checker feedback on and off. Slots on and off. One pass versus adaptive. W8A8 versus ternary versus bf16. Tagger versus canvas on the structural-rule subset of the gold set.

Deterministic scoring everywhere the checker can do it, as in `bench/`. The judge is only for what the checker cannot see.

---

## 5. Phases

| phase | deliverable | exit criterion | effort |
|---|---|---|---|
| 0 | lexicon and checker | every dictionary entry encoded and reviewed; checker passes all spec example pairs | 2 to 3 weeks |
| 1 | glossary format, slotter, inflector | 100 percent slot restoration on a synthetic test; checker respects glossaries | 1 week |
| 2 | data v1 and gold set v1 | 1M pairs; every target checker-clean; judge-sampled meaning at least 4.5; 1,000-sentence gold set | 2 weeks |
| 3 | tagger baseline; ONNX export of a random-weight model to CPU and one NPU | all metrics measured; export runs | 2 weeks |
| 4 | canvas model | beats the tagger on the structural-rule subset at equal latency, or we keep the tagger | 3 to 4 weeks |
| 5 | W8A8 model on ONNX Runtime CPU and one NPU; optional ternary CPU variant | within budget on the reference machine | 2 to 3 weeks |
| 6 | hardening | gold set growth; human review loop; domain adapter training script; per-customer glossary tooling | ongoing |

Roughly three to four months for one person. Compute is small: these are tiny models, and synthetic generation dominates the bill. Expect a few hundred dollars of rented GPU, in line with the heal campaign.

**First week, concretely.**
1. Register for and download the Issue 9 PDF.
2. Extract the dictionary text and get the first 100 rows reviewed to settle the JSONL schema.
3. Extract every rule's example pairs into `ste/checker/tests/spec_examples.jsonl`.
4. Write the tier A checker for the five lookup rules: unapproved word, wrong part of speech, sentence length, -ing form, perfect tense.
5. Run the checker over the example pairs and start the failure list.

---

## 6. Risks and mitigations

| risk | mitigation |
|---|---|
| lexicon encoding errors poison everything downstream | spec example pairs as unit tests; two-person review of every row; a diff tool against the PDF text |
| synthetic messy English does not match what real writers produce | several LLM noise styles; collect real unlabeled technical English from public sources and have the teacher label it; grow the gold set from real text only |
| meaning drift | judge plus human spot checks; conservative protect-and-flag default; the product shows a diff and never auto-publishes |
| NPU op support gaps found late | boring graph; random-weight export in phase 3, not phase 5 |
| small-scale masked diffusion underperforms | the tagger is the fallback and is built first |
| -ing words inside Technical Names, like "landing gear" or "bearing" | glossary and lexicon handle them; a dedicated test set for -ing edge cases |
| passive detection false positives on adjectival participles, like "the valve is closed" | tier A rule only fires with an explicit agent or in `PROC` mode; otherwise tier B |
| license contamination from third-party encodings | own encoding from the spec; no AGPL imports |
| scope creep toward a general simplifier | the lexicon mask makes this impossible by construction, which is the point |

---

## 7. Open questions for the owner

1. Windows or Linux for the shipped XDNA2 runtime? The official execution provider is Windows-only today. Linux means IRON and our own kernels from day one, which is a different project.
2. Is the primary product surface a sentence, a paragraph, or a whole document with structure? This decides whether the 256 bucket matters and whether we need cross-paragraph terminology consistency.
3. Do we have any real STE manuals, even internal ones, to use for evaluation? Even a few hundred sentences would change the data plan.
4. Is the API spend in 3.4 acceptable? Roughly $100 for the hard-case slice, $10 to $30 per evaluation round, and optionally $80 to skip a week of laptop time on the volume slices.

---

## 8. Repo layout

```
ste/
  PLAN.md                      this file
  lexicon/
    build_lexicon.py           PDF text to JSONL rows, for human review
    lexicon.jsonl              the reviewed dictionary
    tokenizer.py               word-level tokenizer, V_in and V_out
  checker/
    rules.py                   one function per rule, tagged by tier
    checker.py                 check(tokens, mode, glossary)
    tests/spec_examples.jsonl  the spec's own example pairs
  glossary/
    schema.json
    slotter.py                 slot and restore
    inflect.py                 technical verb forms
  data/
    seed_ste.py                teacher-generated clean STE, checker-filtered
    noise_rules.py             reversed-dictionary and structural noise
    noise_llm.py               teacher back-translation
    distill_hard.py            teacher with checker in the loop
    build_pairs.py             assemble, dedupe, bucket, align
  model_tagger/                Model A
  model_canvas/                Model B
  eval/
    metrics.py                 compliance, over-editing, slot fidelity, latency
    judge.py                   meaning preservation
    gold/                      the hand-verified set, by rule category
  runtime/
    export_onnx.py
    loop.py                    the host-side pass loop
    backends/                  ort_cpu.py, openvino_npu.py, qnn.py, coreml.py
```

---

## 9. Reading list

- ASD-STE100 Issue 9, January 2025. asd-ste100.org.
- Ghazvininejad, Levy, Liu, Zettlemoyer. 2019. *Mask-Predict: Parallel Decoding of Conditional Masked Language Models.* The confidence-freeze loop in section 3.6 is this.
- Omelianchuk, Atrasevych, Chernodub, Skurzhanskyi. 2020. *GECToR: Grammatical Error Correction, Tag, Not Rewrite.* Model A.
- Gu, Wang, Zhao. 2019. *Levenshtein Transformer.* Insertion and deletion as iterative refinement.
- Savinov, Chung, Binkowski, Elsen, van den Oord. 2022. *Step-unrolled Denoising Autoencoders for Text Generation.* Non-autoregressive iterative denoising at small scale.
- Reid, Hellendoorn, Neubig. 2022. *DiffusER: Discrete Diffusion via Edit-based Reconstruction.* Edit-centric diffusion, closest to Model B.
- Sahoo et al. 2024. *Simple and Effective Masked Diffusion Language Models.* The training objective in section 3.6.
- Nie et al. 2025. *Large Language Diffusion Models.* Evidence the masked-diffusion recipe scales; we are going the other way.
- Ma et al. 2024. *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* The ternary recipe, if phase 5b happens.
- Wei et al. 2024. *T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge.* The CPU kernel approach for ternary.
- AMD. *Ryzen AI Software documentation*, release 1.7.x. ryzenai.docs.amd.com. Model compatibility, the Vitis AI execution provider, Quark quantization, supported ops.
- AMD / Xilinx. *mlir-aie* repository and the IRON tutorial, ASPLOS 2026. The custom-kernel path for XDNA2.
- Linux kernel documentation, *accel/amdxdna*. Driver, firmware loading, and XRT layering for the NPU on Linux.
