# Plan: a tiny iterative editor that rewrites English into ASD-STE100

*Draft, October 2026. Personal project. Guideline targets, not gates: under 100 MB on disk, under 500 ms per sentence on a laptop CPU, the XDNA2 NPU on Strix Point as the preferred deployment target, zero API spend.*

This is the design record for the Simplified Technical English (STE) rewriter. Read it before touching anything under `ste/`. The other plan in this repo, the heal pipeline, is referenced the same way from `heal/README.md`.

---

## 0. What we are building

**Purpose.** Normalize English that goes into or comes out of an LLM: prompts, system prompts, tool descriptions, documentation, agent-to-agent messages. STE is a closed-vocabulary controlled language with about 900 approved words and 53 writing rules. Text in STE is shorter where English is verbose, has one meaning per word, one instruction per sentence, and no tense or voice variety. That makes it predictable for a model to read and cheap to encode. The consumer of the output is a model or a codec, not a human reader.

**Input:** English text, one sentence or one paragraph at a time, plus a mode flag: `PROC` for instructions and prompts, `DESC` for context and explanations. Optionally a glossary of terms to protect.

**Output:** the STE rewrite, a token-aligned diff, and the list of spans the model left unchanged because it was not confident. Optionally, the same text as a sequence of 12-bit word IDs over the closed output vocabulary, for storage or transfer between agents you control.

**Shape:** a small bidirectional transformer that iteratively edits a fixed-length canvas. The canvas starts as a copy of the source. Every pass runs under a hard lexicon mask, so the model cannot emit an unapproved word. A deterministic rule checker runs between passes. It feeds violations back into the next pass and decides when to stop.

**What it is not:** an autoregressive LLM, a general simplifier, or a certified technical-writing tool. When it is unsure it leaves text alone, which is the right failure mode for a preprocessor.

---

## 1. Decisions already made, and why

| decision | choice | why |
|---|---|---|
| where the vocabulary constraint lives | output side only | The encoder must read any English, including unapproved words it has to recognize in order to replace them. The decoder vocab is the STE lexicon plus slots, so the constraint is one logit mask. |
| generation scheme | non-autoregressive iterative refinement over a canvas | Most of the output is a copy of the input. Parallel decoding makes latency a function of passes, not length. Fixed shapes suit NPUs. This is Mask-Predict with a source-initialized canvas, which the masked-diffusion literature later formalized. |
| canvas init | source tokens with NULL padding, not all-mask | Starting from all masks throws away the 70 percent of tokens that are already right. |
| exit decider | deterministic checker, plus "nothing changed", plus a hard cap | Checking STE is mostly lookups. Fixing it is not. Rules verify, the model generates. |
| checker in the loop | per-token violation features fed into each pass | Turns blind refinement into guided repair. Cheap: the features are lookups. |
| lexicon source | the `biz-dfch-asdste100vocab` and `biz-dfch-asdste100rules` packages from PyPI, fetched at build time | They carry the full Issue 9 dictionary with parts of speech, inflections, alternatives, 1,878 aligned example pairs, and all 53 rules with examples. Weeks of transcription become a converter script. Owner's call on the AGPL question for a personal project; see section 2 for the one practical consequence. |
| protected spans | slotted at the edges, never in the weights | Code, identifiers, paths, URLs, numbers, and proper nouns pass through untouched. STE rules 1.5 and 1.12 define the same mechanism for technical names and verbs. |
| unknown words | protect and copy | A preprocessor that leaves a word alone costs nothing. One that guesses costs meaning. |
| graph | static shapes, boring ops, loop on the host | Every NPU toolchain handles MatMul, LayerNorm, GELU, Softmax, Gather. Several choke on RoPE, SwiGLU, RMSNorm, or dynamic shapes. |
| NPU target | AMD XDNA2 on Strix Point, through ONNX Runtime and the Vitis AI execution provider | It is the machine on the desk. The official toolchain is Windows-only as of Ryzen AI 1.7; Linux has the kernel driver and IRON for custom kernels but no production execution provider. See 3.8. |
| quantization | W8A8 as the shipping target, BF16 as the fallback on XDNA2; ternary as a custom-kernel experiment | The Ryzen AI compiler accepts INT8 and BF16, nothing lower. Ternary packs to INT8 on the standard path and only pays off through a hand-written IRON kernel. At 20M to 40M parameters the model fits in cache on the CPU, so bandwidth is not the bottleneck there. |
| teachers | all local: Qwen3.6-27B-A2.8B for volume, Qwen3.8-Flash-Next-131B-A6B for hard cases, gpt-oss-120b as judge | The box is bandwidth-bound at about 120 GB/s, so only MoE teachers decode at a usable rate, and all three fit in 96 GB. The checker filters compliance, so the volume teacher can be modest. The judge is a different lineage from the teachers. No API spend. See 3.4. |
| build order | tagger baseline before the canvas model | A GECToR-style tagger is the simplest thing that can work. It shares the lexicon, checker, slots, and data. It sets the metric floor and is the fallback if small-scale diffusion disappoints. |
| pretrained weights | none | Our vocab is word-level and ours. A short MLM pretrain on technical English with our tokenizer is cheap if we want it. |

---

## 2. Spec facts the design depends on

Source: ASD-STE100 Issue 9, January 2025. The spec is free on request from asd-ste100.org. Rule numbers below follow Issue 9.

- **Two parts.** Part 1 is 53 writing rules in nine sections. Part 2 is the dictionary: about 900 approved words, each with exactly one approved meaning and one approved part of speech, plus a larger list of unapproved words with approved alternatives.
- **Technical Names (rule 1.5) and Technical Verbs (rule 1.12).** Words outside the dictionary are allowed when they name specific things and processes in the subject field. This is our mechanism for protected spans.
- **Sentence length.** Procedures: at most 20 words. Descriptive text: at most 25 words.
- **Paragraphs.** At most six sentences. One topic per paragraph.
- **Noun clusters.** At most three words.
- **Verbs.** Only the infinitive, imperative, simple present, simple past, future, and past participle as adjective. No -ing forms except inside Technical Names. No perfect tenses. Active voice, with narrow exceptions in descriptive text.
- **Procedures.** Imperative mood. One instruction per sentence.
- **Articles and "that".** Use them. Do not drop them to save words. This is the rule most likely to cost tokens; see the risk table.

**What the PyPI packages contain, verified October 2026.**

| package | version | contents |
|---|---|---|
| `biz-dfch-asdste100vocab` | 0.8.1 | 2,200 dictionary rows: 881 approved, 1,319 rejected. Each row has a part of speech, approved rows have meanings with STE example sentences, rejected rows have approved alternatives, and 1,878 of those alternatives carry an aligned non-STE and STE sentence pair. 663 inflected spellings. A separate file of 779 technical words with TN and TV categories. |
| `biz-dfch-asdste100rules` | 0.1.1 | 66 entries covering the 53 rules with section, category, summary, and content items that include the spec's examples. |

Both are AGPL-3.0 and unaffiliated with ASD. The owner has decided to use them. The one practical consequence: this repo is public and Apache-2.0, so the derived lexicon and rule files are not committed. The build step fetches the packages from PyPI and converts them into our schema locally. `ste/lexicon/lexicon.jsonl` and `ste/checker/tests/spec_examples.jsonl` go in `.gitignore`. The Issue 9 PDF is for spot-checking rows the converter flags as odd, nothing more.

---

## 3. Components

Everything lives under `ste/`. Each component gets its own README once it exists.

### 3.0 The development machine

A Strix Point laptop: Ryzen AI 9 HX 370-class, 12 Zen 5 cores, Radeon 890M iGPU, XDNA2 NPU rated at 50 TOPS INT8, and 96 GB of LPDDR5X on a 128-bit bus. For the teachers, the number that matters is the memory bus: about 120 GB/s theoretical, 80 to 100 GB/s in practice. Decode speed for any model on this box is roughly that bandwidth divided by the bytes touched per token. For the student, the number that matters is the NPU, covered in 3.8.

Three consequences:

- **MoE teachers only.** A dense 27B model at Q4 touches about 15 GB per token and decodes at 5 to 8 tok/s. A 27B-A2.8B MoE touches about 1.6 GB and decodes several times faster.
- **The 96 GB is what makes the big models possible.** Qwen3.8-Flash-Next-131B-A6B at 65 GB and gpt-oss-120b at about 63 GB each sit in RAM with room for KV cache. One at a time.
- **The NPU does not help the teachers.** Ryzen AI's LLM path targets small dense models on Windows and does not run a custom MoE GGUF. The iGPU speeds up prefill through Vulkan but shares the same memory bus, so it does not change decode speed.

### 3.1 Lexicon and tokenizer, `ste/lexicon/`

**Lexicon.** One JSONL row per dictionary entry, produced by `build_lexicon.py` from the PyPI data.

```
{"lemma": "ensure", "pos": "verb", "status": "unapproved",
 "alternatives": [{"lemma": "make sure", "pos": "verb"}],
 "inflections": ["ensure", "ensures", "ensured", "ensuring"],
 "examples": {"ste": [...], "non_ste": [...]}, "source": "STE100:9"}
```

The converter lowercases the dictionary's uppercase convention, expands inflections where the package lists them and generates them where it does not, splits words that are approved in one part of speech and rejected in another into two rows, and pulls the aligned example pairs into a separate `pairs.jsonl`. Rows with no part of speech or no alternatives are written to a review list. Budget: two to three days, most of it on the inflection generator and the review list.

**Tokenizer.** Word-level, lowercase, with a case feature.

| vocab region | contents | rough size |
|---|---|---|
| `V_out` | every inflected form of every approved word; function words; punctuation; digit tokens; slot tokens `TN_1..TN_32`, `TV_1..TV_16`, `NUM_1..NUM_16`, `PROTECT_1..PROTECT_16`; control tokens `NULL`, `PAD`, `SEP`, `PROC`, `DESC` | 3k to 4k |
| `V_in` | `V_out` plus every inflected form of every rejected word, plus a 4k to 8k BPE fallback for everything else | 12k to 16k |

The output head only ever scores `V_out`. The lexicon mask is a `[|V_out|]` additive bias, built once per request from the base lexicon plus any glossary additions. Because `V_out` is under 4,096 entries, a 12-bit ID per output word is a complete encoding of any STE text. That is the optional codebook output in section 0.

**Numbers never change.** Numbers, units, and identifiers go through `NUM` slots and are copied back verbatim.

**Case.** The detokenizer restores sentence-initial capitals and protected-span casing. Slots carry their original surface form.

### 3.2 Deterministic checker, `ste/checker/`

**Interface.** `check(tokens, mode, glossary) -> [Violation(rule_id, span, severity, suggestion)]`.

**A useful property.** Every approved word has exactly one approved part of speech. So on the *output* side, where the vocab is closed, part of speech is a lookup. The checker at inference time needs no tagger. A tagger is only needed on noisy *input* text, which only matters offline for data generation and diagnostics.

**Rule tiers.**

| tier | what | how |
|---|---|---|
| A, pure lookup | unapproved word; approved word used in an unapproved part of speech; words per sentence; sentences per paragraph; noun cluster length; -ing form outside a protected span; perfect tense ("has/have/had" plus participle); passive ("be" plus participle); more than one imperative per sentence in `PROC` mode | lexicon lookups, counting, and short pattern rules over the closed output vocab |
| B, heuristic | missing article before a singular count noun; "that" dropped after reporting verbs | pattern rules with known false-positive rates; reported with lower severity |
| C, not checkable | meaning preserved; the right alternative chosen; the downstream model reads it the same way | handled by evaluation, never by the checker |

**Five jobs for one checker.** Evaluation metric. Exit decision. Feedback features into the model. Filter for synthetic targets. The inverse of the noiser.

**Tests.** The example sentences from the rules package and the 1,878 aligned pairs from the dictionary. Every non-STE example must raise at least the rule it illustrates. Every STE example must raise nothing in tier A. Rule IDs in our checker match the package's `R1.1` style so failures point at the spec.

### 3.3 Protected spans and slotting, `ste/glossary/`

The input is LLM-facing text, so most protected spans are not aircraft parts. They are code, identifiers, paths, URLs, product names, and numbers. Detection is mostly automatic; the glossary is optional.

**Glossary format,** when one is wanted:

```
{"technical_names": [{"surface": ["bleed air valve"], "lemma": "bleed air valve", "category": "part"}],
 "technical_verbs": [{"lemma": "ream", "forms": {"base": "ream", "s": "reams", "ed": "reamed"}}],
 "forced_substitutions": [{"from": "a/c", "to": "aircraft"}],
 "extra_approved": ["token"]}
```

**Slotter.** Runs before the model and after it.

1. Apply forced substitutions. Deterministic, before anything else.
2. Protect code: anything inside backticks or fenced blocks, anything that looks like an identifier (`snake_case`, `camelCase`, dotted paths), file paths, URLs, and email addresses. These become `PROTECT_i`.
3. Longest-match glossary terms. Replace each with `TN_i` or `TV_i`, record the surface form in a side table.
4. Heuristics for unglossed names: capitalized multiword runs, alphanumeric codes, units.
5. Numbers become `NUM_i`.
6. Any remaining word that is in neither the lexicon nor the glossary becomes `PROTECT_i`.
7. After decoding, restore every slot from the side table. Technical verbs get inflected by a small rule table with exceptions. STE allows only base, -s, -ed, infinitive, and imperative, so the table is short.

**Training with slots.** Slot 70 percent of protectable spans in the training data and leave 30 percent raw, so the model tolerates detector misses.

**One glossary, two consumers.** The checker reads the same file, so it never flags a protected term as an unapproved word.

### 3.4 Data pipeline, `ste/data/`

There is no public parallel corpus of plain English to STE. We build it. Everything in this section runs on the laptop; nothing calls an API.

**Seed clean STE.**
- The 1,878 aligned pairs from the dictionary and the rule examples. Small, but they are the only human-written gold we have, so they seed the gold set and the checker tests, not training.
- Teacher-generated STE. Prompt the local volume teacher with a short rule summary and a source text, generate a rewrite, keep it only if the checker reports zero tier A violations. Source texts are the kind of thing this tool will actually see: public instruction datasets, system prompts and agent instructions, README and documentation text, tool and API descriptions, forum questions. Not aircraft manuals.

**Teacher plan.** Three jobs, three local models. The checker filters compliance on every slice, so what a stronger teacher buys is meaning fidelity and natural phrasing, not rule knowledge.

| job | teacher | why |
|---|---|---|
| volume: seed STE and LLM back-translation, about 11M output tokens | Qwen3.6-27B-A2.8B Q4_K_M with MTP, from this repo, behind `llama-server` | Fastest thing on the box, already tuned for it, and good enough for paraphrase with the checker downstream. Keep the prompt to a 300-token rule summary plus the one rule being targeted. Prefill is what hurts here, not decode. |
| hard cases: passive-to-active and sentence splitting, 50k pairs with retries, about 4.5M tokens | Qwen3.8-Flash-Next-131B-A6B at 65 GB, from this repo | A weak rewriter passes the checker and quietly changes meaning. This is the strongest model that fits. |
| judge: meaning preservation and downstream equivalence on the gold set and on sampled training pairs | gpt-oss-120b MXFP4, about 63 GB | Different lineage from both teachers, so its blind spots are not theirs. Strong enough to grade. |

**Local throughput, measured on day one.** Expect roughly 20 to 35 tok/s single stream for the 27B-A2.8B and 60 to 100 tok/s aggregate with 8 to 16 parallel slots; about half that for the two big models. Run `llama-server` with parallel slots, continuous batching, Vulkan offload for prefill, and prompt-cache reuse so the shared rule prefix is not re-prefilled per request. Size the context as slots times a per-slot budget of about 1k tokens.

**Time, rough.**

| slice | output tokens | laptop time |
|---|---|---|
| seed STE, 200k sentences with rejection sampling, 27B-A2.8B | about 5M | about 18 hours |
| LLM back-translation, 150k sentences, 27B-A2.8B | about 6M | about 20 hours |
| hard-case distillation, 75k requests with retries, 131B-A6B | about 4.5M | about 30 hours |
| judge, one round of 3k items, gpt-oss-120b | about 0.6M | about 4 hours |

Call it three days of background laptop time for the full set, plus an evening per evaluation round. Rule-based noising below is free and makes up most of the pairs.

**Noise the clean STE into messy English.** Two noisers, mixed about half and half.

| noiser | operations | cost |
|---|---|---|
| rule-based, `noise_rules.py` | reverse the dictionary: swap an approved word for one of the rejected words that lists it as an alternative; merge adjacent sentences with "and", "which", "while"; nominalize verbs into -ing forms; drop articles; insert "prior to", "in order to", "ensure that", "it is necessary to"; convert to perfect tense; convert imperative to "should be" passive | free, millions of pairs |
| LLM back-translation, `noise_llm.py` | the volume teacher rewrites clean STE as "how a person would normally write this", in several styles: verbose, passive, bureaucratic, terse-with-jargon, chatty | about 40 tokens per sentence |

The reversed dictionary is the key trick. The 1,319 rejected words with their alternatives, read backwards, are a labeled noise model for exactly the errors this task fixes.

**Hard-case distillation, `distill_hard.py`.** For the rules a noiser cannot invert cleanly, mainly passive-to-active and sentence splitting, run the big teacher on messy source with the rules in its prompt and the checker in a reject-and-resample loop, up to four tries. Keep only checker-clean outputs.

**Volume.**

| slice | size |
|---|---|
| rule-noised sentence pairs | 1M to 2M |
| LLM-noised sentence pairs | 150k |
| hard-case distilled pairs | 50k |
| paragraph pairs, 2 to 6 sentences | 100k |

Reuse `heal/gen_teacher.py`'s shard writer and `bench/run_bench.py`'s `llama-server` client. Neither needs much change.

**Gold set.** 300 sentences and 50 paragraphs. Source text is real messy input of the kind this tool will see; the STE rewrite is hand-verified by the owner. Tagged by rule category so per-rule accuracy is visible. The dictionary's 1,878 pairs are the seed; the rest comes from real prompts and docs. This set is the only thing that tells us the truth, so it gets built in phase 2 and grows every phase.

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
return collapse(canvas), diff, unresolved_spans
```

`collapse` drops `NULL` tokens and restores slots. If the cap is hit with violations remaining, return the pass with the fewest violations and report the violating spans as unresolved.

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
| token delta | tokens before and after under the tokenizer of the model that will consume the output, plus one other tokenizer | reported per input type; see the risk table |
| meaning preservation | local judge, five-point scale, plus two-way entailment; owner spot-checks 50 per release | mean at least 4.5; at most 1 percent rated "meaning changed" |
| downstream equivalence | run the same local model on the original and the STE version of 200 prompts; judge whether the answers match | at least 95 percent judged equivalent |
| over-editing | token edit rate on input that is already compliant | at most 3 percent |
| slot fidelity | every protected span and number restored exactly | 100 percent, hard gate |
| unresolved | share of sentences returned with an unresolved span | reported, not targeted; drive it down over time |
| passes | mean and p95 pass count | reported |
| latency | p50 and p95 per sentence and per six-sentence paragraph, Strix Point CPU, 4 threads; same on XDNA2 | p95 sentence under 500ms on CPU; stretch 150ms |
| size | on disk including lexicon | under 100MB; stretch 40MB |

**Ablations to run once.** Checker feedback on and off. Slots on and off. One pass versus adaptive. W8A8 versus ternary versus bf16. Tagger versus canvas on the structural-rule subset of the gold set.

Deterministic scoring everywhere the checker can do it, as in `bench/`. The judge is only for what the checker cannot see.

---

## 5. Phases

| phase | deliverable | exit criterion | effort |
|---|---|---|---|
| 0 | converter from the PyPI packages, lexicon, checker | every dictionary row converted or on the review list; checker passes the rule examples and the 1,878 pairs | 1 week |
| 1 | protected-span detector, slotter, inflector | 100 percent restoration on a synthetic test; checker respects glossaries | 1 week |
| 2 | data v1, gold set v1, and the token-delta measurement on real inputs | 1M pairs; every target checker-clean; judge-sampled meaning at least 4.5; 300-sentence gold set; a decision on whether STE saves tokens on our inputs | 2 weeks |
| 3 | tagger baseline; ONNX export of a random-weight model through the Vitis AI provider | all metrics measured; export runs with zero CPU-fallback nodes | 2 weeks |
| 4 | canvas model | beats the tagger on the structural-rule subset at equal latency, or we keep the tagger | 3 to 4 weeks |
| 5 | W8A8 model on ONNX Runtime CPU and on XDNA2; optional ternary IRON kernel | within budget on the Strix Point box | 2 to 3 weeks |
| 6 | hardening | gold set growth; codebook output; adapter training script if a domain ever needs one | ongoing |

Roughly two and a half to three months for one person. Compute is the laptop. Training a 15M to 40M parameter model on 1M to 2M short pairs is hours on the iGPU or a day on CPU; if it drags, one rented GPU-hour is the only line item, and it is optional.

**First week, concretely.**
1. `pip download` both packages into a scratch directory. Add the derived files to `.gitignore` before writing the converter.
2. Write `build_lexicon.py`: PyPI JSONL in, our schema out, review list for odd rows, `pairs.jsonl` for the 1,878 aligned examples.
3. Write the inflection generator and check it against the 663 spellings the package lists.
4. Extract the rule examples from the rules JSON into `spec_examples.jsonl`, keyed by rule ID.
5. Write the tier A checker for the five lookup rules: unapproved word, wrong part of speech, sentence length, -ing form, perfect tense. Run it over the examples and start the failure list.
6. Take 200 real prompts and docs from your own use, run them through the volume teacher with the checker, and count tokens before and after. This is the number that decides what this project is for.

---

## 6. Risks and mitigations

| risk | mitigation |
|---|---|
| STE does not save tokens on our inputs | STE was designed for human readers, not token counts. It adds articles and "make sure", and splitting adds subjects. On terse input it may cost tokens; on verbose input it saves them. Measure in week one. If it does not save tokens, the goal is normalization and the codebook encoding, and that is still worth having. |
| converter errors poison everything downstream | the package's own examples as unit tests; the review list; spot-check odd rows against the PDF |
| synthetic messy English does not match real prompts and docs | several LLM noise styles; seed from real prompts, system prompts, and docs; grow the gold set from real text only |
| meaning drift | local judge plus downstream equivalence on real prompts; protect-and-copy default; owner spot checks |
| code and identifiers get rewritten | the protect step runs before anything else and treats backticks, paths, URLs, and identifier shapes as untouchable; slot fidelity is a hard gate |
| NPU op support gaps found late | boring graph; random-weight export in phase 3, not phase 5 |
| small-scale masked diffusion underperforms | the tagger is the fallback and is built first |
| -ing words inside names, like "landing gear" or "logging" | glossary and protect step handle them; a dedicated test set for -ing edge cases |
| passive detection false positives on adjectival participles, like "the file is closed" | tier A rule only fires with an explicit agent or in `PROC` mode; otherwise tier B |
| AGPL-derived data in a public Apache-2.0 repo | the derived lexicon and rule files are gitignored and rebuilt from PyPI at build time; nothing from the packages is committed |
| scope creep toward a general simplifier | the lexicon mask makes this impossible by construction, which is the point |

---

## 7. Open questions for the owner

1. Windows or Linux for the shipped XDNA2 runtime? The official execution provider is Windows-only today. Linux means IRON and our own kernels from day one, which is a different project.
2. Which model consumes the output? Its tokenizer is the one the token-delta metric should use.
3. Is the primary surface a sentence, a paragraph, or a whole prompt with structure? This decides whether the 256 bucket matters and whether we need cross-paragraph terminology consistency.
4. Is the 12-bit codebook output wanted as a real feature, or is STE text itself the deliverable?

---

## 8. Repo layout

```
ste/
  PLAN.md                      this file
  lexicon/
    build_lexicon.py           PyPI JSONL to our schema, review list, pairs.jsonl
    inflect.py                 inflection generator, checked against the package's spellings
    lexicon.jsonl              derived, gitignored
    pairs.jsonl                derived, gitignored
    tokenizer.py               word-level tokenizer, V_in and V_out
  checker/
    rules.py                   one function per rule, tagged by tier, IDs match the spec
    checker.py                 check(tokens, mode, glossary)
    tests/spec_examples.jsonl  derived from the rules package, gitignored
  glossary/
    schema.json
    protect.py                 code, identifier, path, URL detection
    slotter.py                 slot and restore
  data/
    seed_ste.py                teacher-generated clean STE, checker-filtered
    noise_rules.py             reversed-dictionary and structural noise
    noise_llm.py               teacher back-translation
    distill_hard.py            big teacher with checker in the loop
    build_pairs.py             assemble, dedupe, bucket, align
  model_tagger/                Model A
  model_canvas/                Model B
  eval/
    metrics.py                 compliance, token delta, over-editing, slot fidelity, latency
    judge.py                   meaning preservation and downstream equivalence, local models
    gold/                      the hand-verified set, by rule category
  runtime/
    export_onnx.py
    loop.py                    the host-side pass loop
    codebook.py                12-bit word-ID encode and decode
    backends/                  ort_cpu.py, vitisai_npu.py, later openvino.py, qnn.py, coreml.py
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
- dfch. *biz-dfch-asdste100vocab* 0.8.1 and *biz-dfch-asdste100rules* 0.1.1 on PyPI. The lexicon and rule data this plan builds from.
