# Plan: a tiny iterative editor that rewrites English into ASD-STE100

*Draft, October 2026. Personal project. Guideline targets, not gates: under 100 MB on disk, under 500 ms per sentence on a laptop CPU, the XDNA2 NPU on Strix Point as the preferred deployment target on Windows and Linux, GGUF as the model format, no API spend. Data generation and training run on a rented pod.*

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
| model format | GGUF | One file format that llama.cpp, ggml-xdna, and OpenFlowLM-Next's converter all read. ggml already defines ternary types. We add our own architecture to llama.cpp. See 3.8. |
| NPU target | AMD XDNA2 on Strix Point, on Windows and Linux, through OpenFlowLM-Next's engine, with CPU and Vulkan through llama.cpp as the reference | Both of your NPU projects support both operating systems. But the NPU may be slower than the CPU for a model this narrow. We measure before committing. See 3.8. |
| quantization | Q8_0 on CPU and Vulkan, bf16 on the NPU, TQ2_0 ternary as a CPU experiment | The NPU kernels in both projects take bf16 and keep their own 8-bit weight copies, so there is nothing to quantize for them. Width is a multiple of 256 so ggml's block types fit. See 3.9. |
| teachers and training | one flag-gated session on a rented pod | Three days of laptop time was the real cost, and training the student is under an hour. Same models, run in large batches on a GPU. No API spend. See 3.4. |
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

### 3.0 The development machine and the pod

**The laptop** is a Strix Point machine: Ryzen AI 9 HX 370-class, 12 Zen 5 cores, Radeon 890M iGPU, XDNA2 NPU, and 96 GB of LPDDR5X. It is where the student runs, so every latency number in this plan comes from it. It also runs the converter, the checker, the rule-based noiser, and small experiments. It does not run the teachers. Three days of generation would tie it up, and the memory bus (about 120 GB/s) makes big models slow there.

**The pod** is a rented GPU box for one session that does the heavy work: teacher generation, the judge, student training, and GGUF export. Section 3.4 has the plan and the cost.

**Operating systems.** Windows and Linux both work. OpenFlowLM-Next ships a Windows installer and a Linux guide. ggml-xdna builds for both, with Linux not yet run on an NPU by its author. Pod scripts are bash; laptop scripts are Python.

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

There is no public parallel corpus of plain English to STE. We build it. Generation runs on a rented pod, rule-based noising runs anywhere, and nothing calls an API.

**Seed clean STE.**
- The 1,878 aligned pairs from the dictionary and the rule examples. Small, but they are the only human-written gold we have, so they seed the gold set and the checker tests, not training.
- Teacher-generated STE. Prompt the volume teacher with a short rule summary and a source text, generate a rewrite, keep it only if the checker reports zero tier A violations. Source texts are the kind of thing this tool will actually see: public instruction datasets, system prompts and agent instructions, README and documentation text, tool and API descriptions, forum questions. Not aircraft manuals.

**Teacher plan.** Three jobs, three models. The checker filters compliance on every slice, so what a stronger teacher buys is meaning fidelity and natural phrasing, not rule knowledge.

| job | teacher | why |
|---|---|---|
| volume: seed STE and LLM back-translation, about 11M output tokens | Qwen3.6-35B-A3B, unpruned, from Hugging Face, run with vLLM | `heal/pod_run.sh` already downloads it. It is stronger than the pruned build, and vLLM batches it hard. Keep the prompt to a 300-token rule summary plus the one rule being targeted. |
| hard cases: passive-to-active and sentence splitting, 50k pairs with retries, about 4.5M tokens | Qwen3.8-Flash-Next-131B-A6B, the 65 GB build from this repo, under llama.cpp CUDA | A weak rewriter passes the checker and quietly changes meaning. This is the strongest model that fits. `bench/pod_bench.sh` already builds llama.cpp for CUDA. |
| judge: meaning preservation and downstream equivalence | gpt-oss-120b MXFP4, about 63 GB, run with vLLM | Different lineage from both teachers, so its blind spots are not theirs. |

**One session, flag-gated.** Model it on `bench/pod_bench.sh`: each phase writes a done flag so a rerun resumes, any failure writes `FAILED.flag`, and a wall-clock cap stops the pod. Phases: `env`, `volume`, `hard`, `judge`, `train`, `export`, then pull the artifacts home. Log elapsed time and running cost per phase, as `heal/pod_gate.sh` does.

**Smoke session first.** Before the real run, rent the pod for about 20 minutes with a spend cap of about $2. Run 200 requests through each teacher and read the throughput and cost from the log. Every number in the table below is my guess until that log exists.

**Cost, estimated and unmeasured.** The repo's notes price an H200 community pod at $3.59 per hour.

| tier | hardware | what runs | rough pod hours | rough cost |
|---|---|---|---|---|
| lean | one 48 GB card | the 35B-A3B at Q4 for volume and hard cases; skip the 131B; judge with a smaller model | 4 to 8 | $3 to $10 |
| standard | one H200 | all three teachers as listed, then training | 4 to 6 | $15 to $30 |

Downloads eat pod time: the 131B and the judge together are about 130 GB. Cache them on a network volume if you expect a second run. The lean tier is the one closest to your $1 to $3 line, and it still probably lands above it. Say which tier you want after the smoke session.

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

Reuse `heal/gen_teacher.py`'s shard writer and `bench/run_bench.py`'s `llama-server` client. Neither needs much change. The pod session scripts copy the layout of `bench/pod_bench.sh` and `heal/pod_run.sh`.

**Gold set.** 300 sentences and 50 paragraphs. Source text is real messy input of the kind this tool will see; the STE rewrite is hand-verified by the owner. Tagged by rule category so per-rule accuracy is visible. The dictionary's 1,878 pairs are the seed; the rest comes from real prompts and docs. This set is the only thing that tells us the truth, so it gets built in phase 2 and grows every phase.

### 3.5 Model A: iterative tagger baseline, `ste/model_tagger/`

GECToR-style. An encoder with a per-token tag head.

| | |
|---|---|
| encoder | 6 layers, width 512, 8 heads, FFN 2048, learned absolute positions, LayerNorm, GELU |
| parameters | about 25M including embeddings |
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

**Format: GGUF.** The model is written once to a GGUF file by `export_gguf.py`. The host loop (freezing, re-masking, the checker, slot restore) lives outside the graph in a small library.

**What the two NPU projects actually do.** I read both repos in October 2026.

- **ggml-xdna** adds an NPU backend to llama.cpp. It offloads big matrix multiplies during prompt reading only. It declines chunks under 1,024 tokens. Its own tests show models narrower than 2,048 get slower on the NPU: 0.77x at width 1,536 and 0.89x at width 1,024. Our model is 512 wide and sees 64 to 256 tokens per call. Under ggml-xdna as written, the NPU would decline almost all of our work, and what it took would likely lose to the CPU.
- **OpenFlowLM-Next** is a fork of FastFlowLM, an NPU-first runtime for Windows and Linux. It converts GGUF files to its own format with `q4nx-build`, one builder per architecture, and each model links to a shared kernel binary for its family. It also has an open engine pattern, `open_embedding`, which is the closest thing to our model: a bidirectional encoder (EmbeddingGemma-300M) run in FP32 on the CPU, with the big projections sent to the NPU as bf16 matrix multiplies, and results checked against the CPU at cosine 0.999 or better. The repo's `npu_offload_pipeline` skill documents how to compile kernels for new shapes.

**Plan, in order.**

1. **Reference: llama.cpp with a new `ste` architecture, on CPU and Vulkan.** Bidirectional attention, learned absolute positions, LayerNorm, GELU, a logits head over `V_out`, and the lexicon bias as an input. The patch lives beside the existing one in `patches/`. Two things to verify before writing it: that llama.cpp can return logits at every position of a non-causal batch, and that a per-call additive bias is easy to add. If either is hard, the patch grows. This is the baseline all latency numbers are measured against.
2. **NPU: an `open_ste` engine in OpenFlowLM-Next,** cloned from `open_embedding`. Load from the GGUF through a new `ste.py` builder in `q4nx-build`, or the safetensors route that `open_embedding` uses, whichever is less work. Compile kernels for our shapes: M of 64, 128, 256 and K and N from 512 and 2048. Offload only the dense projections first, as `open_embedding` does.
3. **The deciding measurement.** In week one of phase 3, take a random-weight model of the final shape and time one full pass on three paths: llama.cpp CPU, llama.cpp Vulkan, and the NPU engine. At width 512, per-dispatch overhead may eat the gain, which is exactly what ggml-xdna measured. If the NPU loses, the answer is a fused kernel that runs a whole layer or a whole pass in one dispatch, which is a larger IRON project, or the answer is to ship on CPU. A 25M-parameter model on 12 Zen 5 cores is probably fast enough that CPU is a fine place to stop.
4. **Dropped from the earlier plan:** the ONNX Runtime and Vitis AI path. GGUF replaces it.

**Not planned:** any NPU path that depends on closed kernels. OpenFlowLM-Next's own notes say to replace closed pieces with open ones, and we follow that.

### 3.9 Quantization

| phase | precision | criterion |
|---|---|---|
| 1 to 3 | bf16 training, fp32 eval | none; establish quality |
| 5 | GGUF export: F16 and Q8_0 for CPU and Vulkan; plain bf16 weights for the NPU path | compliance within 0.5 points of bf16 |
| 5b, optional | ternary weights trained from scratch per the BitNet b1.58 recipe, stored as ggml's TQ2_0 type; embeddings and output head stay at Q8_0; CPU only | within 1 point of bf16 and measurably faster on the CPU, else drop it |

**Why width 512.** ggml's k-quant and ternary types work in blocks of 256, so every weight dimension should be a multiple of 256. The Qwen3.6 campaign in this repo hit exactly this: width 384 pushed expert tensors into a large fallback format and decode fell to 9.4 tok/s. Width 512 with FFN 2048 avoids it. The legacy Q8_0 and Q4_0 types use blocks of 32 and would also work at 384, but ternary needs 256.

**Rough budget for the Q8_0 model.**

| | estimate |
|---|---|
| parameters | 15M to 40M |
| on disk, model plus lexicon | 20MB to 45MB |
| compute per pass, 128-token bucket | 4 to 10 GFLOP |
| passes before exit | 2 to 4 typical |
| CPU latency per sentence, 4 threads | 50ms to 300ms |
| XDNA2 latency per pass | unknown; may be slower than the CPU at this width; see 3.8 |

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

**Ablations to run once.** Checker feedback on and off. Slots on and off. One pass versus adaptive. Q8_0 versus ternary versus bf16. Tagger versus canvas on the structural-rule subset of the gold set.

Deterministic scoring everywhere the checker can do it, as in `bench/`. The judge is only for what the checker cannot see.

---

## 5. Phases

| phase | deliverable | exit criterion | effort |
|---|---|---|---|
| 0 | converter from the PyPI packages, lexicon, checker | every dictionary row converted or on the review list; checker passes the rule examples and the 1,878 pairs | 1 week |
| 1 | protected-span detector, slotter, inflector | 100 percent restoration on a synthetic test; checker respects glossaries | 1 week |
| 2 | data v1 (pod session), gold set v1, and the token-delta measurement on real inputs | 1M pairs; every target checker-clean; judge-sampled meaning at least 4.5; 300-sentence gold set; a decision on whether STE saves tokens on our inputs | 2 weeks |
| 3 | tagger baseline (trained on the pod); GGUF writer and `ste` architecture in llama.cpp; a random-weight model of the final shape timed on CPU, Vulkan, and the NPU engine | all metrics measured; GGUF loads; NPU dispatch overhead measured at our width | 2 weeks |
| 4 | canvas model | beats the tagger on the structural-rule subset at equal latency, or we keep the tagger | 3 to 4 weeks |
| 5 | Q8_0 on CPU and Vulkan, bf16 on the NPU through OpenFlowLM-Next; optional TQ2_0 ternary on CPU; a fused NPU kernel only if the measurement in 3.8 says it can win | within budget on the Strix Point box | 2 to 3 weeks |
| 6 | hardening | gold set growth; codebook output; adapter training script if a domain ever needs one | ongoing |

Roughly two and a half to three months for one person. The laptop does everything except the pod session. Training a 25M parameter model on 1M to 2M short pairs is under an hour on an H200, so the pod bill is dominated by teacher generation and model downloads, not training.

**First week, concretely.**
1. `pip download` both packages into a scratch directory. Add the derived files to `.gitignore` before writing the converter.
2. Write `build_lexicon.py`: PyPI JSONL in, our schema out, review list for odd rows, `pairs.jsonl` for the 1,878 aligned examples.
3. Write the inflection generator and check it against the 663 spellings the package lists.
4. Extract the rule examples from the rules JSON into `spec_examples.jsonl`, keyed by rule ID.
5. Write the tier A checker for the five lookup rules: unapproved word, wrong part of speech, sentence length, -ing form, perfect tense. Run it over the examples and start the failure list.
6. Take 200 real prompts and docs from your own use, run them through the local Qwen3.6-27B-A2.8B with the checker, and count tokens before and after. Two hundred prompts take minutes, so this needs no pod. This is the number that decides what this project is for.

---

## 6. Risks and mitigations

| risk | mitigation |
|---|---|
| STE does not save tokens on our inputs | STE was designed for human readers, not token counts. It adds articles and "make sure", and splitting adds subjects. On terse input it may cost tokens; on verbose input it saves them. Measure in week one. If it does not save tokens, the goal is normalization and the codebook encoding, and that is still worth having. |
| converter errors poison everything downstream | the package's own examples as unit tests; the review list; spot-check odd rows against the PDF |
| synthetic messy English does not match real prompts and docs | several LLM noise styles; seed from real prompts, system prompts, and docs; grow the gold set from real text only |
| meaning drift | local judge plus downstream equivalence on real prompts; protect-and-copy default; owner spot checks |
| code and identifiers get rewritten | the protect step runs before anything else and treats backticks, paths, URLs, and identifier shapes as untouchable; slot fidelity is a hard gate |
| NPU slower than CPU at this width | ggml-xdna measured losses below width 2,048; time a random-weight model on CPU, Vulkan, and the NPU in phase 3; CPU is a fine shipping answer for a model this small |
| llama.cpp cannot return per-position logits or take a bias cleanly | check before writing the architecture; the patch directory already exists; worst case the patch is larger |
| OpenFlowLM-Next needs a new builder and kernels per architecture | start from `open_embedding`, the closest existing encoder; only attempt after the CPU path works |
| pod cost runs past the budget | smoke session with a spend cap first; flag-gated phases; wall-clock cap; lean tier available |
| small-scale masked diffusion underperforms | the tagger is the fallback and is built first |
| -ing words inside names, like "landing gear" or "logging" | glossary and protect step handle them; a dedicated test set for -ing edge cases |
| passive detection false positives on adjectival participles, like "the file is closed" | tier A rule only fires with an explicit agent or in `PROC` mode; otherwise tier B |
| AGPL-derived data in a public Apache-2.0 repo | the derived lexicon and rule files are gitignored and rebuilt from PyPI at build time; nothing from the packages is committed |
| scope creep toward a general simplifier | the lexicon mask makes this impossible by construction, which is the point |

---

## 7. Open questions, in plain words

None of these block work. Each has a default. Say nothing and the default stands.

1. **Which AI will read the shorter text?** We measure success partly by counting tokens before and after the rewrite. Every AI model splits text into tokens its own way, so the count depends on which model you pick. *Default:* count with the Qwen3 tokenizer from this repo, plus one other, and report both.
2. **How much text goes in at once?** One sentence per call is fast and simple. A whole prompt of several paragraphs is more useful but needs bigger buffers and more passes. *Default:* build for sentences and short paragraphs first (the 64 and 128 sizes), and add the 256 size later.
3. **Do you want a compact number format for the output?** The output vocabulary is under 4,096 words. So every word could be saved as a 12-bit number instead of text. That is a compressed form, useful if a program, not a model, stores or sends the text. *Default:* no. Skip it until the text version works. It costs nothing to add later.
4. **How much will you spend on the pod?** Section 3.4 has two tiers, about $3 to $10 and about $15 to $30. Both are my guesses. *Default:* run the 20-minute smoke session first, capped near $2, then pick a tier from its real numbers.

Settled: the model format is GGUF. Both Windows and Linux are supported through your two NPU projects. Training and data generation run on a pod.

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
    export_gguf.py             writes the trained model to GGUF
    loop.py                    the host-side pass loop
    codebook.py                12-bit word-ID encode and decode
    backends/                  llama_cpp.py (CPU, Vulkan), openflowlm.py (NPU)
  pod/
    pod_smoke.sh               20-minute throughput and cost probe
    pod_run.sh                 flag-gated session: env, volume, hard, judge, train, export
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
- Cyronius/ggml-xdna. NPU prompt offload for llama.cpp; its Results and Limitations sections are the evidence on narrow models.
- Cyronius/OpenFlowLM-Next, a FastFlowLM fork. `open_embedding`, `q4nx-build`, and the `npu_offload_pipeline` skill.
- AMD / Xilinx. *mlir-aie* repository and the IRON tutorial, ASPLOS 2026. The custom-kernel path for XDNA2.
- Linux kernel documentation, *accel/amdxdna*. Driver, firmware loading, and XRT layering for the NPU on Linux.
- dfch. *biz-dfch-asdste100vocab* 0.8.1 and *biz-dfch-asdste100rules* 0.1.1 on PyPI. The lexicon and rule data this plan builds from.
