# Technical documentation teacher pilot

This is a bounded local data and synthesis pilot for ASD-STE100. It does not
train a model or mark generated text as verified gold.

## Frozen sources

`prepare.py` saves exact upstream files, license notices, commit IDs, URLs, and
SHA256 hashes in `data/out/doc-teacher-pilot-v1/source-manifest.json`.
`select_sources.py` deterministically selects 36 checker-flagged passages from
curl CLI, PostgreSQL, and Python documentation, plus 12 checker-clean unchanged
controls. Rust Book passages remain an untouched whole-family holdout.
`validate_sources.py` checks source hashes, passage provenance, split separation,
and selection counts. Source rights and notices are recorded in the manifest;
preserve them with any redistribution.

## Local synthesis

`queue.py` yields until the existing DFlash2 collection and controlled timing
finish, then starts `run_local.py` through Dibs with `cpu` and `port-8765`.
The runner verifies the pinned local Bonsai2 GGUF and llama-server binaries by
SHA256. It initially asks for 12 passages, four from each training source
family. The remaining 24 require a decision after inspecting those results.

`synthesize.py` sends one source passage at a time. It masks literal code,
options, names, and numbers, then rejects missing or reordered markers, altered
numbers/options, altered negation/modals/conditions, and checker failures. It
keeps raw responses and rejection reasons. Accepted outputs are still only
`teacher_candidate_owner_unverified`: deterministic checks cannot certify that
all technical meaning survived.
`review_packet.py` renders the first 12 sources, rewrites, and rejection reasons
to `data/out/doc-teacher-pilot-v1/first-12-review.md` after the run.

## Review gate

Inspect all 12 source/target pairs and rejection traces. Specifically check
agent/action, scope, conditions, negation, modal force, quantities, references,
and added claims against the source. Record a semantic verdict per pair before
running the remaining 24 or using any generated pairs for training. Keep the
Rust family untouched for later evaluation.

Run the local deterministic checks with:

```sh
python data/doc_pilot/validate_sources.py
python -m unittest discover -s data/doc_pilot -p 'test_*.py'
```

## Version 2: full STE context

v1 accepted 2 of the first 12 passages, and both changed meaning (see
`data/out/doc-teacher-pilot-v1/first-12-assistant-review.json`). v2 runs the
same 12 inputs with the same model, binary, seed and temperature. v1 code and
outputs are unchanged.

- `ste_context.py` builds a system prompt of about 27,300 tokens: task and modal
  policy, 9 worked examples, all 66 rule summaries, the approved words with
  forms, the software word list, and every unapproved word with its dictionary
  alternatives. The prompt is AGPL-derived, so it is written only to ignored
  `data/out/doc-teacher-pilot-v2/system-prompt.txt`. The worked examples come
  from unselected candidate passages; a test makes sure they are not inputs,
  controls or holdout passages, and that every example passes the v2 gates.
  Do not select the example source IDs (`context-manifest.json`) as teacher
  inputs later.
- Each user turn lists the checker findings for that source, with dictionary
  alternatives. Retry feedback names the flagged words and the failed gates in
  plain words. Up to 3 attempts.
- `gates_v2.py` counts what modal, negation and condition words do instead of
  comparing the words themselves: `may` is satisfied by `can`/`possible`,
  `not possible` by `cannot`, `whether` by `if`; `should` by `must` or by
  `we recommend`. A DESC target must not gain command sentences. The `should`
  policy (advice -> "we recommend that you", requirement -> "must") is the
  assistant's default, pending owner review.
- `synthesize_v2.py` evaluates the system prompt once, saves it as a
  llama-server slot (`--slot-save-path`), and restores the slot before every
  request, so each request evaluates only a few hundred new tokens. The chat
  template renders the system turn differently with thinking on, so the cached
  prefix is rendered with the same `enable_thinking: false` as the requests;
  without that, nothing is reused. `run_local_v2.py` uses a 40,960-token context
  and `--cache-ram 0`.

```sh
dibs run cpu port-8765 -- python data/doc_pilot/run_local_v2.py --limit 12
python -m unittest discover -s data/doc_pilot -p 'test_*.py'
```

Run results for the first 12 (October 28): the gates accepted 7. Assistant review: 5 keep the
meaning and 2 drop qualifiers that the checker does not flag. Of the 5
rejections, 3 change meaning, 1 is ungrammatical, and 1 is a valid rewrite
that the gates reject ("nothing else" -> "only"). The full verdicts are in
`data/out/doc-teacher-pilot-v2/first-12-assistant-review.json`. Owner review is
still required before the remaining 24 or any training use.

## Version 3: v2-review fixes and a meaning check

Same 36 frozen inputs in the v2 order (the first 12 are the v2 passages), same
model, binary, server options, seed and temperature. The fixes were designed on
the first 12 only; the other 24 show whether they generalize.

- `gates_v3.py` keeps the v2 meaning classes and adds:
  - "nothing else", "no other" and "nothing but" count as "only", not as a
    negation;
  - a source word whose dictionary alternatives contain a negation ("lost" ->
    "not find") may become one explicit negation, or "cannot";
  - "once" may become "when";
  - a pasted dictionary alternative after a form of "be" ("is not find") is
    rejected.
- `dropped_words` lists source words that the checker accepts but the rewrite
  leaves out ("convenient", "reliable"). It is not a gate: on the v1/v2 attempts
  it also flagged correct paraphrases ("individual" -> "each"). The meaning
  check gets the list, and each record keeps it under `review_flags`.
- `ste_context_v3.py` edits the task text. A dictionary alternative is for one
  sense and is a base form, every qualifier keeps its meaning, and words in no
  list are accepted. It adds a MEANING CHECK task with 5 examples built from the
  v2 worked-example sources. A flagged -ing or -ed form of an approved or
  software word gets those forms in its hint, before the dictionary
  alternatives. For example, "matching" gets the software verb match, matches,
  matched.
- Retry feedback names the words that each failed gate counted. For an added
  negation, it quotes the sentence. It also says when a retry repeats an earlier
  attempt.
- `synthesize_v3.py` sends a rewrite that passes the gates to a MEANING CHECK
  request with the same cached system prompt. Reported differences fail the
  attempt and go back as feedback. The check is the same model, so it is a
  second look, not an independent judge.
- A rewrite with missing, repeated or reordered literal markers gets a retry
  with feedback that names the markers. v1 and v2 stopped without a retry.
- The first attempt is greedy. Retries sample at temperature 0.7 with seed
  42 + attempt - 1. At temperature 0 the teacher repeated a failed rewrite word
  for word in 5 of the 7 v2 passages that needed a retry, and on the first v3
  passage, even with specific feedback. The meaning check stays greedy.
- Two runs were stopped after their first passage and are kept under
  `aborted-run-1/` and `aborted-run-2/`: the marker error with no retry, then
  the repeated retries. The system prompt did not change, so the saved slot
  was reused.
- `review_packet.py --limit N` renders the first N results (a multiple of 3).
  The v1 and v2 packets render byte-identically.

```sh
dibs run cpu port-8765 -- python data/doc_pilot/run_local_v3.py --limit 36
PYTHONPATH=data:checker:tokenizer:data/doc_pilot python -m unittest discover -s data/doc_pilot -p 'test_*.py'
```

Run results for all 36 (October 29). Verdicts are in
`data/out/doc-teacher-pilot-v3/all-36-assistant-review.json`, and the packet is
`first-36-review.md`. One run gives two acceptance rules. Up to the first
attempt that passes the gates, only gate feedback has been sent, so that
attempt is what a run without the meaning check would accept.

| Same 12 passages | v1 | v2 | v3 gates only | v3 with meaning check |
|---|---|---|---|---|
| Accepted | 2 | 7 | 10 | 6 |
| Meaning kept (assistant review) | 0 | 5 | 7 | 6 |

On the 24 new passages, gates alone accepted 20 and 14 keep the meaning; with
the meaning check, 7 were accepted and all 7 keep it. Over all 36, the meaning
check flagged all 9 accepted meaning changes, but it also flagged 11 of 21 good
rewrites, and 9 of those were lost. Remaining gate gaps: "advisable" is not an
advice word; splitting a sentence repeats "can"; "should be seen by" in the
sense of permission. The teacher still drops literal markers (2 passages) and
drops units and certainty words ("in bytes per second", "guaranteed").

## Version 4: v3-review fixes and 96 new passages

`select_sources_v4.py` runs the v1 parsers and segmenter on the same frozen,
hash-checked source files and selects 96 passages that no earlier pilot used
as an input, a control or a prompt example: 36 curl (at most 4 from each option
file), 36 PostgreSQL (18 from each file) and 24 Python logging. The selection
rule is the v1 rule (checker-flagged, 15-55 words, self-contained, SHA order).
The 36 v1 inputs follow in the v2 order. The fixes were designed on those 36,
so the 96 new passages measure how the fixes generalize. They come from the
same documents, not new ones. Model, temperatures, seeds and attempts are the
v3 ones. v1-v3 ran on the CPU; v4 runs on the Radeon GPU with the PrismML
Vulkan build that `utilities/dflash-compare` in openflowlm-next validated
(full offload, `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`), so a v3/v4 difference
can also come from the backend. The ternary DFlash drafter is not loaded: slot
restore reloads only the target context, not the drafter's state for the cached
system prompt.

- `gates_v4.py` adds to the v3 gates:
  - "advisable", "preferable" and "it is best to" count as advice, and
    "require" as obligation (its dictionary alternative is "necessary");
  - "allow", "permit" and "let" count as possibility ("is allowed to run" ->
    "can run");
  - a divided sentence can repeat each modal word of the source once;
  - hedge ("presumably", "likely"), certainty ("guaranteed", "always") and
    frequency ("usually", "often") words are meaning classes. "possibly" can
    stand for a hedge without counting as an added possibility;
  - each unit word ("bytes", "seconds") must stay;
  - literal markers can change order. `restore` replaces each marker by its
    number, so only a missing, repeated or extra marker gets a retry. Its
    feedback quotes the input words around a missing marker.
- `ste_context_v4.py` edits the task text:
  - a word that the checker findings do not name is accepted and stays;
  - a list of what v3 dropped (units, hedges, evaluative adjectives,
    restrictive clauses, the scope of "not ... any");
  - "it is advisable to" and "is allowed to" in MODAL WORDS.
  Three invented worked examples (in no source file) show these kept.
- The MEANING CHECK asks only for differences that change what a reader knows
  or must do. It lists the v3 false alarms as kept meaning (proper -> correct,
  "etc." -> "and other", dropped emphasis words). Each difference has the form
  "'input words' -> 'rewrite words': what changes". Five invented examples join
  the five v3 ones.
- A flagged "even", "just" or "already" gets a hint for the adverb sense, which
  the dictionary alternatives do not cover.
- Each record keeps `gates_only`, the first attempt that passed the gates, so
  both acceptance rules are in the record. `review_packet_v4.py` shows both.

```sh
PYTHONPATH=data:checker:tokenizer:data/doc_pilot python data/doc_pilot/select_sources_v4.py
dibs run gpu port-8765 -- python data/doc_pilot/run_local_v4.py --limit 132
```

### Version 4 results (October 30)

The GPU run took 65 minutes for 132 passages. The assistant read every
accepted rewrite against its source (`data/out/doc-teacher-pilot-v4/all-132-assistant-review.json`;
not owner-verified):

| | Gates only: accepted | kept the meaning | With the meaning check: accepted | kept the meaning |
|---|---|---|---|---|
| v3, the 36 v1 inputs | 30 | 21 | 13 | 13 |
| v4, the 36 v1 inputs | 31 | 26 | 23 | 22 |
| v4, 96 new passages | 69 | 54 | 50 | 45 |

The meaning check on the first rewrite that passed the gates:

| | Meaning changes flagged | Good rewrites flagged |
|---|---|---|
| v3, the 36 v1 inputs | 9 of 9 | 11 of 21 |
| v4, the 36 v1 inputs | 5 of 5 | 7 of 26 |
| v4, 96 new passages | 10 of 15 | 12 of 54 |

- On the 36 inputs it was designed on, v4 fixes most of what v3 lost (units,
  hedges, "guaranteed", "convenient and reliable") and the check flags fewer
  good rewrites, so the with-check rule keeps 22 instead of 13.
- On the new passages, 5 of the 15 meaning changes pass the check. Three change
  the agent: who selects the credentials (curl, not you), who does not
  overwrite the file (curl, not the server), which host does not get the
  credentials. The others turn "either A, B or C" into "A. It is also B. It is
  also C." and "in the same way" into "the same items". The final v1 rewrite
  of n=132 ("the task that the groups run") also changes the meaning.
- 13 good rewrites are lost to the check. Its false alarms are small word
  changes ("alleviates" -> "reduces"), the "should" -> "must" that the check
  text lists as kept, and marker numbers that v4 lets move (the check sees
  `[[L0]]` and `[[L1]]` change places).
- An unparsed check rejects the rewrite without another attempt (n=106 and
  n=124).
- 32 passages never pass the gates: 17 unapproved words that the teacher does
  not replace (R1.1), 11 modal changes (mostly "should", "it is a good idea",
  "might ... may"), and condition, negation and marker changes.
- Four kept rewrites have an English error ("the more data pieces", "In
  practice, but", "tries to work this procedure", a fragment).

## Granite 4.2 3B as an independent meaning check (October 29)

`judge_npu.py` asks Granite 4.2 3B (Q4_1, with its Uno drafter, served by
`oflm serve --speculative` from OpenFlowLM-Next PR #209 on the NPU) for the
meaning check on the 33 v3 rewrites with an assistant verdict (9 change the
meaning, 24 keep it). The judge sees a short STE note and the MEANING CHECK
text, not the word lists. Results are in `data/out/doc-judge-granite-v1/`.

| Same 33 rewrites | Meaning changes flagged | Good rewrites flagged |
|---|---|---|
| Bonsai 2 self-check (v3) | 9 of 9 | 11 of 24 |
| Granite, v3 check text | 9 of 9 | 24 of 24 |
| Granite, v4 check text | 4 of 9 | 10 of 24 |

Malformed JSON (8 and 9 of 33 replies) is counted as a flag here, because each
such reply listed differences. With the v3 text Granite flags everything; with
the v4 text it misses "guaranteed", "can be anything you want" and a dropped
restrictive clause. It often reports the changes that the check text lists as
kept ("should" -> "must"). It is not usable as the meaning check. Speed on a
busy box: plain decode 14.3 tok/s, speculative 15.3 (outputs identical),
bfp16 prefill 534 tok/s against 113 on the default route.

## K2-Horizon-7B as an independent meaning check (October 29)

The same 33 rewrites, judged by K2-Horizon-7B (Q4NX, PR #209 build and PR #209
K2 kernels on the NPU) with `--reasoning-effort low --max-tokens 3072
--repetition-penalty 1.1`. K2 always reasons first, and greedy decoding without
a penalty loops until the token limit; the penalty turns Uno speculation off
(it applies only to penalty-free greedy requests), so decode ran at about
4.5 tok/s. Results are in `data/out/doc-judge-k2-v1/`.

| Same 33 rewrites | Meaning changes flagged | Good rewrites flagged | Ran out of tokens |
|---|---|---|---|
| Bonsai 2 self-check (v3) | 9 of 9 | 11 of 24 | 0 |
| K2, v3 check text | 8 of 9 | 9 of 24 | 2 |
| K2, v4 check text | 6 of 9 | 6 of 24 | 1 |
| Bonsai and K2 (v3 text) both flag | 8 of 9 | 6 of 24 | |
| Same, a reply that ran out counts as a flag | 9 of 9 | 7 of 24 | |

With the v3 text, the one missed change (n=36) is a reply that reasoned to
the 3072-token limit and never answered; every change that K2 answered for
was caught. With the v4 text it returns `{"differences": []}` for n=10 and
n=24. Its false alarms are mostly small word changes (several -> some,
complies with -> uses, appear -> show, a dropped "already", "even" or
"just"), and both arms flag "should" -> "must", which the check text lists as
kept. A check took 108 s on average (v3 text); the 3 replies of 66 that ran
out took about 13 minutes each. As a second opinion that must agree with
Bonsai's flag before a rewrite is counted as changed, K2 keeps every meaning
change in this set and cuts the false alarms from 11 to 7. The set is small
and the v3 check text was written on earlier versions of these passages, so
this needs a fresh labeled set before it decides anything.

## Version 5: the same method, faster (October 30)

v4 took 30 s a passage: every request carried the 30k-token v4 prompt, one
request at a time, on a 27B dense model. v5 keeps the v4 gates, hints,
feedback, attempts and meaning check, and changes only how they run:

- `synthesize_v5.py` runs N passages at once, one llama-server slot each, and
  restores the cached system prefix into its own slot before each request
  (servers run with `--kv-unified`: a slot file loads only into a server with the
  same KV stream count). `--backend openai` drops the slots for a server with
  its own prefix cache (vLLM); `--inputs` takes any passage file.
- `ste_context_v5.py` has shorter prompts. Half the v4 prompt is the unapproved
  word list, which the checker findings already give for each passage's words.
  `core` (12k tokens) keeps the rule summaries and the approved and software
  lists, `tiny` (3k) only the task and the examples; both get their own 2k-token
  meaning-check prefix (the STE note, the check text and its examples).
  `traps` is tiny plus MEANING TRAPS (below).
- `judge_eval.py` scores a meaning-check judge on every rewrite with an
  assistant verdict (`data/out/doc-judge-eval/labeled.jsonl`).
- `build_pool_v5.py` builds a pool for a scaled run from the whole
  documentation trees at the v1 commits: 21,380 checker-flagged passages
  (PostgreSQL 12,492, Python 5,301, curl 3,587) after the pilot passages are
  removed. The Rust Book is not read.

Speed and quality on passages 1-48 (all new in v4), Radeon 8060S, PrismML
llama.cpp Vulkan build. "Kept" is the assistant's verdict, not the owner's:

| Teacher, prompt, slots | Passages/hour | Gates only: accepted / kept | With the check: accepted / kept |
|---|---|---|---|
| v4: Bonsai 2 27B, full (30k), 1 | 122 | 35 / 27 | 25 / 21 |
| Bonsai 2 27B, tiny, 8 | 270 | 39 / 29 | 31 / 29 |
| Bonsai 2 27B, tiny, 8, integer dot on | 292 | 38 / 29 | 30 / 28 |
| Qwen3.8-27B Q4_K_M, tiny, 8 | 266 | 37 / 28 | 32 / 26 |
| Qwen3.6-35B-A3B Q4_K_XL, tiny, 8 | 726 | 36 / 28 | 17 / 15 |
| Qwen3.6-35B-A3B Q4_K_XL, core, 8 | 570 | 37 / 31 | 18 / 17 |
| gpt-oss-20b MXFP4, tiny, 8 | 143 | 35 / not reviewed | 21 / not reviewed |

- The short prompt costs nothing in quality: Bonsai with the tiny prompt passes
  more passages and keeps more meaning than with the 30k-token one.
- Extra slots barely help on this GPU: these hybrid (Gated DeltaNet) models run
  about as many tokens a second with 8 slots as with one (Bonsai about 68 prompt
  and 8 output tokens a second in total; Qwen3.6-35B-A3B about 170 and 35).
  `-b 2048` or 16 slots made Qwen3.6 slower (520 and 478 an hour). The limit is
  the compute of each token: the 27B dense models prefill about 200 tokens a
  second cold, the 3B-active MoE about 650.
- gpt-oss-20b reasons before each answer (about 380 tokens a request at low
  effort) and does not reuse a restored slot (sliding-window layers, no
  `--swa-full`), so it is the slowest.
- Qwen3.6-35B-A3B is fast but its meaning check flags half of the good rewrites,
  so the with-check rule keeps few of them.

Meaning-check judges on the 189 labeled rewrites that all of them saw (41 change
the meaning, 148 keep it), with the short meaning-check prompt:

| Judge | Changes flagged | Good rewrites flagged |
|---|---|---|
| Bonsai 2 27B | 26 of 41 | 17 of 148 |
| Qwen3.8-27B Q4_K_M | 26 of 41 | 16 of 148 |
| Qwen3.6-35B-A3B (no thinking) | 36 of 41 | 76 of 148 |
| gpt-oss-20b (low effort) | 33 of 41 | 65 of 148 |

Bonsai 2 27B is a ternary Qwen3.8-27B (its model card names the base), and the
two judge alike: either one, or both together, gives the same result (both
must flag: 25 and 10; either flags: 27 and 23). Qwen3.6 and gpt-oss find more
changes only by flagging almost half the good rewrites. The v4 check (Bonsai
with the 30k prompt) flagged 19 of 80 good rewrites; with the short prompt
Bonsai flags 17 of 148. A Qwen3.6 judge with thinking wrote 600 to 1,900 tokens
for each check and was stopped: too slow to use.

MEANING TRAPS (`--variant traps`) adds to the tiny prompt the meaning changes
that passed every gate and the meaning check in the v4 and v5 reviews: a
changed agent, a which-clause made into a rule, a comparison turned around
("least recently" -> "last"), one-of made into all-of, a lost "Since version",
a changed object ("the requested page" -> "the request"), an attempt made into
a fact. Its examples are invented; the meaning check gets the same list.
Qwen3.8-27B Q4_K_M, 8 slots:

| Passages | Prompt | Gates only: accepted / kept | With the check: accepted / kept |
|---|---|---|---|
| 48 fresh pool passages (no pilot saw them) | tiny | 41 / 30 | 36 / 29 |
| 48 fresh pool passages | traps | 42 / 30 | 38 / 32 |
| 48 fresh pool passages, Bonsai 2 27B | traps | 38 / 28 | 31 / 29 |
| 48 fresh pool passages, Qwen3.6-35B-A3B | traps | 40 / 32 | 16 / 16 |
| v4's 96 new passages | traps | 76 / 67 | 63 / 61 |
| v4's 36 v1 inputs | traps | 34 / 31 | 31 / 31 |

For comparison, v4 had 69 / 54 and 50 / 45 on the 96, 31 / 26 and 23 / 22 on
the 36, and v3 30 / 21 and 13 / 13 on the 36. The traps were written from the
errors in these 132 passages, so the 132 rows overstate what they do: the fresh
passages are the honest measure, and there the gain is small (3 more kept
rewrites accepted, 84% instead of 81% of the accepted ones keep the meaning).
On the fresh passages the remaining changes are mostly word senses ("final
function" -> "last function", "more efficient" -> "more satisfactory"), a
qualifier that became a condition ("especially if" -> "if"), and planned work
written as done.

### Scaling: the pool on a pod for under $5

Measured per passage with Qwen3.8-27B and the traps prompt (132 passages): 3.0
requests, 893 new prompt tokens and 127 output tokens over a cached 3k-token
prefix. The pod plan (`pod/doc_teacher_run.sh`, packed by `pack_pod_v5.py`):
Qwen/Qwen3.8-27B-FP8 (the base of Bonsai 2 27B, Apache-2.0) on vLLM with
prefix caching, 128 requests at once, the v5 runner with `--backend openai`,
the traps prompt and the in-loop check, over the 21,380-passage pool. The
tarball carries the code, the modules that are not on GitHub main, the tested
lexicon files and the pool, and the script skips `phase_env` (no llama.cpp
build, no lexicon fetch).

| GPU (RunPod, secure) | $/h | Estimated passages/hour | 21,380 passages |
|---|---|---|---|
| L40S 48 GB | 1.09 | 8,000-20,000 | 1.1-2.7 h, $1.2-2.9 |
| H100 PCIe 80 GB | 2.89 | 15,000-40,000 | 0.5-1.4 h, $1.5-4.1 |

The range is an estimate until the smoke run measures it. Its anchor is a
published vLLM run of Qwen/Qwen3.8-27B-FP8 on a DGX Spark, a box with about this
machine's memory bandwidth: 16 requests at once gave about 104 output and 209
total tokens a second (1,000 in, 1,000 out), and about 3,500 prompt tokens a
second prefilled (8,000 in, 16 requests, 37 s to the first token)
(forums.developer.nvidia.com, "Qwen3.8-27B on DGX Spark using vllm"). The
L40S has about 3x that bandwidth and about the same FP8 compute; with 128
requests at once that is roughly 1,500-2,500 output and 3,500-7,000 prompt
tokens a second, or 0.14-0.34 s for a passage. Setup (vLLM install, a 28 GB
download, the server start) adds about 25 minutes. The same model in this
machine's llama.cpp Vulkan build makes about 10 output tokens a second with 8
slots, so the engine, not the hardware, is most of the local gap. Run the smoke (`BUDGET=1 N=600`, about 25 minutes, under $1), read
its passages-per-hour line, then the pool with `BUDGET=4`: common.sh stops at
85% of BUDGET, the outputs survive, and the rest of the pool runs locally for
free (`synthesize_v5.py --inputs pool.jsonl`, about 260 passages an hour with
Qwen3.8-27B Q4_K_M on this machine, 3.5 days for the whole pool). Expect about
79% of the passages accepted with the check and about 84% of those keeping the
meaning (the fresh-passage A/B), so about 17,000 candidates for review, not gold.

### Many passages in one request (OpenRouter)

`batch_openrouter_v5.py` sends up to `--batch` passages in one request (each
with its id, mode, checker findings and masked text, after the traps prompt
and a batch note) and gates every rewrite. It has no retries or meaning check:
it measures first attempts. Nemotron 3 Ultra 550B-A55B (free tier) took all 48
fresh passages in one request in 7 minutes (12,259 prompt tokens; 22,168
output tokens, 19,063 of them reasoning at low effort):

| First attempt only, 48 fresh passages | Pass the gates | Keep the meaning |
|---|---|---|
| Nemotron 3 Ultra, one request of 48 | 26 | 22 |
| Laguna XS 2.1, one request of 48 (32 s, no reasoning) | 28 | 20, 11 of the 28 not grammatical English |
| Qwen3.6-35B-A3B, traps, local | 24 | 20 |
| Bonsai 2 27B, traps, local | 22 | 19 |
| Qwen3.8-27B, tiny, local | 21 | 15 |
| Qwen3.8-27B, traps, local | 18 | 13 |

So a batch of 48 does not hurt a strong model. Its rewrites change less (it
keeps passives and most of the wording), and it makes the same word-sense
errors ("final function" -> "last function"). Laguna XS 2.1 is fast and passes the
gates most often, but by word-for-word dictionary swaps ("not can to continue",
"will be check", "a lot more not easy"): the checker does not catch broken
grammar, so it is not a usable teacher. Laguna S 2.1 failed with a reset
connection, Laguna XS once with 429, and Inkling with 403: the free tier is not
dependable for bulk.

On the 131 judged fresh-passage rewrites (32 change the meaning), the meaning
check catches 21 with Bonsai and 18 with Qwen3.8, and flags 14 and 9 of the 99
good ones. Both miss the word-sense errors and planned work written as done:
a stronger judge is the next quality lever, not more of these.

### Paid models, the batched pipeline and the production run (October 30)

`pipeline_v5.py` is the v4 method with batches: each batch of 48 passages gets up
to 3 rounds of one rewrite request (a retry carries the previous rewrite and the
checks it failed) and one meaning-check request for the rewrites that pass the
gates, by a model of another family. `openrouter_v5.py` sends the batches;
`judge_or_eval_v5.py` scores an OpenRouter judge on the labeled rewrites;
`local_rejudge_v5.py` adds a second, local check as a review flag.

First attempts on the 48 fresh passages, one request each (`batch_openrouter_v5.py`):

| Model | Pass the gates | Keep the meaning | Cost of the request |
|---|---|---|---|
| Qwen3.8-27B | 21 | 20 | $0.0082 |
| Nemotron 3.5 Super | 22 | 20 | $0.0018 |
| DeepSeek V4.1 Flash | 21 | 19 | $0.0019 |
| DeepSeek V4 Flash | 25 | 19 | (not recorded) |
| Mistral Small 2603 | 20 | 17 | $0.0036 |

Judges on all 481 labeled rewrites (105 change the meaning, 376 keep it), the
traps judge prompt, 48 checks a request:

| Judge | Changes flagged | Good rewrites flagged | Of the unflagged, keep the meaning | Cost |
|---|---|---|---|---|
| DeepSeek V4.1 Flash | 85 | 81 | 93.7% | $0.018 |
| Qwen3.8-27B | 50 | 42 | 85.9% | $0.045 |
| Nemotron 3.5 Super | 48 | 50 | 85.1% | $0.014 |
| Mistral Small 2603 | 68 | 142 (48 unparsed) | | $0.018 |
| DeepSeek or Qwen3.8 flags | 90 | 92 | 95.0% | |

The whole pipeline on the 48 fresh passages, DeepSeek V4.1 Flash as the judge:

| Rewriter | Accepted | Keep the meaning | Cost |
|---|---|---|---|
| Qwen3.8-27B | 30 | 30 | $0.018 |
| Nemotron 3.5 Super | 19 | 19 | $0.012 |

With Nemotron 3 Ultra (free tier, reasoning) as the rewriter the same pilot
accepted 31, all keeping the meaning, in 25 minutes for the 48.

gpt-oss-120b through `:nitro` (Cerebras) is the fastest model tried, and the
weakest rewriter:
- With 48 passages a request and low reasoning it replies `{"rewrites": []}`
  after 39 reasoning tokens. Cerebras stops every reply at 8,192 output tokens,
  whatever `max_tokens` asks, so the JSON is cut off at 48 a request with medium
  reasoning and at 12 with high.
- At 12 a request with medium reasoning the pipeline accepted 30 of the 48 in
  18 seconds, for $0.071 (four times Qwen3.8). Of the 30, 7 do not keep the
  meaning or are not usable English ('the called abstract interface', 'a lot
  more not easy', 'more efficient' -> 'better', the aggregate's 'final function'
  -> 'last function', 'for a transfer that is, because of rlimit, cannot
  continue'), and most of the others hardly change the source.
- As a judge (low reasoning, 12 checks a request, all 481 labeled rewrites in
  7 seconds, $0.094): 67 of 105 meaning changes flagged, 88 of 376 good rewrites
  flagged, 19 checks missing from the replies. On its own it is below DeepSeek
  V4.1 Flash, but it is a third family and the quickest second check: of the 20
  changes that DeepSeek misses it flags 7 (Qwen3.8 5, Nemotron 3 Ultra 5, Bonsai
  with the traps prompt 4 of 8 on a smaller set), and it also flags 45 of the
  295 good rewrites that DeepSeek passes.

So Qwen3.8-27B stays the rewriter, and gpt-oss-120b is the second meaning
check of the production set (`or_rejudge_v5.py`, a review flag in
`second_judge_or.jsonl`).

Production (`data/out/doc-teacher-v5/prod-*`):
- Qwen3.8-27B with the DeepSeek judge, pool passages 1-12,000, capped at $4.40:
  10,502 passages done in 27 minutes before the cap, 6,113 accepted (58%). An
  audit of 40 accepted rewrites drawn at random: all 40 keep the meaning (3
  with a slight shift: 'via' -> 'It uses', 'a certain amount' -> 'a specified
  quantity', 'treated exactly the same as' -> 'the same as'). The 1,498 passages
  the cap stopped were taken out of `rejected.jsonl` (the old file is kept as
  `rejected.with-cap-failures.jsonl`).
- Nemotron 3 Ultra (free tier) with the DeepSeek judge, from passage 12,001 and
  on the 1,498 above, at about 970 passages an hour: stopped after 90 minutes,
  when a second key made the paid model affordable for the rest. 1,458 + 337
  passages done, 1,080 + 230 accepted (73%). An audit of 40 accepted rewrites
  drawn at random: all 40 keep the meaning (3 with a slight shift: 'as well as
  a comparison with equivalent btree indexes' -> 'that also compares equivalent
  btree indexes', 'freed later than they would be in the default build' ->
  'freed subsequently', and the test-server option rewritten as 'You do this
  when you verify').
- Qwen3.8-27B with the DeepSeek judge on all the passages left
  (`prod-qwen38-rest`, `--skip-runs` the two runs above): 9,083 passages in 17.5
  minutes, 5,340 accepted (59%), $3.65. An audit of 40 accepted rewrites drawn
  at random: 39 keep the meaning; one turns 'should be NIL' into 'must be NIL'
  (the open "should" question), 5 have a slight shift ('once per outer row' ->
  'when each outer row is processed', 'upper executor context' -> 'top', 'the
  backing device' -> 'the device that supports it', 'exact port number' ->
  'correct port number', 'may have removed' -> 'can remove') and one is not
  good English ('do the preceding example this procedure').
- Second meaning check: gpt-oss-120b on every accepted rewrite (12,763 in about
  3 minutes, $2.46), `second_judge_or.jsonl` in each run. It flags 1,959 (15%).
  In the two audits it flagged 10 of the 80: 5 at a real slight shift or the
  should/must change, 5 at none. Bonsai on this machine checked the first
  1,841 of the Qwen3.8 slice (`second_judge.jsonl`, 148 flagged) before it was
  stopped for the faster check.

All 21,380 pool passages are processed. `merge_v5.py` writes the candidate set
to `data/out/doc-teacher-v5/final/accepted.jsonl`: 12,763 pairs (Qwen3.8-27B
11,453, Nemotron 3 Ultra 1,310; PostgreSQL 7,426, Python 3,237, curl 2,100;
DESC 12,585, PROC 178), each with `review_flags.second_judge` and, where it ran,
`review_flags.local_judge`. Cost of the production runs and the second check:
$4.44 + $0.06 + $3.65 + $2.46 = $10.61, of about $11.1 spent on OpenRouter in
all (two keys, the tests included, by the cost each reply reports).

Repair pass for the 1,959 rewrites that the second check flagged
(`build_repair_v5.py`, then `pipeline_v5.py --judge2`): each starts as a retry
of its rewrite with gpt-oss-120b's differences as the feedback, Qwen3.8-27B
rewrites, up to 2 rounds, and a repair is accepted only when it passes the gates,
DeepSeek and gpt-oss. 4 minutes, $0.96. 830 were accepted, but 273 of them are
the same text checked clean a second time, which does not clear the flag (the
first of the smoke test had dropped 'already' and passed the second time).
`merge_v5.py --repairs repair-flagged` applies the other 556, and leaves 1,403
rewrites flagged. An audit of 40 applied repairs drawn at random, against the
rewrites they replace: 25 better ('may' kept as 'it is possible that', 'by
default' and 'etc.' put back, 'should' kept as 'we recommend', 'expose' no
longer 'get'), 12 neither better nor worse, 3 worse: 'is possibly wanting to
extend', an added 'This is also true despite the previous statement', and one
that put two literals into the wrong words ('utf-`U+D800`-be' and '(32 - 32)').

That last one shows a gap in the judge message: `judge_message_v4` masks the
input and the rewrite separately, so the markers of the rewrite are numbered in
their own order and a swap of two literals looks the same as no swap. The
flagged original had the literals right in a different order, the judge's
flag on it came from the renumbering, and the repair followed the flag.
`merge_v5.py` now refuses a repair that puts a literal inside a word that it is
not inside in the source; only this one was refused. In the final set no other
accepted rewrite moves a literal into another word, 23 change the order of the
literals (of the 7 I read, 6 are correct sentence changes and one moves
'(i.e.)' to the end as '(i.e)') and 14 change a
literal's edges (mostly a URL that a sentence end now follows; 4 of the 5 I read
have damage: '),).', '(i.e)', '..'). Future runs should show the judge the
rewrite with the rewriter's own markers, and gate on a literal's edges. The pairs are teacher candidates
(`teacher_candidate_owner_unverified`), not gold.

### Fitting the student model (October 30)

`model_gate_v5.py` asks the model's own tokenizer whether it can encode a pair, and
names every word that breaks the contract. The vocabulary of the v2 and v4
checkpoints is `data/out/fixes-v1/vocabulary` (2,575 output words, with the seven
software forms of October 25), not `tokenizer-joint-v1`. Of the 11,360 unflagged
pairs, 5,201 encode. The others, by what would make them encode:

| Pairs | What breaks the contract | Fix |
|---|---|---|
| 1,171 | Only tokenizer gaps: contractions, "been", "being" and -ing forms are not input words, so the tokenizer copies them as names and a rewrite cannot remove them; a copied word (a table name, "user") may not be used more often than in the input | Make those forms input-only words, allow a copied word to repeat |
| 895 | Only words the rewrite added that the model cannot write: "happens", "ensures", "specific", "means", "two", "specify" | Repair pass with the gate's feedback |
| 2,467 | Only input words kept that the model cannot write: some should have been rewritten ("stored", "provided", "specific"), some are technical terms that the dictionary rejects in another sense ("thread", "foreign" key, "backup", "zero", "warning", "scan") | Repair the first kind; the second kind needs owner-approved lines in `tech_words.txt` |
| 414 | Only a dropped word that is not in the dictionary ("like", "actually", "whereas", "simply", "finally", "etc"): the handoff keeps such words as they are, so the tokenizer copies them | An owner decision |
| 1,205 | Two or more of the above | |
| 7 | The input itself does not encode (too many copied spans, too long) | None |

Also on October 30: the tokenizer reads a sentence-initial "A" before a lowercase
word as the article (it was a copied variable name, which blocked 163 rewrites),
and the checker no longer takes an -s verb after "this" for a noun ("This
happens", "This ensures" are now errors; calibration: STE side unchanged, non-STE
caught 1,498 -> 1,499 strict and 1,321 -> 1,322 with the software words). After
"that" it would also be right, but a worked example of the v4 prompt keeps "a
program that rotates log files", so "that" is unchanged. The model gate catches
these words in any case. The gate's feedback finds the dictionary alternatives
of more word forms than the checker does (stored -> store, modified -> modify,
creating -> create); the same wider matching in the checker would flag 18
dictionary STE examples ('housing', 'fitting', 'coupling'), so the checker keeps
its narrow matching.

The owner approved the fixes the same day:
- Vocabulary `data/out/fixes-v2/vocabulary` (`model_tagger/extend_vocabulary.py`,
  from fixes-v1 and the v2 checkpoint): 37 new output words from 24 new lines of
  `tech_words.txt` (zero, foreign, scan, thread, backup, warning, switch, sampling,
  idle, blocking, timing, isolation, account, facility, capability, injection,
  transport, graph, volatile, product, enabled, disabled, handling, escape, with
  their forms), and 339 new input-only words: contractions, "been", "being", "am"
  and the -ing forms of approved and software verbs, except technical -ing nouns
  ("encoding", "logging", "parsing"), which stay copyable. Output 2,612 words,
  input 9,800 IDs. The checker's calibration with the software words moves from
  1,322 to 1,319 non-STE caught, as the list approves more words; strict is
  unchanged.
- A copied word may still not repeat. Allowing it would change the copy guard in
  the tokenizer, the rollout audit, the decoder and the guarded editor for about
  370 pairs; the repair pass asks the rewriter to keep the count instead.
- Pairs that drop a plain word the dictionary does not list ("like", "actually",
  possessives) are left out, as the handoff keeps such words: 1,494 pairs.
- Repair pass with the model gate (`build_repair_v5.py --reason model`, then
  `pipeline_v5.py --model-gate data/out/fixes-v2/vocabulary`, 3 rounds, both
  judges): 4,000 rewrites, 1,935 repaired, $1.92, 12.5 minutes. The gate's feedback
  names the dictionary alternatives, says that STE has no -ing verb forms (the
  rewriter otherwise replaced "acquiring" with "getting"), and asks for the
  input's capital letters on a copied word. An audit of 30 repairs drawn at random:
  all 30 keep the meaning; 3 shift slightly ('ask' -> 'tell', 'specific' ->
  'specified' twice) and one is not good English ('an alternative to import').

Of the 12,763 pairs, 7,794 are training-ready (second check clean, and they
encode with fixes-v2), 1,403 are flagged and 3,566 do not encode.

Trained on (October 30, `model_tagger/run_doc_v5.py`; see the tagger README, "local
v5"): 5,423 of the 7,794 pairs enter training (2,371 would need the loop to go on past
a checker-clean state). The trained model does not learn to rewrite documentation:
dev edit-tag accuracy 3.4% on these pairs, 0 of 679 documentation dev inputs resolved
by the guarded editor, 2 of 679 exact with raw decoding (as the v2 baseline). It
deletes the word to replace without writing the replacement.
