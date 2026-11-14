# MemDraft: a drafter that only looks things up

**Status, 2026-11-10: restarted.** The first attempt (a continuation cache in front of DFlash2) failed; its write-up is the second half of this file. The restart keeps the name and changes the question, the drafter and how it is measured. The drafter is built and runs in llama.cpp. What the target would accept is measured offline. **Tokens per second, the endpoint, has only pilot numbers so far: on Qwen3.5-9B, two-token drafts gave 1.24× on the GPU and 1.18× on CPU, and on CPU they beat llama.cpp's lookup drafters (0.99×). On coding prompts on CPU, a table with code in its text gave 1.37×. On 2026-11-12, with 39M tokens of the 9B's own code in the table and a reranker over the proposals, coding prompts ran 1.93× faster than plain decoding on the GPU (pilot, 12 prompts) and 1.51× on CPU. The clean run is still to do.**

**Question.** Does a drafter that runs no model make decoding faster in tokens per second than plain decoding, and than llama.cpp's own free drafters, on small and mid-size models?

The drafter proposes tokens from two lookups and nothing else:

- **A table built ahead of time** from text the target model wrote. For the last 1 to 6 tokens it holds the token the model most often chose next, and how often that guess was right on held-out text.
- **The current request.** The same lookup against the prompt and the tokens generated so far.

Each proposal has a confidence. The most confident one is drafted, and the chain goes on while its chance of being accepted in full stays above a threshold `tau`. A draft step takes about 2 µs.

## What changed from the first attempt

| | First attempt | Restart |
|---|---|---|
| Where the table comes from | Filled at run time from the request stream (28K tokens, 21K entries). Nothing was built ahead. | Built ahead from 13 to 20M tokens of the target's own output |
| Keys and drafts | Exact 4- or 8-token keys, drafts up to 16 tokens | 1- to 6-token keys with a confidence each, drafts of 1 to 4 tokens |
| When to draft | Fixed rules, then a learned ranker | While the chance of acceptance is worth the cost of verifying one more token |
| Compared against | DFlash2, which ran behind MemDraft in every arm | Plain decoding. The drafter stands alone. |
| Endpoint | Acceptance and tokens per step | Tokens per second |
| Hardware | CPU, 8 threads | GPU and CPU |

Two corrections to the record:

- **Nothing was taken from Qwen3.8-Flash-Next.** Its n-gram table holds embedding vectors, not next tokens. The comparison was an analogy in the first planning conversation (2026-10-25). The first attempt's table was its own, but it only ever held what the current run had seen.
- **"MemDraft did not beat prompt lookup" was never measured.** MemDraft never ran live without DFlash2 behind it. In tokens per step, MemDraft + DFlash2 (4.48) was ahead of lookup + DFlash2 (4.07). The 10.1 against 11.1 tok/s came from request logs, on 32 prompts picked afterwards for low CPU contention; it was not a controlled run.

## What a free draft is worth

A decode step that carries `n` draft tokens costs about `1 + beta * n` plain steps, where `beta` is what one more token in the batch costs. The lookup costs nothing, so a draft token is worth sending when its chance of being accepted is above roughly `beta`. Acceptance rate is therefore the wrong target: where `beta` is 0.06, a guess that is right one time in five still pays.

`beta` depends on the model and the backend, and it decides the result:

| Backend and model | One more draft token | Each accepted token, on top | A change of batch size | Source |
|---|---:|---:|---:|---|
| Vulkan (Radeon 8060S), Qwen3.5-9B Q4_K_M | 0.065 of a step (2.0 ms) | 0.02 (0.65 ms) | 0.185 (5.7 ms) | Pilot: GPU alone, CPU busy |
| CPU, 8 threads, Qwen3.5-9B Q4_K_M | about 0.13 of a step | | not measured | 8 prompts, arms taking turns, machine busy |

A plain step on the 9B is 30.8 ms through Vulkan and about 170 ms on CPU.

**CPU is not the hard case it first looked like.** The first version of this table had the 9B on CPU at about 0.85 per draft token, from one 40-token prompt on a machine at load 30 to 44. Measured again on 2026-11-10 at load 5 to 13, a step with two draft tokens cost 1.26 plain steps (median of 8 prompts, range 0.96 to 1.41). The server log shows why: a token inside a batch costs 8 to 15 ms, a step of one token 100 to 220 ms. Most of a step's cost on CPU does not depend on how many tokens are in the batch. When every core is busy that stops being true, so the CPU cost moves with the load.

llama.cpp (7d7f028) adds two costs to a drafted step that the simple formula leaves out. Both were found in this restart, and both hit short n-gram drafts hardest:

- **A rejected draft.** Qwen3.5, 3.6 and 3.8 keep recurrent state, which cannot be rolled back by deleting cache entries. For model drafters (MTP, DFlash2) llama.cpp keeps a snapshot of that state per draft token (`n_rs_seq`). With only n-gram drafters enabled it does not: it saves a checkpoint before every drafted step (50 MiB on a 4B model) and, when any draft token is rejected, restores it and decodes the accepted tokens a second time. `ngram-simple`, `ngram-mod`, `ngram-cache` and the first MemDraft all ran this way when no model drafter was enabled. The patch here asks for snapshots.
- **A changed batch size.** llama.cpp reuses its compute graph only when the batch has the same shape as the step before; otherwise it builds and allocates the graph again. Drafts of varying length change the shape at most steps (287 of 476 steps in one short run; plain decoding rebuilds none). On the 9B a rebuild costs 5.7 ms, as much as three draft tokens. `--spec-table-pad` drafts a fixed number of tokens at every step, so the shape never changes.

## Results so far

### Offline: what the target would accept

No timing and no model run. The data is 35M tokens that Qwen3.5-9B wrote for manifold-compression's healing shards (GSM8K, MATH, OpenThoughts and UltraChat prompts, thinking on and off, sampled at temperature 1.0). The 19.5M tokens in finished answers also carry the model's top choice at every position. Tables are counted on two shards. The third shard is held out: a quarter of its problems set the confidences and choose `tau`, and the rest (4.8M tokens) are the test. "Greedy" scores a draft token as right when it equals the model's top choice.

Predicted speed against plain decoding, for a cost of `beta` per draft token and nothing else (`ngram_study.py`, table from 13.1M tokens):

| Cost of one draft token (`beta`) | Table only | Request only | Both | Tokens per step, both |
|---:|---:|---:|---:|---:|
| 0.05 | 1.38× | 1.30× | **1.51×** | 1.70 |
| 0.10 | 1.29× | 1.22× | **1.38×** | 1.60 |
| 0.20 | 1.18× | 1.12× | **1.24×** | 1.49 |
| 0.30 | 1.12× | 1.08× | **1.15×** | 1.41 |
| 0.50 | | | 1.06× | 1.31 |
| 0.85 | | | 1.00× | 1.09 |

With free verification (`beta` 0) the ceiling is 1.75 tokens per step. By kind of text, both lookups at `beta` 0.10:

| | Plain chat | Chat, thinking | GSM8K | MATH | OpenThoughts |
|---|---:|---:|---:|---:|---:|
| Predicted speed | 1.14× | 1.25× | 1.45 to 1.54× | 1.50 to 1.55× | 1.43× |
| Tokens per step | 1.24 | 1.40 | 1.70 to 1.84 | 1.80 to 1.87 | 1.69 |

- **The two lookups add up.** The table carries plain chat, where the current request has little to copy. Math and reasoning text gain from both.
- **Chat is the weak domain:** 1.24 tokens per step against 1.7 to 1.9 on math. Only 4.3M of the table's tokens are chat.
- **Counting the text is enough.** A table counted from the tokens the model wrote (24.7M) matches one counted from its top choices on the finished half (13.1M): both predict 1.29× with the table alone at `beta` 0.10. Per token, top choices are worth a little under twice the text. No extra model pass is needed to build a table; the model's own output will do.
- **More text helps slowly.** Both lookups at `beta` 0.10: 1.31× from 0.8M tokens, 1.34× from 3.3M, 1.38× from 13.1M, so about 0.035 for every fourfold. On chat with the table alone: 1.11× from 0.25M chat tokens, 1.15× from 1.0M, 1.19× from 4.3M.
- **The table can be small.** Contexts seen once are useless:

| Keep contexts seen at least | Entries | Table file | Predicted speed, `beta` 0.10 |
|---|---:|---:|---:|
| once | 27.0M | 512 MiB | 1.381× |
| twice | 4.66M | 128 MiB | 1.379× |
| 3 times | 2.38M | 64 MiB | 1.375× |
| 5 times | 1.15M | 32 MiB | 1.371× |
| 10 times | 0.51M | 8 MiB | 1.361× |

- **A table does not carry far outside its text.** Counted only from the math sources, the table predicts 1.07× on chat, where a chat-only table predicts 1.19×. Counted only from chat, it predicts 1.12× on math, where the mixed table predicts 1.27 to 1.41×. "Works for everything" needs text from everything.
- **Sampling costs little.** Scored against the tokens the sampler wrote at temperature 1.0, the same table gives 1.35× against 1.38× for greedy.
- **The confidences mean what they say.** On held-out plain chat, guesses with confidence of 0.8 or more are right 88% of the time and exist at 6% of positions; at 0.5 or more, 48% right at 31% of positions. Typical right guesses are word endings (`Unsweet` then `ened`), list and table markup, and stock phrases (`case` then ` studies`).

### The drafter in llama.cpp

`patch_llama.py` adds `--spec-type ngram-table` to llama.cpp 7d7f028. The drafter itself is `memdraft.h`, about 250 lines with no llama.cpp types.

- **It matches the study code exactly.** On the 4.8M held-out tokens, `replay.cpp` and the numpy drafter count the same steps, drafted and accepted tokens in all 18 settings tried (`build_table.py --check`).
- **Its cost is about 2 µs per step:** 0.3 to 5 µs across settings, measured on a loaded machine with the 512 MiB table. A decode step takes 10 to 100 ms.
- **One table serves the Qwen 3.5 to 3.8 family.** The 9B, 27B, 35B-A3B and Bonsai-2 GGUFs on this machine all use the same 248,070 tokens.

### Live

**Pilot, 2026-11-09 21:52 to 21:58.** Qwen3.5-9B Q4_K_M on the Radeon 8060S through Vulkan, thinking off, temperature 0, 96 tokens per prompt, with the 64 MiB table counted from the 9B's shards (`run_pilot.sh`). No other process was on the GPU (`gpu_alone.py`), but the rest of the machine was busy (load 21 to 27), so these are pilot numbers, not the clean measurement. The prompts are new (`prompts/live-v1.jsonl`): chat, math, code, and editing a given text. The table's text has no code and no editing.

| Arm | Prompts | Tokens/s | Against plain | Draft tokens accepted |
|---|---:|---:|---:|---:|
| Plain decoding | 4 | 32.5 | 1.00× | |
| Table + request, `tau` 0.3, drafts of 0 to 4 tokens | 4 | 35.5 | 1.09× | 113 of 310 |
| Table + request, always 1 token | 8 | 38.3 | 1.18× | 174 of 579 |
| **Table + request, always 2 tokens** | 4, then 8 | 40.7, 40.2 | **1.25×, 1.24×** | 118 of 515, 226 of 1,055 |
| Table + request, always 3 tokens | 8 | 38.4 | 1.18× | 245 of 1,520 |

- **The 8-prompt arms have a borrowed baseline.** In that second run the GPU guard stopped the plain arm after 6 of 8 prompts, when another session's browser test opened the GPU. Those 6 ran at 32.1 to 32.5 tok/s, the same as the first run; the ratios use 32.4.
- **Two-token drafts by kind of prompt (8 prompts):** chat 1.09×, code 1.18×, editing 1.32×, math 1.41×.
- **Acceptance is not the measure.** The two-token arm accepts 21% of what it drafts and is the fastest.
- **Variable-length drafts lose their gain to graph rebuilds.** On the chat prompt they were slower than plain decoding (0.86×).
- **The cost model holds.** From the step costs above it predicts the three fixed-length arms within 4% (1.21×, 1.27×, 1.22× against 1.18×, 1.25×, 1.18×).
- **What the offline study predicts with these step costs** (`results/study-9b-measured-costs.json`): 1.34× on the held-out shard with two-token drafts. That is 1.09× on plain chat, 1.19× on chat with thinking, and 1.43 to 1.51× on GSM8K and MATH. If the runtime kept one graph per batch size, variable-length drafts would reach 1.44× (plain chat 1.17×).
- **Output** equalled plain decoding on 3 of 4 prompts in each arm of the first run; llama.cpp's speculative decoding is not bit-exact.

Before the pilot, CPU runs on the 4B Bonsai-2 only checked that the drafter works with the 9B's table (36% of its drafts accepted). A first Vulkan check at 21:03 ran beside a ROCm job whose dibs hold had vanished; its timings are discarded.

#### Against llama.cpp's lookup drafters ("prompt lookup")

`run_lookup.sh` runs the table drafter and llama.cpp's own free drafters on the same prompts. The stock drafters run as shipped, with their default settings:

| Drafter | What it does |
|---|---|
| `ngram-simple` | When the last 12 tokens occurred earlier in the request, drafts the up to 48 tokens that followed |
| `ngram-mod` | The same idea with a 24-token match and drafts of 48 to 64 tokens |
| `ngram-cache` | Counts 1- to 4-token contexts in the request, drafts up to 8 tokens while one continuation dominates. The classic lookup decoding. |

On Qwen3.5 all three pay for a checkpoint before every drafted step and a second decode after a rejection (see step costs); the table drafter uses snapshots.

**CPU, 2026-11-10 03:38 to 03:54.** Qwen3.5-9B Q4_K_M on 8 threads, 8 prompts (2 per kind), 96 tokens each, temperature 0, thinking off. For every prompt the three arms ran back to back, because CPU speed drifts within minutes on this machine. Load was 5 to 13 and a training job was on the GPU, so this is a pilot.

| Arm | Tokens/s | Against plain | Per prompt: median (range) | Draft tokens accepted | Tokens per step |
|---|---:|---:|---:|---:|---:|
| Plain decoding | 5.88 | 1.00× | | | 1.00 |
| **Table + request, always 2 tokens** | 6.95 | **1.18×** | 1.19× (0.83 to 1.51) | 220 of 1,066 | 1.40 |
| `ngram-cache` | 5.85 | 0.99× | 0.97× (0.74 to 1.27) | 73 of 474 | 1.11 |

- **The table drafter was faster than `ngram-cache` on all 8 prompts,** and faster than plain decoding on 7.
- **By kind of prompt** (table drafter, then `ngram-cache`): chat 1.00× and 0.85×, code 1.08× and 0.98×, editing 1.34× and 1.10×, math 1.36× and 1.08×.
- **`ngram-simple` and `ngram-mod` hardly fire on answers this short.** In a run just before (same prompts, 128 tokens, one arm after another), `ngram-simple` drafted on 3 of 8 prompts and had 2 of 139 draft tokens accepted; `ngram-mod` never drafted. In that run the table drafter reached 1.41 tokens per step, `ngram-cache` 1.15, the other two 1.00. Its timings are not used: plain decoding ranged from 4.5 to 8.2 tok/s between arms.
- **Chaining works but had nothing to add here.** With `--spec-type ngram-simple,ngram-table` the long-match drafter gets the first try and the table fills in elsewhere. On these prompts the long-match drafter added 139 draft tokens and 1 accepted one.
- **Limits.** Eight prompts, short answers, one model. Long answers that copy their input are where `ngram-simple` and `ngram-mod` should do best, and no prompt here gets that far in 96 to 128 tokens.

**GPU: not run yet.** The same comparison through Vulkan (`run_lookup.sh vk`) takes turns of at most 9 minutes in the breaks between other sessions' GPU jobs, and each turn measures plain decoding again as the baseline for the arms it runs. It was queued on 2026-11-10 and its one turn finished no arm; the queue ended with a reboot that afternoon and has not been started again.

**Offline, the table against the request alone.** With the 9B's measured GPU step costs on the held-out shard (`results/study-9b-measured-costs-request.json`): drafting from the current request only gives 1.18× (plain chat 1.03×, chat with thinking 1.09×, GSM8K and MATH 1.19 to 1.26×), from the table only 1.24× (plain chat 1.13×), from both 1.34×. This is the request lookup done with this drafter's confidences and cheap rollback, not a stock llama.cpp drafter.

### On code (2026-11-10)

Neither the table's text nor the offline study's held-out text has any code. `gen_code.py` had Qwen3.5-9B answer 800 coding prompts from OSS-Instruct (`ise-uiuc/Magicoder-OSS-Instruct-75K`: half Python, half eight other languages) at temperature 0 with thinking off, up to 1,024 tokens each: 433K tokens, on CPU. At temperature 0 the token written is the model's top choice, so the answers score as the shards do.

**Offline** (`code_study.py`, `results/code-study.json`). A prompt's hash decides its part: a quarter set the confidences and the drafting rule, a quarter are the test (156 finished answers, 68,040 tokens, 79% of them inside code blocks), and half are text for a table of the 9B's own code. The set also has a gpt-3.5 solution for each of its 75K prompts, which gives code by another writer: for the same prompts as the own-code table, and for every prompt outside the held-out ones. Predicted speed uses the 9B's measured GPU step costs.

| Table | Tokens in it | First guess right, table alone | With the request lookup | Predicted speed, with the request lookup |
|---|---:|---:|---:|---:|
| None | | | 36.9% | 1.24× |
| Shipped text (no code) | 19.5M | 21.2% | 42.7% | 1.34× |
| The 9B's own code, a quarter | 0.05M | 22.3% | 43.6% | 1.35× |
| The 9B's own code, half | 0.11M | 24.4% | 44.7% | 1.37× |
| The 9B's own code | 0.22M | 26.4% | 45.6% | 1.38× |
| gpt-3.5's code for the same prompts | 0.10M | 19.1% | 41.9% | 1.32× |
| gpt-3.5's code for all other prompts | 18.6M | 31.2% | 47.0% | 1.41× |
| Shipped + own | 19.8M | 25.5% | 45.3% | 1.38× |
| Shipped + gpt-3.5's | 38.1M | 31.8% | 47.3% | 1.42× |
| **Shipped + own + gpt-3.5's** | 38.3M | 32.8% | 48.0% | **1.43×** |

- **Code is not a weak domain.** With the shipped table, which has never seen code, the prediction is 1.34×, the same as on the held-out shard and far above plain chat (1.09×).
- **The request does most of it.** The in-request lookup alone gets 36.9% of first guesses right on code, against 12.6% on plain chat. Code repeats its own names.
- **The target's own code is worth far more per token than anyone else's.** 0.22M tokens of it beat the 19.5M-token shipped table. At about 0.1M tokens, the 9B's code gets 24.4% alone and gpt-3.5's code for twice as many prompts gets 19.1%. Each doubling of the 9B's code added 2 points alone and about 0.015× with the request lookup. How long that slope holds is not known: 0.22M tokens is all there is.
- **The prose around the code is the slow part.** With the largest table, the first guess is right at 52.2% of positions inside code blocks and 31.8% outside.
- **Python and the other languages are alike:** 1.42× and 1.45× with the largest table.
- **Keeping contexts seen at least 3 times costs little here too:** 1.425× for the largest table (128 MiB, `--export`), and 1.33× for the shipped table.
- **Limit.** gpt-3.5's code comes from the same prompt set as the test. A held-out prompt's own solution is in no table, but the style of problem is the same, so that table may do less for other coding requests.

**Live on CPU, 19:57 to 20:25** (`run_code_live.sh`, `results/code-9b-cpu-20261110-1956.json`). Qwen3.5-9B Q4_K_M on 8 threads, 12 test prompts (`prompts/code-v1.jsonl`: 6 Python, 6 other languages), 192 tokens each, temperature 0, thinking off, two-token drafts, arms taking turns every two prompts. Load was 11 to 12, so this is a pilot. The code table is shipped + own + gpt-3.5's, contexts seen at least 3 times, counted when 234 of the 800 answers existed.

| Arm | Tokens/s | Against plain | Per prompt: median (range) | Draft tokens accepted | Tokens per step |
|---|---:|---:|---:|---:|---:|
| Plain decoding | 8.50 | 1.00× | | | 1.00 |
| **Code table + request** | 11.62 | **1.37×** | 1.34× (0.96 to 2.08) | 893 of 2,635 | 1.67 |
| Shipped table + request | 10.32 | 1.21× | 1.21× (0.84 to 1.96) | 741 of 2,939 | 1.50 |
| Request only | 9.93 | 1.17× | 1.18× (0.80 to 1.83) | 646 of 3,122 | 1.41 |
| `ngram-cache` | 9.23 | 1.09× | 1.01× (0.77 to 1.68) | 563 of 2,369 | 1.35 |
| `ngram-mod` | 9.13 | 1.07× | 1.05× (0.73 to 1.89) | 205 of 491 | 1.10 |

- **The offline study predicted the tokens per step:** 1.61, 1.52 and 1.42 for the three table arms, against 1.67, 1.50 and 1.41 measured.
- **On code the table's text matters more than on the mixed prompts.** The code table took the shipped table's 1.21× to 1.37×.
- **Output equalled plain decoding on 4 of 12 prompts** in each table arm (`ngram-cache` 2, `ngram-mod` 9). The three table arms wrote the same text as each other although they drafted different tokens, and plain decoding itself differed on 8 of these 12 prompts from the run that wrote the answers 16 at a time. So the differences look like the batch shape tipping near-ties, not the drafts. That has not been checked further.
- **Limits.** Twelve prompts, 192 tokens, CPU only, a busy machine: the order of the arms is firmer than the numbers.

### Own code at scale, and the reranker (2026-11-11 and 12)

**Generation.** The 9B (Q4_K_M, temperature 0, thinking off, up to 1,024 tokens) answered every OSS-Instruct prompt past the 2,000-prompt pool that holds the test and calibration prompts: places 2,000 to 6,199 on this machine's CPU (4,200 answers, 2.3M tokens, 43 tok/s), the other 68,985 on one rented A40 (`run_gen_pod.py`: llama.cpp 7d7f028 with CUDA, two servers on the GPU at about 770 tok/s together; 36.9M tokens in 13.8 hours, $8.71 of a $15 cap). 17% of the pod's answers stop at the length limit. On 12 prompts answered both ways the pod's text and the CPU's agree up to near-ties, and in the table they are worth the same (1.455× and 1.455× when one stands in for the other, token for token).

**The table.** Offline, code test answers, pad 3, the 9B's GPU step costs. The reference holds the shipped text, 1.15M tokens of own code and the gpt-3.5 solutions:

| Pod answers added | Own code in the table | Drafter | Reranker (rerank-more/600.pt, as it is) |
|---:|---:|---:|---:|
| none | 1.15M tokens | 1.455× | 1.559× |
| 4,000 | 3.2M | 1.476× | 1.583× |
| 8,000 | 5.4M | 1.486× | 1.598× |
| 16,000 | 9.6M | 1.501× | 1.613× |
| 32,000 | 18.2M | 1.523× | 1.633× |
| 64,000 | 35.4M | 1.542× | 1.654× |
| all, with the 4,200 from the CPU | 39.5M | 1.545× | 1.655× |

Each doubling of own code still added about 0.02× at the end.

**The reranker** (`rerank-scratch/rerank7.py`) scores each distinct proposed token with an MLP (577-512-256-128-1) over what the lookups propose, token kinds and frequencies, and 64-wide projections of the target's embeddings, starting from the earlier stand-in reranker. More training data helped it little: fine-tuned on this machine's 4,800 answers 1.666×, with a quarter of the pod's answers too 1.666×, with all of them (a fifth of their positions, 17.7M rows) **1.676×**. Over the whole push, the table gave +0.096× and the reranker's extra data +0.021×.

**In llama.cpp.** `export_reranker.py` writes the table in a version 2 format (exact confidences and the count cells the reranker reads; contexts seen at least 3 times: 14.8M entries, 256 MiB) and the reranker (124 MiB, embeddings already projected). `--spec-table-reranker FILE` turns it on. On the 68,040 test positions the C++ drafter picks the same token as the numpy reranker at 99.975% of them. A draft call with the reranker takes about 60 µs (AVX2 chosen at load time; 3 µs without the reranker). Cutting rare contexts costs the reranker 0.026× offline (1.676× to 1.650×).

**Live on CPU, 2026-11-12 16:37 to 17:10** (`run_rerank_live.sh cpu`, `results/rerank-9b-cpu-20261112-1637.json`). Same 12 test prompts and settings as the 2026-11-10 run; load rose from 6 to 20 during the run, so the speeds are noisy and the tokens per step are the firmer numbers.

| Arm | Tokens/s | Against plain | Per prompt: median (range) | Draft tokens accepted | Tokens per step |
|---|---:|---:|---:|---:|---:|
| Plain decoding | 5.03 | 1.00× | | | 1.00 |
| Code table of 2026-11-10, 2 tokens | 6.85 | 1.36× | 1.26× (0.98 to 2.29) | 904 of 2,611 | 1.68 |
| **Table with the pod's code, 2 tokens** | 7.58 | **1.51×** | 1.39× (1.06 to 3.28) | 1,021 of 2,380 | 1.84 |
| Same table + reranker, 2 tokens | 7.58 | 1.51× | 1.58× (1.05 to 2.66) | 1,074 of 2,271 | 1.93 |
| Same table + reranker, 3 tokens | 7.49 | 1.49× | 1.46× (0.94 to 3.39) | 1,156 of 3,088 | 2.10 |

- **The larger table shows up live:** 1.36× to 1.51×, faster on all 12 prompts.
- **The reranker adds 5% tokens per step** (1.84 to 1.93; offline replay predicted 6%), which did not show in tokens per second on this busy machine.
- **Three-token drafts accept the most but cost more per step on CPU.** The GPU, where a draft token costs less, is where they should pay.

**Live on the GPU, 2026-11-12 17:14 to 17:18** (`run_rerank_live.sh vk`, `results/rerank-9b-vk-20261112-1712-turn1.json`). Vulkan on the Radeon 8060S, the same 12 prompts and 192 tokens, every arm in one turn with no other process on the GPU (the guard watched throughout); the rest of the machine was busy. One prompt round per arm, so a pilot.

| Arm | Tokens/s | Against plain | Per prompt: median (range) | Draft tokens accepted | Tokens per step | Step cost |
|---|---:|---:|---:|---:|---:|---:|
| Plain decoding | 33.7 | 1.00× | | | 1.00 | 1.00 |
| Code table of 2026-11-10, 2 tokens | 56.1 | 1.67× | 1.76× (1.22 to 2.21) | 882 of 2,624 | 1.66 | 0.96 |
| Table with the pod's code, 2 tokens | 60.9 | 1.81× | 1.96× (1.29 to 2.21) | 987 of 2,421 | 1.80 | 0.98 |
| Same table, 3 tokens | 62.4 | 1.86× | 2.02× (1.22 to 2.30) | 1,099 of 3,288 | 1.98 | 1.04 |
| Same table + reranker, 2 tokens | 62.4 | 1.86× | 2.02× (1.29 to 2.33) | 1,034 of 2,325 | 1.87 | 0.99 |
| **Same table + reranker, 3 tokens** | **65.0** | **1.93×** | 2.10× (1.32 to 2.42) | 1,150 of 3,140 | 2.08 | 1.05 |

- **On the GPU a drafted step costs about what a plain step does** (0.96 to 1.05 of one), so speed follows tokens per step almost one for one.
- **Every step up shows:** the larger table +0.14×, the reranker +0.05× at either draft length, three tokens over two +0.05 to 0.07×. All 12 prompts were faster than plain decoding in every arm.
- **Output** equalled plain decoding on 7 or 8 of 12 prompts per arm.

## What is not known yet

- **Tokens per second on a quiet machine.** `run_measure.sh` is the clean run (step costs, then the drafter against plain decoding, `ngram-simple`, `ngram-mod` and `ngram-cache`). It has to be started by hand, because `dibs run --measure` holds every other session's requests while it waits.
- **The lookup drafters on the GPU, and on long answers that copy their input.** The CPU pilot above is 8 short prompts.
- **Step costs on other models,** and on CPU under a known load.
- **CPU with fewer threads, and very short answers.** Three throwaway checks of the benchmark script on 2026-11-10 (4 threads, 2 prompts, 12 tokens each, far too little to count) had a drafted step at 1.7 to 1.8 plain steps, against 1.26 in the 8-thread run, and the table drafter slower than plain decoding. Either tokens added to a batch cost more when decoding is limited by arithmetic and not by memory bandwidth, or a fixed cost at the start of each answer shows at 12 tokens. Neither has been measured.
- **`ngram-cache` with a table of its own.** llama.cpp's lookup cache can load a static file of 2-token contexts built ahead (`--lookup-cache-static`). That is the nearest existing thing to this drafter and has not been run.
- **Whether to fix the runtime instead of the drafter.** Keeping one graph per batch size is predicted to be worth about 0.1× more than fixed-length drafts. It is a change inside llama.cpp's context, not in the drafter.
- **Whether the 9B's table drafts as well for other models** in the family (27B, 35B-A3B, 2B). One short run on the 4B Bonsai-2 accepted 36% of the table's drafts.
- **Chat with more chat text.**
- **Code on the GPU,** and coding requests that are not like OSS-Instruct's (editing a given file, long programs).
- **Where the own-code curve flattens.** 39M tokens of the 9B's code still gained about 0.02× per doubling offline. Prompts beyond OSS-Instruct would be needed to go further.
- **The GPU numbers on a quiet machine and with more prompts.** The GPU run above is one round of 12 prompts, beside a busy CPU.
- **Why greedy output changes with the batch shape as often as it did on code** (8 of 12 prompts within 192 tokens), and whether it matters for the answers.
- **Quality.** Speculative decoding in llama.cpp is not bit-exact (see the first attempt's lessons). Outputs here matched plain decoding on most prompts; no task score has been run.

## Files

| File | What it does |
|---|---|
| `ngram_study.py` | The offline study: counts tables, calibrates confidences, drafts every held-out position, predicts speed for a step cost |
| `build_table.py` | Builds the table file the drafter reads; `--check` proves the C++ and numpy drafters agree |
| `memdraft.h` | The drafter: table lookup, in-request index, chain rule |
| `replay.cpp` | Decodes recorded requests with `memdraft.h`, with no model |
| `patch_llama.py` | Adds `--spec-type ngram-table` and its options to a llama.cpp tree. In a chain of drafters it runs after `ngram-simple`, `ngram-map-*` and `ngram-mod`, before `ngram-cache`. |
| `live_bench.py` | Tokens per second of a llama-server, one arm per set of options |
| `show_live.py` | Prints a `live_bench.py` result: every prompt under every arm, then paired speeds per arm |
| `gpu_alone.py` | Waits until no other process is on the GPU, or none but preemptible jobs (`--preemptible-ok`) |
| `run_pilot.sh`, `run_measure.sh` | The pilot and the clean timing runs |
| `run_lookup.sh` | The table drafter against llama.cpp's lookup drafters, on the GPU or CPU |
| `gen_code.py`, `run_gen_code.sh` | Has the target answer coding prompts and keeps the answers as token ids |
| `code_study.py` | The offline study on that code: tables from the shipped text, the target's own code and another model's code. `--export` writes one of them for the drafter. |
| `run_code_live.sh` | The table drafter on coding prompts against plain decoding and the lookup drafters, on CPU |
| `run_gen_more.sh` | The target answers more coding prompts on this machine's CPU |
| `run_gen_pod.py`, `gen_pod_worker.py`, `gen_pod_helper.py`, `gen_pod_requirements.txt` | The same on one rented RunPod GPU, under a dollar cap (budget file `runpod_budget_gen_*.json`); `collect_pod_rows.py` joins its rows |
| `export_reranker.py` | Writes the code table (version 2) and the reranker for the drafter, and checks the C++ reranker against numpy |
| `run_rerank_live.sh` | The new table and the reranker live on coding prompts, CPU or GPU |
| `test_ngram_study.py` | Checks the fast paths against slow reference versions |
| `results/` | Study and timing results |

Scratch, outside git: the patched llama.cpp and the test GGUFs are in `~/.cache/model-experiments/memdraft/`; tables are in `target/memdraft/` (`9b-all-c3.mdt` is the 64 MiB table used live). The 9B's code answers (`rows-9b.jsonl`) and the code tables are in `target/memdraft/code/` (`9b-code-800-c3.mdt` is the 128 MiB table with all the code); the OSS-Instruct file is in `~/.cache/model-experiments/memdraft/code-prompts-src/`.

```bash
S=manifold-compression/results/heal-shards-pod-20261105
SH="$S/mc-shard-ec420510e5b4/shards/shard_000 $S/mc-shard-d65446e1cd6b/shards/shard_001 $S/mc-shard-1135ab4d74a4/shards/shard_002"
PY=target/manifold-audit/env/bin/python
$PY memdraft/ngram_study.py --shards $SH --cache target/memdraft/cache --out memdraft/results/study.json
$PY memdraft/build_table.py --shards $SH --cache target/memdraft/cache --out target/memdraft/9b-all-c3 --all-shards --min-count 3
python3 memdraft/patch_llama.py ~/.cache/model-experiments/memdraft/llama.cpp   # then build with /usr/bin/gcc, not ~/.local/bin/cc
llama-server -m MODEL.gguf --spec-type ngram-table --spec-table-file target/memdraft/9b-all-c3.mdt --spec-table-tau 0.3

# code: the 9B answers 800 prompts (hours on CPU), then the study and a table with code in it
setsid nohup memdraft/run_gen_code.sh 800 > target/memdraft/code/gen.log 2>&1 < /dev/null &
$PY memdraft/code_study.py --rows target/memdraft/code/rows-9b.jsonl --out memdraft/results/code-study.json
$PY memdraft/code_study.py --rows target/memdraft/code/rows-9b.jsonl --min-count 3 --tables 'shipped + own + other all' \
    --export 'shipped + own + other all=target/memdraft/code/9b-code-800-c3' --prompts-out memdraft/prompts/code-v1.jsonl
```

## The first attempt (2026-10-25 to 2026-11-07): failed

A retained-continuation cache for speculative decoding, in front of DFlash2.

**Question.** Does a bounded table of past token continuations, kept according to how many tokens the target actually accepted, make greedy decoding faster than DFlash2 alone, or than existing n-gram + DFlash2 setups, at the same memory budget?

**Answer: no.** On new requests the cache barely changed accepted tokens per proposal and lowered acceptance. The one timed comparison was 4% slower on new prompts. On Qwen3.6-35B-A3B, MemDraft in front of DFlash2 produced fewer tokens per target step than DFlash2 alone. Gains appeared only on exact repeats. No controlled latency result was ever obtained for any setting. The project ran 2026-10-25 to 2026-11-07 in `pair-cache-spec/`. It was retired on 2026-11-08 and its code, results and caches were deleted. This write-up is what's left.

### What was built

- `pair_cache.h`: a C++17 RAM table with 128-byte slots, 4- or 8-token keys, 1–16-token continuations and 8-way buckets. Sizes run from 512 KiB (4,096 entries) to 1 GiB (8.4 M entries).
- Three retention policies: `utility` (accepted-token credit, aged; the default), `frequency` and `lru`. Request or global scope, and an optional snapshot file.
- `ranker.h`: a 128→32→16 MLP (4,656 parameters) that picks a candidate and a draft length, or declines and lets DFlash2 draft. A larger MLP and a 2-layer token transformer were tried offline.
- A llama.cpp patch on upstream `b3fa699` adding `--spec-type ngram-pairs` and `--pair-*` flags, chained before `draft-dflash`.
- Replay and sweep tools, live acceptance harnesses, DFlash2 drafter quantization studies, and Qwen3.6 head-to-head and recurrence drivers.
- Runs were on the CPU of the Ryzen AI Max+ 395 (8 threads), greedy, thinking off, context 4096.

Weights:
- Qwen3.8-27B Q4_K_M target (`ggml-org/Qwen3.8-27B-GGUF` @ a316ea5) with Inco's DFlash2 Q4_K_M drafter (`incoai/Qwen3.8-27B-DFlash2-GGUF` @ 144eb06).
- Later, Qwen3.6-35B-A3B UD-Q4_K_M with a Qwen3.6 DFlash2 Q8_0 drafter.

### Results

"Acceptance" means accepted ÷ drafted tokens. "Tokens/step" means generated ÷ (generated − accepted), the output tokens per target verification. Neither is wall-clock speed.

#### Acceptance pilot, Qwen3.8-27B (one run per setting, dibs CLEAN)

| | DFlash2 | heuristic + DFlash2 | ranker + DFlash2 |
|---|---|---|---|
| 8 new prompts: acceptance / accepted per proposal | 51.0% / 3.50 | 44.5% / 3.70 | 45.1% / 3.73 |
| 4 exact repeats | 58.8% / 4.11 | 77.6% / 11.74 | 81.9% / 12.32 |
| server time, new | 122.8 s | 127.4 s | 127.8 s (0.961×) |
| server time, repeats | 50.3 s | 36.2 s | 37.9 s (1.327×) |

All 24 hybrid outputs matched DFlash2. The baseline process was warmer, because it had served 8 training requests first.

#### Three-domain study, Qwen3.8-27B

The study used 90 requests across notes, code and chat: 36 training, 36 new and 18 exact repeats. It covered 448 cache-only replay configurations, 12 ranker configurations, and 8 live settings (432 completions).

| Live setting | New: acceptance / per proposal | Repeats: acceptance / per proposal | Outputs matching DFlash2 |
|---|---|---|---|
| DFlash2 alone | 49.1% / 3.37 | 51.5% / 3.60 | reference |
| 128 KiB | 43.7% / 3.53 | 43.6% / 3.80 | 53/54 |
| 2 MiB | 39.9% / 3.39 | 68.1% / 9.11 | 51/54 |
| 32 MiB | 39.7% / 3.39 | 74.2% / 10.44 | 52/54 |
| 1 GiB | 39.7% / 3.39 | 74.9% / 10.53 | 52/54 |
| 2 MiB, LRU | 39.9% / 3.39 | 65.4% / 8.20 | 51/54 |
| 2 MiB, MLP ranker | 40.1% / 3.41 | 66.9% / 8.65 | 50/54 |

- **New requests were flat:** 3.37 accepted per proposal without the cache, 3.39–3.53 with it.
- **Table size beyond 0.5–2 MiB added nothing.** The 1 GiB table filled only about 21,000 entries on this stream.
- **Coverage of new requests at 2 MiB:** notes 51%, code 47%, chat 2%.
- **Retention policy barely mattered.** Repeat tokens at 2 MiB: utility 1,330, LRU 1,290, frequency 1,331. Global scope beat request scope, 1,330 to 544.
- **No learned selector beat the heuristic.** New tokens with fixed candidate states: heuristic 1,120, native MLP 1,129, larger MLP 1,150, transformer 915 (overfit).

#### Qwen3.6-35B-A3B head-to-head (2026-11-06; 6 arms × 200 prompts, 50 per domain, CPU)

| Arm | Acceptance | Tokens/step | code | chat | reasoning | notes | Output identical to no-spec |
|---|---:|---:|---:|---:|---:|---:|---:|
| no speculation | – | 1.00 | | | | | – |
| MTP (`draft-mtp`, 3) | 76.1% | 3.25 | 3.75 | 2.78 | 3.54 | 3.14 | 65/200 |
| **DFlash2 Q8_0 (cap 7)** | 54.0% | **4.69** | 6.93 | 3.19 | 5.79 | 4.46 | 50/200 |
| prompt lookup (`ngram-simple`) | 26.4% | 1.33 | 2.34 | 1.10 | 1.22 | 1.27 | 69/200 |
| lookup + DFlash2 | 45.7% | 4.07 | 5.69 | 3.10 | 4.56 | 3.84 | 58/200 |
| MemDraft (32 MiB) + DFlash2 | 39.5% | 4.48 | 7.70 | 3.22 | 4.85 | 4.26 | 63/200 |

- **MemDraft beat DFlash2 alone only on code** (7.70 against 6.93). Its own drafts were accepted 25.5% of the time.
- **The timing stage never ran.** Rough decode speeds worked out afterwards from the request logs, on the 32 prompts with little CPU contention: no speculation 9.6, DFlash2 14.1, MemDraft 10.1, lookup + DFlash2 11.1 tok/s. These are not a controlled measurement.
- **The recurrence workload never answered its question.** It was meant to test whether history from earlier requests helps, using multi-turn code and notes, reworded repeats, a fixed JSON schema and exact retries. Only the MemDraft arm started, and it stopped at 27 of 144 requests with no logged error and no control arms.

#### DFlash2 drafter quantization (MemDraft off)

Over 400 prompts × 2 targets, a Bonsai2-tuned ternary DFlash2 drafter (35% smaller) lost 4.9–9.7 points of acceptance in every domain on the Qwen3.8 Q4 target. On its own Bonsai2 target it gained 1.2–4.0 points. The two checkpoints also differ in training, so this isn't a pure quantization effect. The controlled timing stage never got the CPU.

### Why it failed

- The cache only pays off on exact repeats. On new requests, 16-token cache drafts are often wrong, so they cut acceptance while barely raising accepted tokens per proposal.
- Acceptance is not speed. Long rejected drafts cost verification time, so new prompts got slower.
- DFlash2 is already a strong drafter. Any n-gram source placed in front of it lowered overall tokens/step on Qwen3.6, whether lookup or MemDraft.
- Close prior art already covers the idea (Cacheback, SuffixDecoding, ReSpec, public DFlash2 + n-gram runs), and no advantage over it was shown.
- Every timing stage ended up queued behind other jobs on a shared machine, so no speed result was ever clean.

### Lessons worth keeping

- **Greedy output from llama.cpp depends on batching, so speculative decoding is not bit-exact here.**
  - On Qwen3.8, 18 of 432 cache completions diverged from the DFlash2 reference. A probe reproduced one divergence with DFlash2 alone at another draft cap and with plain decoding. The top-2 logprobs at the split were −1.106 and −1.184.
  - On Qwen3.6, only 50 of 200 DFlash2 outputs, 65 MTP and 69 lookup outputs matched plain greedy decoding.
  - Compare speculative settings by task quality, not by identical text.
- **DFlash2 out-drafts MTP on Qwen3.6-35B-A3B on general prompts:** 4.69 tokens/step against 3.25. This matters for npu-kernel-coder, which uses Qwen3.6 with MTP; see the note below.
- **A tiny model's speedup means nothing.** stories15M went from 879 to 2,431 tok/s on repeated prompts.
- **Hook speculative-decoding commits at the token commit.** The first adapter learned tokens only at the next draft call, so each request's final bonus token was lost (178 of 181 seen; 181/181 after the fix).
- **Measurement on a shared machine:**
  - Load 49 gave 2× swings on identical prompts.
  - Warm and cold processes were compared, which is unfair.
  - Two concurrent 35B CPU servers were OOM-killed.
  - A cache saved only on graceful shutdown was lost on each kill.
  - Timing stages queued behind other dibs holders never ran.
  - Plan timing as a single `dibs run --measure` job from the start.

### Note for npu-kernel-coder

The coder model (Qwen3.6-35B-A3B) runs with its MTP head: 96% acceptance and 92.5 tok/s on the shape-port pilot (`npu-kernel-coder/eval/results.md`). On MemDraft's general prompts, DFlash2 gave more tokens per step than MTP. So the Qwen3.6 DFlash2 Q8_0 drafter was kept, at `/home/cyrus/models/qwen3.6-35b-a3b-mtp/`, for a head-to-head on the coder workload.
