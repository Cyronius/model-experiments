# MemDraft: a retained-continuation cache for speculative decoding (failed)

**Question.** Does a bounded table of past token continuations, kept according to how many tokens the target actually accepted, make greedy decoding faster than DFlash2 alone, or than existing n-gram + DFlash2 setups, at the same memory budget?

**Answer: no.** On new requests the cache barely changed accepted tokens per proposal and lowered acceptance. The one timed comparison was 4% slower on new prompts. On Qwen3.6-35B-A3B, MemDraft in front of DFlash2 produced fewer tokens per target step than DFlash2 alone. Gains appeared only on exact repeats. No controlled latency result was ever obtained for any setting. The project ran 2026-10-25 to 2026-11-07 in `pair-cache-spec/`. It was retired on 2026-11-08 and its code, results and caches were deleted. This write-up is what's left.

## What was built

- `pair_cache.h`: a C++17 RAM table with 128-byte slots, 4- or 8-token keys, 1–16-token continuations and 8-way buckets. Sizes run from 512 KiB (4,096 entries) to 1 GiB (8.4 M entries).
- Three retention policies: `utility` (accepted-token credit, aged; the default), `frequency` and `lru`. Request or global scope, and an optional snapshot file.
- `ranker.h`: a 128→32→16 MLP (4,656 parameters) that picks a candidate and a draft length, or declines and lets DFlash2 draft. A larger MLP and a 2-layer token transformer were tried offline.
- A llama.cpp patch on upstream `b3fa699` adding `--spec-type ngram-pairs` and `--pair-*` flags, chained before `draft-dflash`.
- Replay and sweep tools, live acceptance harnesses, DFlash2 drafter quantization studies, and Qwen3.6 head-to-head and recurrence drivers.
- Runs were on the CPU of the Ryzen AI Max+ 395 (8 threads), greedy, thinking off, context 4096.

Weights:
- Qwen3.8-27B Q4_K_M target (`ggml-org/Qwen3.8-27B-GGUF` @ a316ea5) with Inco's DFlash2 Q4_K_M drafter (`incoai/Qwen3.8-27B-DFlash2-GGUF` @ 144eb06).
- Later, Qwen3.6-35B-A3B UD-Q4_K_M with a Qwen3.6 DFlash2 Q8_0 drafter.

## Results

"Acceptance" means accepted ÷ drafted tokens. "Tokens/step" means generated ÷ (generated − accepted), the output tokens per target verification. Neither is wall-clock speed.

### Acceptance pilot, Qwen3.8-27B (one run per setting, dibs CLEAN)

| | DFlash2 | heuristic + DFlash2 | ranker + DFlash2 |
|---|---|---|---|
| 8 new prompts: acceptance / accepted per proposal | 51.0% / 3.50 | 44.5% / 3.70 | 45.1% / 3.73 |
| 4 exact repeats | 58.8% / 4.11 | 77.6% / 11.74 | 81.9% / 12.32 |
| server time, new | 122.8 s | 127.4 s | 127.8 s (0.961×) |
| server time, repeats | 50.3 s | 36.2 s | 37.9 s (1.327×) |

All 24 hybrid outputs matched DFlash2. The baseline process was warmer, because it had served 8 training requests first.

### Three-domain study, Qwen3.8-27B

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

### Qwen3.6-35B-A3B head-to-head (2026-11-06; 6 arms × 200 prompts, 50 per domain, CPU)

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

### DFlash2 drafter quantization (MemDraft off)

Over 400 prompts × 2 targets, a Bonsai2-tuned ternary DFlash2 drafter (35% smaller) lost 4.9–9.7 points of acceptance in every domain on the Qwen3.8 Q4 target. On its own Bonsai2 target it gained 1.2–4.0 points. The two checkpoints also differ in training, so this isn't a pure quantization effect. The controlled timing stage never got the CPU.

## Why it failed

- The cache only pays off on exact repeats. On new requests, 16-token cache drafts are often wrong, so they cut acceptance while barely raising accepted tokens per proposal.
- Acceptance is not speed. Long rejected drafts cost verification time, so new prompts got slower.
- DFlash2 is already a strong drafter. Any n-gram source placed in front of it lowered overall tokens/step on Qwen3.6, whether lookup or MemDraft.
- Close prior art already covers the idea (Cacheback, SuffixDecoding, ReSpec, public DFlash2 + n-gram runs), and no advantage over it was shown.
- Every timing stage ended up queued behind other jobs on a shared machine, so no speed result was ever clean.

## Lessons worth keeping

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

## Note for npu-kernel-coder

The coder model (Qwen3.6-35B-A3B) runs with its MTP head: 96% acceptance and 92.5 tok/s on the shape-port pilot (`npu-kernel-coder/eval/results.md`). On MemDraft's general prompts, DFlash2 gave more tokens per step than MTP. So the Qwen3.6 DFlash2 Q8_0 drafter was kept, at `/home/cyrus/models/qwen3.6-35b-a3b-mtp/`, for a head-to-head on the coder workload.
