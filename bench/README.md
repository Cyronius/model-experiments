# bench

Shared benchmark harness for GGUF models behind llama-server. `run_bench.py` runs the suites for each model in `registry.json`; `mtbench_judge.py` scores MT-Bench answers afterwards.

## Suites

| suite | what it measures | scoring |
|---|---|---|
| `tools` | tool-calling accuracy (thinking on) | rules |
| `gsm8k` | grade-school maths | final number |
| `mmlu` | multiple-choice knowledge | letter |
| `ifeval` | following verifiable instructions (541 prompts from `google/IFEval`) | rules, via `lm_eval.tasks.ifeval` |
| `mtbench` | open-ended two-turn chat (80 questions) | Claude Haiku judge, 1-10, via `mtbench_judge.py` |
| `perplexity` | wikitext-2, 32 chunks | llama-perplexity |
| `speed` | prompt and generation tokens/s | llama-server timings |

Every open-ended answer (ifeval, mtbench) also gets `degenerate_rate`: the share of answers whose repeated word-4-gram share is above 0.15, which is how looping shows up.

Setup: `python -m pip install --user lm_eval langdetect immutabledict nltk` for the IFEval scorer. MT-Bench judging runs through Claude Haiku subagents in a Claude Code session (no API key needed): `python mtbench_judge.py batches --runid R`, judge each `batch_NN.json` into `batch_NN_out.json` following `INSTRUCTIONS.md`, then `python mtbench_judge.py merge --runid R`.

## Decisions

**2026-10-19: chat usability is gated on IFEval and MT-Bench, not on tools / GSM8K / MMLU alone.**
- Why: in the manifold-compression smoke test, a 2B build that kept tools at 0.90 and MMLU at 0.533 looped on half of 40 open-ended prompts under greedy decoding and lost most of its coherence (Haiku-judged 2.75 of 5 vs 4.17 for stock). Short-answer suites hide that, and the old `loop_rate` only counts tool answers that hit the cap with no visible content.
- IFEval is the main gate: deterministic, no judge cost or noise, and a widely published number. MT-Bench adds a judged open-ended chat score; it is judged by Claude Haiku, so its scores compare our models with each other, not with published GPT-4-judged MT-Bench numbers.
- `degenerate_rate` (repeated 4-gram share above 0.15) measures looping directly; the threshold separated every Haiku-flagged loop (at least 0.18) from every stock answer (at most 0.12) on 80 judged answers.
- Speed: run on the iGPU (Vulkan build, `-ngl 99`) with 4 requests in flight for these two suites.
