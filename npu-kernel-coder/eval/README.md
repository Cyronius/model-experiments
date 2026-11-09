# Stage 0: shape-port baselines

The first model-side eval for [npu-kernel-coder](../plan.md). Each task asks a model to port a kernel to a new shape, entirely in [npuasm](../npuasm/) text. It gets build A of a design and the names of builds A and B, and must return the unified diff that turns A into B. The reference is B's real lifted text, so the compiler's output is the target.

## Tasks

`make_tasks.py` builds them from the local corpus (`../npuasm/results/corpus.jsonl`). A pair qualifies when:
- both builds sit in the same design directory, with build names that match once numbers become `#`;
- the core counts and lifted file sets match;
- the names differ by more than their hash;
- A's text is under 80k characters and the A→B difference is 1–300 changed lines;
- the two shapes are neighbours, at most one pair per name template, with duplicates across projects dropped.

The current set is 77 tasks from 33 design families: 12 easy (≤10 changed lines), 33 medium, 32 hard (>80). Prompts are 5k–23k Qwen tokens (median 14.8k). The split is by generator script (`generators.py`, since 2026-11-08): test 48 tasks from the generators that never enter training (gemma4 text kernels, gemma4 multimodal ops, lm_head), dev 29 from generators trained on at other shapes (attention GEMM, gemma4 projections, layer norm, gpt-oss). The first split held out design directories, but one generator writes into many directories, so test tasks had near-twins in dev. `python3 generators.py tasks` relabels an existing task set. The pilot stays pinned to its original 12 tasks (`run.PILOT`). `tasks/` and `runs/` are generated and not tracked.

The prompt (`prompt.py`) is a one-page primer on the npuasm format, the request, and A's files. Models are asked for a unified diff whose hunks are located by their context, so line numbers can be approximate. They are told to answer with the diff only and to use no tools.

## Scoring

`score.py` applies the diff, matching each hunk's context and removed lines nearest the stated line, then reports:

| Metric | Meaning |
|---|---|
| `exact` | every file equals B's reference text |
| `gap_closed` | 1 − d(answer, B) / d(A, B), with d counting changed lines: 1 is exact, 0 no progress, below 0 worse than A |
| `builds` | npuasm rebuilds the answer (every core assembles) |
| `rules_pass` | the rule gate finds no violation |
| `rules_pass_aligned` | it passes once branch-target labels are padded to 16 bytes (the repair runs only when alignment was the sole problem) |
| `bytes_match` | the rebuilt CDOs and `insts.bin` equal B's vendor binaries |

`check_harness.py tasks` is the oracle check. difflib's own A→B diff must be exact and byte-identical, and an empty answer must close no gap. Both hold on all 77 tasks.

## Functional check on the NPU

Exact match is the wrong bar when a port changes the core code, because Peano reschedules it. `npu_check.py` therefore checks answers that build and pass the rule gate differentially, with the real build B as the oracle on the NPU, so no CPU reference is needed:
1. B's runtime-sequence signature (buffer sizes and element types, from its `aie.mlir`) sizes every buffer.
2. All buffers are filled from one seed: floats N(0, 0.5), integer bytes uniform.
3. B's vendor xclbin runs twice; if the two runs disagree, the task is not scored.
4. The answer runs once, rebuilt by npuasm and packaged with B's xclbin as template.
5. Every buffer is compared, so stray writes outside the outputs show up too.

Verdicts: `equal` (bitwise), `close` (float buffers within 1e-2 of their scale), `differs`, `hang`, `vacuous` (B wrote nothing, so the task isn't scored). `--guard N` appends N bytes after every buffer, filled from a separate RNG so the buffers themselves don't change. B never writes them, so an answer that writes past the end of B's buffers (one still sized for a larger A) comes out `differs`. Answers that fail the rule gate only on branch alignment are checked after the alignment repair (`rules_pass_aligned`). `--recheck-equal --seed N --out-name npu_seedN.json` reruns the equal/close answers with another seed. Each run is its own process (`npu_exec.py`, pyxrt, 20 s timeout). After a hang, B is rerun as a health check, and the session stops if it fails. Run it under a dibs NPU hold:

```sh
dibs run npu --label "npu-kernel-coder differential check" --eta 30m -- \
    python3 npu_check.py --only-inexact --guard 1048576 runs/<tag>
```

**Control.** `run.py --backend null --model null-a --tasks all` answers every task with no edit (A unchanged). Where A itself passes the NPU check, the check can't tell a port from no port. `functional.py` uses `runs/null-a` to count NPU-only passes on those tasks as "unverifiable". With 1 MB guards that leaves 10 of 77 tasks; see [results.md](results.md).

`functional.py runs/<tag> [runs/<tag> ...]` counts functionally correct answers per difficulty: exact text, a byte-identical rebuild, or an NPU `equal`/`close` (the guarded verdict when there is one). It also prints a paired comparison of consecutive runs.

## Deterministic baseline

`baseline_scale.py ../npuasm/results/corpus.jsonl` writes `runs/baseline-scale/`. It needs no model: it rescales A's numbers to B's shape, fitted from the design's other builds where they exist and guessed from round multiples otherwise. It goes through the same scorer and NPU check as model runs. See [results.md](results.md).

## Running

```sh
python3 make_tasks.py ../npuasm/results/corpus.jsonl tasks
python3 check_harness.py tasks

# OpenAI models via Codex CLI signed in with ChatGPT (subscription, not API billing):
python3 run.py --backend codex --model gpt-6.1-sol --effort medium --tasks pilot
python3 run.py --backend codex --model gpt-6-astra --effort medium --tasks pilot

# Claude via Claude Code signed in with claude.ai (subscription, not API billing):
python3 run.py --backend claude --model claude-opus-5-5 --effort xhigh --tasks all --jobs 4

# Local Qwen3.5-27B Q4_K_M through llama.cpp. The GPU is shared: reserve it with dibs.
dibs run gpu --label "npu-kernel-coder stage-0 Qwen pilot" --eta 3h -- sh -c '
  /var/cache/lemonade/bin/llamacpp/vulkan/llama-server -m ../../models/claim-curation/teacher-qwen35-27b/Qwen3.5-27B-Q4_K_M.gguf \
    -c 32768 -ngl 99 -fa on --jinja -np 1 --host 127.0.0.1 --port 8091 & S=$!
  until curl -sf http://127.0.0.1:8091/health >/dev/null; do sleep 2; done
  python3 run.py --backend llama --model qwen3.5-27b-q4km --max-tokens 8192 --tasks pilot
  kill $S'
```

`--tasks pilot` is a fixed stratified subset of 12 tasks: 4 per difficulty, chosen test split first under the original split and pinned since. Codex runs in an empty read-only sandbox with the prompt on stdin. It loads the user's `~/.codex/AGENTS.md`, about 17k tokens of overhead per call, mostly cached. `claude` runs `claude -p` in an empty temp directory with all tools off (`--tools ""`), no MCP servers, no settings files and a one-line system prompt; `~/.claude/CLAUDE.md` still loads (about 600 tokens). `--bare` would skip it but accepts only an API key. `--jobs N` runs N tasks at once.

## Results

See `runs/<tag>/summary.json`; the numbers are recorded in [results.md](results.md) once runs finish.
