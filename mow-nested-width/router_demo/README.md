# Router demo: per-request width routing for Gemma 3n E4B

Gemma 3n E4B is trained with MatFormer nesting: its FFN weight matrices
contain a smaller, independently-trained model inside them. Slicing the
FFN's intermediate dimension in half (16384 -> 8192, keeping attention and
everything else untouched) produces a checkpoint that still answers
correctly and, measured in llama.cpp on an A40 (bf16, GPU-offloaded), decodes
**1.44x faster** (76.9 vs 53.4 tok/s) and prefills 1.26x faster. See
`.claude/plans/dynamic-width-gemma4.md` for the full measurement history.

This demo puts those two checkpoints behind one router. Each incoming chat
request is classified as easy or hard by a small set of transparent rules
and sent to whichever backend fits: the half-width model for quick,
low-stakes turns; the full model for anything that looks like code, math, or
a request for careful reasoning. It is a working illustration of
per-request width selection, not a production serving system -- see LIMITS
below for what is deliberately out of scope here.

## What's in here

| File | Purpose |
|---|---|
| `setup_models.py` | Downloads, slices, converts, and quantizes both checkpoints; fetches llama.cpp. Run once. |
| `router_server.py` | Launches both llama-server backends and exposes `POST /chat` with rule-based routing. |
| `chat_cli.py` | Minimal interactive client. |
| `test_router.py` | pytest suite -- no models, no network, no GPU. |

## Setup

Requires Python 3.11+, a HuggingFace token with access to
`google/gemma-3n-E4B-it` (set `HF_TOKEN`, or run `huggingface-cli login`
first), git, and about 50 GB of free disk.

```
pip install huggingface_hub safetensors torch
python router_demo/setup_models.py
```

This is one command but a long one (it downloads a ~15 GB checkpoint twice
over, converts, and quantizes). It is idempotent: every step checks whether
its output already exists and skips it, so an interrupted run can just be
restarted. It creates its own venv at `router_demo/.venv` for the GGUF
converter's dependencies (transformers, sentencepiece, etc.) -- those never
touch the interpreter you ran `setup_models.py` with.

When it finishes, `router_demo/models/` holds `full-Q4_K_M.gguf` (~4.5 GB)
and `half-Q4_K_M.gguf` (~3.5 GB), and `router_demo/bin/` holds
`llama-server.exe`, `llama-quantize.exe`, `llama-bench.exe`, and their DLLs.

## Running the demo

```
python router_demo/router_server.py
```

In another terminal:

```
python router_demo/chat_cli.py
```

Type messages; each reply is followed by a status line, e.g.:

```
[half | 74.3 tok/s | reason: short-simple]
[full | 51.8 tok/s | reason: code]
```

`router_server.py --dry-run` skips launching the backend subprocesses --
useful for pointing the router at backends started some other way, or as
the basis for the automated tests, which instead call the router's
`build_server()` directly with mock backends (see `test_router.py`).

### Hardware notes

`CONFIG["ngl"]` in `router_server.py` is 99 (offload all layers) by default,
for a GPU with enough VRAM for both models resident at once. Set it to `0`
to run on CPU. The Q4_K_M quantization is what makes CPU serving practical
at all -- the bf16 intermediates in `models/` are much larger and are only
there as the conversion's midpoint.

## Routing rules

`route(messages) -> (tier, reason)` in `router_server.py` is a pure
function -- no I/O, fully unit-tested. Roughly: route to the full model if
the latest message contains a code fence or code-shaped tokens
(`def`, `class`, `{`, `}`, `;`), math beyond simple arithmetic (`=`, `^`,
"integral"/"derivative", or a run of more than 6 digits), a reasoning
keyword (prove/derive/debug/analyze/step-by-step), or is over 120 words.
Otherwise, if any *earlier* message in the conversation would have
triggered one of those rules, stay on the full model (stickiness, upward
only -- see the comment above `route()` for why). Everything else goes to
the half model.

## LIMITS

This is a demonstration of the routing idea, not the deployment architecture
described in the project plan:

- **Two resident models, not one shared checkpoint.** `router_server.py`
  runs two separate `llama-server` processes, so it holds both the full and
  half checkpoints in memory at once -- the opposite of the memory win a
  real shared-weight runtime would give you. A single-copy runtime that
  restricts its compute graph to a width slice at request time is future
  work; nothing here builds it.
- **No mid-conversation KV handoff.** Once a conversation is routed to a
  tier for a turn, that turn's context lives only in that backend's KV
  cache. Switching tiers means the new backend starts cold on that turn.
  The project's phase-0/phase-2 experiments show tier switches on shared KV
  are cheap in principle; this demo does not implement the handoff.
- **The router is rule-based, not learned.** It is a small set of hand-written,
  auditable heuristics on the raw text, not a trained classifier. The
  project plan's phase 3 (a learned router validated against an oracle) is
  separate, unimplemented work.
