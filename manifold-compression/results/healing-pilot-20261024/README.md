# Controlled healing pilot, 2026-10-24

This experiment tests whether a short whole-model repair helps the fresh
width-1792 initializations, and whether correcting the final norm improves
recovery. It retains all MLP neurons, heads, and layers. It is a small recovery
pilot, not a broad parity benchmark.

`protocol.json` records the settings fixed before outcome evaluation.
`data.json` contains the exact examples, source hashes, tokenizer-specific
counts, held-out prompts, and transparent scoring rules. The 48 corpus training
examples and 12 corpus validation examples come from disjoint splits of
[UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
at revision `8049631c405ae6576f93f445c6b8166f76f5505a`; 16 additional training
examples were written and reviewed locally. Eighteen training conversations
include two user/assistant turns. No assistant answer is cut to fit the limit:
long examples are rejected, and every supervised response includes EOS.
Only the final assistant response is scored; earlier turns provide context.

Each architecture has two matched training arms. Both start with the same
projected body and the same trainable dense adapter capacity. The legacy arm
starts with the old diagonal and inverse-scale function; the corrected arm
starts with the full matrix and corrected RMS scale. Both can learn matrix
entries during healing. This isolates initialization, rather than confounding
it with different numbers of trainable parameters.

Each arm receives one pass over the same 64 conversations, two per update,
for 32 optimizer updates. Qwen sees 14,021 input tokens and 7,870 supervised
assistant tokens; SmolLM2 sees 15,512 and 8,072 respectively. Training uses
full-vocabulary forward teacher KL plus 0.1 times cross-entropy on the actual
assistant tokens, including the end token. The optimizer is AdamW at 2e-5,
two-update warmup, no weight decay, and gradient clipping at 1. All model
parameters train. Parameters remain FP32, with CPU BF16 autocast during
training; validation and generation run in FP32. Teacher hidden states are
cached exactly in FP32; training head multiplication uses BF16.

There are 12 held-out complete conversations for token-weighted KL, target
NLL, EOS NLL, and teacher top-1 agreement. Sixteen separately authored generation
cases cover exact answers, code, formatting, explanation, and multi-turn
context. Decoding is greedy with thinking disabled, no repetition penalty,
and a 512-new-token cap. Completion, repeated four-gram fraction, and mechanical
checks are reported separately. A repetition flag requires at least 40 words
and repeated four-gram fraction above 0.20; it can flag legitimate repetition.
The rubric checks are intentionally simple and do not constitute a complete
semantic quality judge. Raw answers must also be inspected.

Stock, both starting functions, and both final trained checkpoints use the
same generation cases. The recovered 500M-token Qwen student is an additional
reference; it has a different size and training history, so it is not a matched
training arm. The final fixed-budget checkpoint is evaluated without tuning
on these outcomes.

Local source files:

- `../../healing_data.py`: data filtering, token alignment, and scoring rules.
- `../../healing_pilot.py`: cache, training, reload checks, and generation.
- `../../test_healing_pilot.py`: mask/end-token, padding, control initialization,
  loss-gradient, split, and scoring checks.

Example commands from the repository root:

```bash
target/manifold-audit/env/bin/python manifold-compression/healing_data.py --out manifold-compression/results/healing-pilot-20261024/data.json

target/manifold-audit/env/bin/python manifold-compression/healing_pilot.py cache --arch qwen35

target/manifold-audit/env/bin/python manifold-compression/healing_pilot.py train --arch qwen35 --arm corrected --threads 8

target/manifold-audit/env/bin/python manifold-compression/healing_pilot.py evaluate --arch qwen35 --label corrected-after --threads 8
```

Repeat training with `--arm legacy` and both arms with `--arch smollm2`.
Generation labels are `stock`, `legacy-before`, `corrected-before`,
`legacy-after`, `corrected-after`, and Qwen-only `recovered-500m`.
The trainer refuses to overwrite a completed healed checkpoint. `summarize.py`
requires all paired runs to finish before writing `combined.json`.

The dedicated CPU environment is pinned in `requirements-lock.txt`.
`tests.txt` records the test results. RunPod spending is $0.
