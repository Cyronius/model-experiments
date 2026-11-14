# Full-width rotation audit, 2026-10-24

The exact shared PCA rotation preserves predictions in both Qwen3.5-2B and
SmolLM2-1.7B-Instruct (Llama architecture). The final-normalization diagonal
approximation used in `build_projected.py` changes predictions even with zero
compression. This is a concrete source of initialization damage, not proof
that it explains the looping of the trained compressed student.

## Results

| Model | Exact rotation mean KL | Exact top-1 agreement | Diagonal final norm mean KL | Diagonal top-1 agreement | Wall time |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-2B | 7.40e-12 | 100% | 0.11296 | 82.65% | 22.85 s |
| SmolLM2-1.7B-Instruct | 7.94e-11 | 100% | 0.54545 | 83.04% | 19.88 s |

These are within-model comparisons, not quality comparisons between model
families. Four calibration prompts define a shared uncentered PCA basis with
per-site trace normalization. Four separate evaluation prompts contribute 98
Qwen and 171 SmolLM2 token positions, including prompt/template tokens. Prompt
texts, source revisions, and downloaded weight hashes are in the JSON reports.
No dimensions or MLP neurons were removed. No training was performed in this
audit. Computation used FP32 and FP64 KL evaluation to resolve numerical error.

Full-sequence versus cached inference agreed for both original and exactly
rotated models. Each cache check uses a prefill followed by eight single-token
updates on the same fixed token sequence; it is not an extended generation or
long-context validation. The recovered 500M checkpoint also loaded with no
missing or mismatched keys and passed this cache check (mean KL 2.55e-12).
GGUF conversion equivalence has not been checked by this audit.

## Why the old identity check missed this

The old builder sets P = I for its identity test. The actual build instead uses
a PCA basis. With a full-width orthogonal basis Q, the learned final RMSNorm
gain D becomes Q^T D Q. That generally has off-diagonal terms. Keeping only its
diagonal loses information even when the residual width remains unchanged.

The exact diagnostic retains the full gain matrix as a small in-memory module
before the tied output head. All other residual readers/writers are rotated
and internal norm gains are folded into their readers. The diagonal arm changes
only that final module, reproducing the builder's shortcut at full width.
Off-diagonal entries account for 12.56% and 13.32% of the gain matrix's Frobenius
norm in these Qwen and SmolLM2 bases, respectively.

The exact module is a diagnostic and is not yet an exportable stock Qwen or
Llama checkpoint. A production correction must explicitly handle tied heads
and account for any adapter or untied-head parameter cost. The current 500M
student has already trained all its weights; replacing its norm with a matrix
from this unrelated rotation is not a valid repair.

## Local compute and next decision

Machine: Ryzen AI Max+ 395, 16 physical CPU cores, 123 GiB OS-visible RAM. Audits
used 12 CPU threads, torch 2.14.1+cpu and transformers 5.14.1. Downloads and
environment setup are excluded from the wall times above.

A separate timing probe ran three full forward/backward/AdamW updates per
precision, using the recovered 0.752B student, Qwen3.5-2B teacher, one 512-token
row, and full-vocabulary forward KL in 32-position chunks. No updated weights
were saved. After the first warm-up step it achieved 30.46 tokens/s in FP32 and
57.06 tokens/s with CPU BF16 autocast. At that measured BF16 rate, 5M total
tokens take about 24.3 hours and 20M about 97.4 hours. This excludes generation,
evaluation, and data preparation; it is not a prediction for assistant-only
token counts or longer contexts. The synthetic repeated text measures runtime
only. GPU training has not been benchmarked on this machine.

Use this machine for further correctness tests. Before a large repair run,
compare compression with corrected output handling against the existing
initialization. For multi-million-token training, a CUDA GPU pod is the
predictable faster option with the existing training code; local AMD GPU
training would require a separate runtime setup and benchmark.

## Reproduction

From the repository root, using the dedicated local environment:

```bash
target/manifold-audit/env/bin/python -m pytest -q manifold-compression/test_audit_rotation.py
target/manifold-audit/env/bin/python manifold-compression/audit_rotation.py --model models/manifold-audit/Qwen3.5-2B --out manifold-compression/results/rotation-audit-20261024/qwen35-2b.json
target/manifold-audit/env/bin/python manifold-compression/audit_rotation.py --model models/manifold-audit/SmolLM2-1.7B-Instruct --out manifold-compression/results/rotation-audit-20261024/smollm2-1.7b.json
target/manifold-audit/env/bin/python manifold-compression/audit_rotation.py --model models/manifold-compression/distill-08b/continuation-500m/ckpt_500M --out manifold-compression/results/rotation-audit-20261024/recovered-500m-cache.json --cache-only
```

Four small-model tests passed, covering both architectures with tied and
untied output heads and nonuniform normalization gains. They verify exact
rotation, cached inference, and detection of the diagonal shortcut. The exact
environment is recorded in `requirements-lock.txt`; its CPU torch wheel comes
from https://download.pytorch.org/whl/cpu. `benchmark_repair.py` records the
disposable training timing procedure.
