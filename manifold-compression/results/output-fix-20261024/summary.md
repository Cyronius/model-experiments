# Output correction and width-cut comparison, 2026-10-24

The final-normalization shortcut has been removed from the default builder.
A full mixing matrix now preserves the coordinate interactions that the old
implementation discarded. The final RMS scale also multiplies the matrix;
the old implementation divided by it. This fixes the full-width rotation
invariance defect identified in the preceding [audit](../rotation-audit-20261024/summary.md).
It does not make a reduced-width model equivalent to its source.

## Experiment

Both Qwen3.5-2B and SmolLM2-1.7B-Instruct (Llama architecture) are projected
from residual width 2048 to 1792, a 12.5% cut. All attention heads, layers, and
MLP neurons remain. There is no healing, neuron selection, quantization, or
sampling-based tuning. Each model gets its own shared PCA basis. The three
arms within each model have exactly the same projected body and tied output
head; only the final norm changes.

Let P be the retained orthonormal basis, D the source final-norm gains, and
c the calibration mean RMS(projected)/RMS(original). The arms are:

- Legacy: diagonal(PᵀDP) / c.
- Dense, old scale: PᵀDP / c, isolating the diagonal shortcut.
- Dense, corrected scale: c PᵀDP, the new default.

The scale direction follows directly by equating the normalized reader on a
vector in the retained subspace. On actual compressed states the fixed mean
scale remains an approximation; its algebraic direction does not guarantee
lower KL on every sample. With rectangular P, the matrix represents only
retained-space gain interactions. It cannot restore discarded state directions
or output-head components outside that space.

| Model | Legacy KL | Dense / c KL | Corrected dense × c KL | KL reduction vs legacy | Top-1 agreement, legacy → corrected |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-2B | 0.618496 | 0.570050 | 0.577941 | 6.6% | 76.9% → 78.2% |
| SmolLM2-1.7B | 0.714482 | 0.482939 | 0.441041 | 38.3% | 68.9% → 77.0% |

| Model | Original parameters | Corrected parameters | Evaluation positions | Calibration tokens | Local wall time |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-2B | 1,881,825,088 | 1,649,864,256 | 78 | 4,051 | 200.8 s |
| SmolLM2-1.7B | 1,711,376,384 | 1,500,665,600 | 74 | 4,248 | 213.7 s |

Lower KL is closer to the untouched source. Neither corrected initialization
is at parity. Removing the diagonal shortcut helps both models on this sample.
Changing the RMS scale direction slightly worsens Qwen KL relative to the dense
old-scale arm (0.57005 → 0.57794) and improves SmolLM2 (0.48294 → 0.44104).
The default follows the algebra rather than selecting a scale on these eight
evaluation prompts.

Both saved FP32 checkpoints reload with zero logit difference at all evaluated
positions and no missing, unexpected, or mismatched weights. Both pass the
fixed-sequence cache check: mean KL 1.57e-11 for Qwen and 1.27e-10 for SmolLM2.
The complete test suite passes **20 tests**.
External Hugging Face loading of each real saved bundle also matches the
repository loader exactly on a separate six-token prompt; see
`qwen35-external-load.json` and `smollm2-external-load.json`.


Twenty calibration prompts and eight disjoint evaluation prompts come from
`smoke/prompts.jsonl`. Cached Qwen answers supply the continuation prefixes,
not correctness labels; they contain errors. Prefixes are capped at 256 tokens.
KL is forward KL from the corresponding untouched model, scored every 16th
assistant position. The JSON files record prompt IDs, source revisions,
download manifests, calibration counts, and the cached-answer checksum.
These are small within-model initialization diagnostics, not an architecture
ranking or a broad quality benchmark.

Four held-out prompts also have greedy generations capped at 96 new tokens.
Those short samples cannot establish completion or long-form loop rates.
The two dense arms have identical greedy outputs because their only difference
is a positive scalar on the logits. Their probability distributions differ.
The corrected Qwen answer still falsely describes Python alias assignment as
copying a list; it also gives an incorrect water density. Better KL alone is
therefore insufficient evidence of restored quality.

## Implementation and verification

`projection_models.py` provides serializable Qwen and Llama adapters. The
learned norm vector remains trainable, and the dense matrix can also be trained.
`build_projected.py` defaults to the corrected matrix and adds a full-width
PCA rotation test. Its existing local gain-fitting stage holds the initialized
matrix fixed; `polish_tail.py` includes both norm parameters during tail repair.
`common.py` loads the adapter through the local module.

The adapter adds 3,211,264 parameters at width 1792, about 12.25 MiB in FP32
or 6.125 MiB in BF16. At width 1024 it adds 1,048,576 parameters. Vocabulary
embeddings remain tied. `export_projected_stock.py` can fold the adapter into
an independent output head for stock architecture compatibility, at the cost
of duplicating vocabulary weights. This is algebraically equivalent, subject
to floating-point rounding. The export checks the FP32 fold before optional
BF16 storage rounding; GGUF conversion/runtime equivalence remains untested.

The Qwen real checkpoint exposed a serialization issue absent from freshly
initialized tiny models: Transformers saves native converted keys with a
`model.language_model` prefix, then skips the conversion for unregistered custom
classes. The adapter explicitly registers the native mapping. A regression
fixture reproduces that pretrained prefix, including external AutoModel loading.
The probe now fails on unmatched checkpoint weights, numerical mismatch, or a
failed cache check rather than merely recording a bad result.

The test suite covers exact full-width rotation, diagonal-error detection,
rectangular RMS scaling, tied and untied heads, both architectures, width
changes, HF local/external loading, BF16 parameter persistence, gradient flow,
stock-head materialization, and the actual production builder's rotation path.
The numerical gate also rejects non-finite logits. See `tests.txt`.

Fresh corrected FP32 HF directories:

- `models/manifold-compression/output-fix-qwen35-1792/`
- `models/manifold-compression/output-fix-smollm2-1792/`

These are fresh modest-cut initializations. They are not repairs of the
recovered 500M-token student. That trained student's coordinates and weights
cannot be corrected with a matrix from an unrelated new PCA basis.

In-repo loading uses `projection_models.load_causal_model(path)`. External
callers can review the bundled module and use
`AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`.
The tested environment is torch 2.14.1+cpu / Transformers 5.14.1; dependencies
are recorded in `requirements-lock.txt`. No installed library files were edited.

## What to do next

The correction removes an avoidable initialization defect. Remaining loss at
a modest width cut calls for a controlled recovery experiment before another
aggressive rebuild or long corpus-distillation run.

1. Keep the modest width cut and all MLP neurons for the first healing pilot.
   Compare corrected and legacy initialization with the same training examples,
   token budget, and held-out evaluation. This separates recoverability from
   compression ratio and avoids changing several variables together.
2. Use complete assistant responses and their ending tokens for the repair
   objective, alongside teacher KL. Include code, instructions, and multi-turn
   tasks. Ensure packing/truncation does not systematically remove answer ends.
3. Evaluate full responses long enough to observe completion and repetition;
   measure instruction-following and factual/task correctness against each
   model's own stock baseline. Short KL reductions do not satisfy parity.
4. If the modest-cut models do not recover, localize loss by projection site
   and depth, including a control with an independently folded output head.
   That control isolates retained-space tied-head limitations. Only then test
   stronger cuts, more calibration data, or larger source models.

This experiment does not test healing, calibration-size scaling, or whether
larger models are easier to compress. The full-width invariance defect occurs
in both Qwen and Llama; it is not specific to Qwen's hybrid attention.

All work ran on the local Ryzen CPU. RunPod spending: **$0**. A GPU would be
useful for a multi-million-token healing pilot after its data and full-response
evaluation are ready. The earlier CPU throughput measurement was for the
smaller recovered student and should not be extrapolated to these wider models.

## Reproduce

From the repository root:

```bash
target/manifold-audit/env/bin/python -m pytest -q manifold-compression/test_projection_models.py manifold-compression/test_audit_rotation.py

target/manifold-audit/env/bin/python manifold-compression/probe_output_fix.py --model models/manifold-audit/Qwen3.5-2B --out manifold-compression/results/output-fix-20261024/qwen35-2b-1792.json --save-model models/manifold-compression/output-fix-qwen35-1792 --max-new 96

target/manifold-audit/env/bin/python manifold-compression/probe_output_fix.py --model models/manifold-audit/SmolLM2-1.7B-Instruct --out manifold-compression/results/output-fix-20261024/smollm2-1.7b-1792.json --save-model models/manifold-compression/output-fix-smollm2-1792 --max-new 96

PYTHONPATH=manifold-compression target/manifold-audit/env/bin/python manifold-compression/results/output-fix-20261024/verify_external.py
```
