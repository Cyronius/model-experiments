# Compact FLUX conditioning: encoding control

Encoding-only scaffolding for the agreed Phlegm experiment. No student is trained,
and no image-quality or speed result exists. Keep the Klein 4B denoiser frozen.
VAE distillation, enrichment and smaller image denoisers are separate experiments.

Training and checkpoint work live here in `model-experiments/flux-conditioning`.
Phlegm will run the eventual student and frozen denoiser. Its native Klein route
now has an experimental local projected-conditioning input, checked offline and
awaiting live NPU acceptance. This harness does not register a new model. The matching design and current native
limitations live in Phlegm's `specs/open-engine/flux-conditioning-distillation.md`
and `specs/open-engine/flux2-klein-native.md`.

## Target contract

Teacher: `black-forest-labs/FLUX.2-klein-4B`, declared revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`. Use the published teacher for this control.
The exporter takes a local diffusers-layout checkpoint; it never downloads weights.
Hashes of the encoder weights/configs, tokenizer files and projection shard(s)
identify the actual files, independently of the declared revision label. Local
files are not proven to match the Hub revision by their path or config alone.

The reference semantics are one user message, generation prompt, thinking disabled,
right padding/truncation to 512 tokens, and Qwen3 intermediate hidden states 9/18/27
concatenated in that order. Apply `context_embedder` (including bias if present)
to obtain `[512, 3072]` projected conditioning. Do not apply the projection again
when integrating with Phlegm. Keep **all** padding rows: the denoiser sees 512
conditioning positions. Text IDs are `[0, 0, 0, position]`.

Sources: [pinned pipeline](https://github.com/huggingface/diffusers/blob/c60830ee365d520ab52b110dda562dd26f7b4d7f/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py),
[teacher snapshot](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294),
and Phlegm `open_kernels/klein_pipeline.py` at
`7742985b53e464a1a8171321fe4e33281905ec8a`. The harness records its own hash and
package versions and fails if the tokenizer differs from the fixed native template.
The first real teacher comparison passes all 12 cases; see [results](RESULTS.md).
Both export paths use the same installed Transformers Qwen3 implementation; this
checks exporter semantics. Image quality and Phlegm execution remain open.

The first control has 512 denoiser-facing positions. Internal width/slot mapping,
recurrence and ternary QAT are later controlled ablations. In particular this code
does not choose a 256-to-512 slot mapping or a student architecture.

## Commands

Run in a dedicated environment. `requirements.txt` gives compatibility bounds;
export manifests retain the exact installed versions. BF16 is the default teacher
arithmetic; use a distinct output directory for the optional FP32 diagnostic.

```powershell
python -m pytest -q flux-conditioning/test_harness.py
python flux-conditioning/download_teacher.py --output artifacts/flux-conditioning/teacher
python -c "from pathlib import Path; p=Path('flux-conditioning'); Path('artifacts/flux-conditioning/cases.jsonl').write_text((p/'heldout.jsonl').read_text()+(p/'boundary.jsonl').read_text(), encoding='utf-8')"
python flux-conditioning/harness.py export --model artifacts/flux-conditioning/teacher --cases artifacts/flux-conditioning/cases.jsonl --output artifacts/flux-conditioning/targets-bf16 --threads 8
python flux-conditioning/compare_reference.py --model artifacts/flux-conditioning/teacher --targets artifacts/flux-conditioning/targets-bf16 --output artifacts/flux-conditioning/reference-bf16 --threads 8
```

The downloader selects about 15.8 GB: the original text encoder, tokenizer and
transformer weights/config. It verifies LFS SHA-256 or Git blob IDs at the fixed
Hub revision, and pins both upstream source files by commit and SHA-256. It does
not fetch the VAE or a complete image pipeline. Original weights are needed;
OpenFlowLM's `FLUX.2-klein-4B-NPU2` BFP-packed bundle is a runtime asset, whose
packed GEMMs cannot supply the original BF16 teacher tensors. Existing downloads
can resume. Use fresh target/reference output directories for another validation.

When `verified-upstream.json` is present in the teacher directory, the exporter
checks all used source hashes against it and records that verification. The
reference comparison rehashes those files, executes the two pinned upstream
static text methods without importing diffusers, and independently loads the
projection into `nn.Linear`. The CPU device helper returns the model parameter
device. No harness tokenization, tap concatenation or projection routine is used
by that path. Its report checks exact equality, including masks and positions.

The encoder is loaded in full, without its language-model head. Budget memory for
the full Qwen3-4B model and its hidden states, plus the projection. Exporting even
this small held-out set can be slow on CPU. Use `--device cuda` only on a machine
with sufficient memory; no training or cloud instance is launched here.

`heldout.jsonl` is a **draft held-out smoke set**, not training data or an agreed
acceptance suite. It covers binding, counts, text, constraints, faces, detail and
a longer description. `boundary.jsonl` adds tokenizer-verified 511/512/513-token
diagnostic prompts; the exporter enforces their expected lengths. These repetitive
boundary controls make no image-quality claim. Export records untruncated token
counts. The saved seed is
image-evaluation metadata and does not affect deterministic text encoding.
Define larger train/validation/test corpora, budgets and acceptance thresholds
before the architecture bakeoff; the reader rejects duplicate prompts across splits.

Each target file contains `input_ids`, `attention_mask`, `text_ids`, and
`projected_conditioning`, plus a case hash. The complete export's `manifest.json`
records case and artifact hashes. Export refuses existing output directories; an
interrupted directory has no complete manifest and should be retained or moved
aside before retrying. Bulky outputs belong under ignored `work/`.

Use `save_prediction` in a student adapter to write each `[512, 3072]` result to
the corresponding case filename, bound to that exact target file:

```python
from pathlib import Path
from harness import save_prediction

save_prediction(Path("work/teacher-bf16/control-fox.safetensors"),
                student_output, Path("work/student/control-fox.safetensors"))
```

```powershell
python flux-conditioning/harness.py evaluate --targets flux-conditioning/work/teacher-bf16 --predictions flux-conditioning/work/student --split test --output flux-conditioning/work/student-metrics.json
```

Evaluation rejects changed targets, wrong reference bindings, invalid positions,
shape errors and nonfinite values. Reports show all positions, real input tokens,
and padding separately, with hashes of both target and prediction. These are
diagnostics: there is deliberately no conditioning-error pass threshold.

## Phlegm integration and image-quality gate

The twelve Windows BF16 and twelve Trashcan FP32 teacher exports agree exactly
with their same-dtype pinned reference methods. Phlegm now accepts a local
`--projected-conditioning FILE` for its Klein backend, bypassing native text
encoding and `ctx_emb`. It checks the protocol, all 512 positions, finite
BF16/F16/F32 values and exact prompt IDs/mask, then restores the BF16 input at
each denoising step. Use the original prompt with that flag. Runtime output
records the consumed artifact hash. HTTP JSON cannot select this local file.

Current `save_prediction` and reference outputs retain `input_ids` and
`attention_mask` as well as `text_ids`. Older prediction files lacking them are
numerical evaluation artifacts only. Targets already contain all four required
tensors. The projected mode needs the full transformer/VAE checkpoint, tokenizer,
model index and compatible NPU kernels, but no original text-encoder weights.
The downloaded teacher subset omits the VAE/model index and cannot generate images.

Phlegm's `open_kernels/flux2_conditioning_replay.py` prepares a live gate using
native captured conditioning: exact baseline/capture/replay, zero-input sensitivity,
and exact request restoration with saved identical noise. It needs the normal
text encoder as well. This gate awaits complete image assets. The existing
Windows IRON environment was verified on 2026-11-11 with Python 3.13,
a successful pyxrt device open and a passing official SAXPY NPU computation.
Activate `. C:\dev\mlir-aie\iron_env.ps1` in PowerShell. The earlier
missing-installation report was a sandbox access error. After native replay
passes, compare projected teacher inputs under the existing
native numerical gate. PyTorch targets and native compensated-BFP conditioning
must not be assumed bit-identical. See the Phlegm native spec for the command.

For image comparisons, retain exact initial-noise files and their hashes; seeds
alone do not identify noise across PyTorch and Phlegm's NumPy generators. Record
Phlegm revision, teacher/conditioning hashes, kernel hashes, resolution, steps,
noise hash and the complete hardware/software stack per case. Compare student
and teacher conditioning with the **same frozen Phlegm denoiser**, exact noise,
step schedule and VAE, then score the case's explicit criteria. Use independent
reference images to distinguish inherited native arithmetic drift from student
effects. Define quality thresholds before examining student outputs.

Do not use projected targets as `prompt_embeds` on an unchanged diffusers pipeline:
that API expects the unprojected 7680-wide tensor. Live NPU integration acceptance and image-quality
comparisons remain follow-ups. Define the student architecture and
training/image-quality acceptance before launching the first training experiment.
