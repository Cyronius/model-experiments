# Real teacher export control, 2026-11-09

All 12 BF16 target exports exactly match the pinned upstream reference text
methods plus the published context projection. Maximum absolute conditioning
error is zero for every real and padding row. Token IDs, masks and text position
IDs also match exactly. This validates the export boundary; no student, generated
image, NPU route or performance comparison was tested.

| Cases | Full input token counts | Export/reference result |
|---|---|---|
| control-fox, counts, attributes, spatial | 18, 32, 28, 36 | exact |
| text, constraints, face, detail | 32, 35, 33, 32 | exact |
| long-bindings | 193 | exact |
| boundary-511, boundary-512, boundary-513 | 511, 512, 513 | exact; only 513 truncates |

Teacher: `black-forest-labs/FLUX.2-klein-4B` at
`e7b7dc27f91deacad38e78976d1f2b499d76a294`. All 14 selected original checkpoint
files were checked against pinned Hub LFS hashes/Git blob IDs. The user's existing
`C:/Users/josha/.flm/models/FLUX.2-klein-4B-NPU2` is the packed runtime bundle;
the original teacher components were downloaded separately (15,812,009,114 bytes).
The user authorized downloading the original checkpoint if needed.

Reference: Diffusers commit `c60830ee365d520ab52b110dda562dd26f7b4d7f`, executing
`Flux2KleinPipeline._get_qwen3_prompt_embeds` and `_prepare_text_ids` extracted by
AST from the pinned source. Their bodies are unmodified; the standalone CPU helper
returns the model's parameter device. The reference uses `Qwen3ForCausalLM`, while
the exporter uses the headless `Qwen3Model`; both use the same installed
Transformers implementation and eager attention. The reference applies the
published projection through `nn.Linear`; the exporter uses tap concatenation
and `F.linear`. Neither path ran the denoiser or the VAE.

Windows build 26200, Python 3.11.9, torch 2.13.0+cpu, transformers 5.14.1,
safetensors 0.8.0, huggingface_hub 1.20.1. CPU reports AMD Family 26 Model 36,
24 logical processors, approximately 94.1 GB physical memory. Eight torch intra-op
threads and 12 inter-op threads; BF16, no cache, no network during inference.
This is correctness evidence, with no timing or hardware speed claim.

Local evidence root: `C:/code/model-experiments/artifacts/flux-conditioning/`.
The source files contain no weights or bulky captures.

| Artifact relative to that root | SHA-256 |
|---|---|
| `teacher/verified-upstream.json` | `3ae69c4397ff6db52c42c72ce37719b6d7bda740dcdb961928c871733ceae224` |
| `teacher-control-20261109/cases.jsonl` | `b7d7c1bec7bda4b77823064c9e7786d93761925e7fe69b2a5da9401918eaaa1c` |
| `teacher-control-20261109/targets-bf16/manifest.json` | `4b8eba82bfc733c082d03c1610d0d8bffbac900d7853c800b169a3f2cb4a18bb` |
| `teacher-control-20261109/reference-bf16/report.json` | `0f3fb825b3a93562120a7e7aea562601d654b55ccef66bbf9e5f963bf86f2886` |

The report also records exporter hash
`969a74c092a7bf61c3591adc40075b83381d9ab76b6dce542379fafe0cc6bb8f`
and comparison-script hash
`d7afa75695385b9fe378346781941d2f428ff60782b2b9c1f877fbe498e33054`.
The pipeline reference source SHA-256 is
`8afefd06451d2bed0f34b59e930e60383c6a47fc8cebbec74ea30bfcdc7b79a9`.

`teacher-control-20261109/commands.json` retains the exact two inference commands;
`export.log` and `reference.log` retain raw stdout/stderr. Source snapshots live
under the run's `source/`. The reusable downloader is derived from the successful
local download script; its CLI was checked, while the original local script
performed this download. No packages were installed.

Focused harness checks: 17 pass, including tiny real Qwen3 export, provenance
bindings, tokenization boundaries and source-method extraction. The experimental Phlegm input is now implemented; next run its
live native capture/replay gate, then compare teacher artifacts with exact noise. Define student training resources and image-quality
acceptance before the architecture bakeoff.

## Parallel worker and runtime input, 2026-11-10

Trashcan (Linux, Xeon E5-2697 v2, 64 GB system RAM) completed all twelve FP32
controls exactly, in a two-case trial followed by ten remaining controls.
The latter ran at nice 10, 12 CPU threads: export 300.314 s, reference 336.378 s,
sampled peak RSS 22,352,650,240 and 22,234,296,320 bytes respectively. These are
development stage times including loading/provenance work, not benchmarks.
No D700 GPU or NPU inference was used. The worker is idle after completion.

Remote root: `/home/cyrus/Work/flux-conditioning-worker`. Remaining-control
evidence is `artifacts/remaining-controls-20261110/`; source/command provenance,
exit codes and stage samples are in `status.json`. Reference report SHA-256:
`7d80ff393fc0dc941fb30b70e87fd20f6a0e6836f456c32290d56630d0b3d2fb`;
target manifest SHA-256:
`e2d21f6d924d0d4ef6645e95a75d2bf5c99999f5bd76ebc41774ef93f828b3d8`.
The first two-case report SHA-256 is
`46ae50e38be0dbcbff940ed8dcbdfdc8da363cbe0359b82c01733f301b8e6ecd`.
FP32 results remain separate from the Windows BF16 targets.

The Phlegm runtime input passes 23 focused Python checks on both Windows and a
transferred Linux snapshot, plus six core image and four CLI checks. Twelve real
Windows BF16 artifacts preserve bits; twelve Linux FP32 artifacts match independent
Torch BF16 conversion, with prompt IDs/masks checked. Companion adapter outputs
now retain IDs/masks for runtime ingestion; all 17 harness checks pass.
Linux snapshot/report are in `validation-20261110-final/` on the worker; small
local copies and hashes are in Phlegm's ignored
`.plans/conditioning-input-20261110/`. No NPU image gate, trained student,
student-quality threshold or performance result is established.
