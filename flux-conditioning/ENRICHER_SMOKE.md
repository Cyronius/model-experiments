# Dumb enricher smoke test, 2026-11-10

Question: can a tiny diffusion model write the conditioning rows of an appended
prompt detail, using vectors only, so that frozen FLUX.2 Klein 4B draws the detail?
Answer for this toy set: yes. Plan and design decisions are in
`.claude/plans/archive/flux-enricher-smoke.md` and `.claude/plans/flux-enricher-decisions.md`.

The teacher stands in for the compact encoder. The data is a closed toy set:
24 subjects x 16 appended details. CPU only; no NPU, no Phlegm, no speed claim.

## Files

- `enricher_smoke.py`: `cases`, `train`, `score`. Tiny transformer enricher (5.6M parameters).
- `image_check.py`: `noise`, `stage1`, `render`, `sheet`. Renders `[512, 3072]` conditioning
  through the Klein transformer with `context_embedder` replaced by identity (it has no bias).
- `stage3_cond.py`: joins base-prompt rows with sampled enricher rows.
- Outputs: `artifacts/flux-conditioning/enricher-smoke-20261110/` (ignored).

Environment: Python 3.11.9, torch 2.13.0+cpu, transformers 5.14.1; image path in
`artifacts/flux-conditioning/venv-diffusers` (diffusers 0.42.1, huggingface_hub 1.33.0).
VAE and pipeline configs come from the pinned teacher revision; the VAE weights
match the Hub LFS SHA-256 (`image-pipeline/verified-upstream.json`). 4 steps,
guidance 1.0 (distilled model), 512 x 512, bf16, 12 threads: about 68 s per image.

## Stage 0: append premise

All 384 enriched prompts share their base prompt's tokens up to the template end.
Their conditioning rows are bit-identical over that prefix. A re-export of
`control-fox` is bit-identical to Codex's 2026-11-09 target.

## Stage 1: teacher-only image checks (same noise)

| Variant | Pixel MAD vs true |
|---|---|
| Base, padding -> per-prompt mean | 21.6 / 20.1 |
| Base, padding -> other prompt's padding | 20.8 / 21.2 |
| Base, padding -> one global mean vector | 18.7 / 20.9 |
| Enriched, padding -> base prompt's padding | 9.5 / 17.3 |
| Enriched, padding -> per-prompt mean | 20.5 / 25.7 |
| Calibration: same prompt, other noise | 37.1 |
| Calibration: base vs enriched, same noise | 101.1 |

Values are for noise A / noise B. Padding replacements keep the scene and change
only pose and small details. Enriched content rows with base padding still show
the enrichment (dusk, cedars, wet fur). Conclusion: the enricher only needs the
changed text rows; padding rows carry little meaning.

## Stage 2: learned enricher (vector gates)

`D` = median distance between the 16 teacher windows of one subject. Gate set in
advance: median nearest distance <= 0.35 D. A plain average sits at about 0.64 D.

| Run | Train | Held-out cells | Held-out subjects | Distinct (of 16) |
|---|---|---|---|---|
| Velocity, full width | 1.15 | 1.14 | 1.16 | 8 |
| x0, full width | 0.80 | 0.81 | 0.80 | 7–9 |
| Velocity, PCA-256 | 0.68 | 0.67 | 0.71 | 5–8 |
| x0, PCA-256 | 0.66 | 0.65 | 0.71 | 7–8 |
| Regression, PCA-256 | 0.53 | 0.53 | 0.55 | 1 |
| Velocity, whitened PCA-256, 10k steps | 0.69 | 0.69 | 0.70 | 5–9 |
| **x0, whitened PCA-256, 10k steps** | **0.42** | **0.42** | **0.45** | **7–9** |

- Full-width diffusion failed because the 256-wide body cannot see 3,072-wide noise.
  Its samples mixed different details row by row.
- Latent diffusion in whitened PCA space fixed most of that. x0 prediction worked
  better than velocity prediction.
- The best run misses the 0.35 D gate. It beats regression and stays diverse.
- Held-out cells: 4 of 352 samples land nearest to the held-out detail (uniform: 44).
  The model reproduces trained subject/detail pairs; it does not compose new ones.

## Stage 3: sampled rows to images (best run, noise A)

Three samples each for a training subject (red fox), a subject with held-out
cells (barn owl) and a fully held-out subject (windmill).

| Subject | Sample image shows | Nearest teacher by distance |
|---|---|---|
| Red fox | pixel art sprite | pixel art sprite |
| Red fox | inside a glass snow globe | snow globe |
| Red fox | thick fog | thick fog |
| Barn owl | watercolor painting | pixel art sprite |
| Barn owl | neon lights | neon lights |
| Barn owl | glass case, no snow (partial snow globe) | watercolor |
| Windmill | thick fog | snowy forest |
| Windmill | snow | snowy forest |
| Windmill | pixel art | watercolor |

8 of 9 samples give a clean, recognisable enrichment; 1 is partial. All 3
windmill samples work, although no windmill enrichment was in training.
The distance-nearest teacher often differs from what the image shows, so the
vector gate underrates the model. Use image checks, or a better vector metric,
for later runs. Contact sheets: `stage3-sheet-*.png` in the output folder.

## Limits

Nine images from one noise file; no blind rating. A closed set of 16 details;
no open-domain claim. The teacher replaces the encoder. No NPU, size or speed result.
