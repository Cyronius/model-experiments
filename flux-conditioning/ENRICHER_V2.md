# FLUX enricher v2 on curated DiffusionDB data, 2026-11-14

Question: with real curated data, does the enricher make a plausible enrichment for a base
prompt it has never seen? Answer: yes, with weak dependence on the base. Plan: `.claude/plans/archive/flux-enricher-v2.md`. Design decisions:
`.claude/plans/flux-enricher-decisions.md`.

## Files

- `enricher_v2.py`: `data`, `train`, `score`.
- `v2_images.py`: padded base targets, samples on base padding, nearest teacher prompts.
- Outputs: `artifacts/flux-conditioning/enricher-v2/` (ignored): `data/` (PCA rows), `run1/`
  (checkpoints, logs, scores, images).

## Data

Pod export `pod-20261112/pairs-gpu-fp32` (GPU FP32 compute, BF16 storage).
25,245 groups and 332,023 enriched prompts after the limits (context 64 rows, window 32 rows;
83 + 70 items dropped). Split by group-id hash: train 22,718 groups, validation 1,295, test 1,232.
1,585 groups come from OpenRouter LFM 2.5 curation; the rest from Qwen3.5-9B.

Whitened PCA of normalized train rows: 256 directions hold 95.5% of the variance
(64: 90.2%, 512: 97.6%, 1024: 99.0%).

## Run 1

RTX 4090 pod, torch 2.9.1. Denoiser: 8 layers, dim 512, x0 prediction, rectified flow, plus a
2-layer length head; 32.3M parameters. Batch 256, lr 3e-4 one-cycle, 86,196 steps (110 min,
13.1 steps/s). Regression baseline: same network, 28,732 steps. Sampling: 16 steps, 8 samples per base,
length drawn from the length head.

Fix found in the CPU smoke test: window rows also need a position counted from the window end.
Without it the model could not place the 9 template rows, and own-group fit was 5.3 D.

| Split | Groups | Enricher own/D | Mean own/D | Regression own/D | Held-out teacher own/D |
|---|---|---|---|---|---|
| Train (300 sampled) | 300 | 0.764 | 0.813 | 0.611 | |
| Validation | 1,295 | 0.797 | 0.814 | 0.661 | |
| Test | 1,232 | 0.794 | 0.811 | 0.659 | 0.787 |

| Test metric | Enricher | Mean | Regression | Held-out teacher |
|---|---|---|---|---|
| Specific fraction (ratio <= 0.85) | 0.184 | 0.000 | 0.496 | 0.392 |
| Median specificity ratio | 0.921 | 1.163 | 0.851 | 0.879 |
| Median distinct nearest (of 8) | 5 | 1 | 2 | |
| Length MAE (rows) | 1.50 | | | |

"Held-out teacher" scores each real teacher enrichment against the other enrichments of its group.
It is the score of a perfect sampler that writes a new valid enrichment.

- Training rows loss fell from 1.26 to 0.16; validation rows loss from 1.26 to 0.177, still falling slowly.
- The length head overfit: validation cross-entropy rose from 1.87 (step 300) to 2.89.
- LFM-curated groups score slightly worse (test own/D 0.835 against 0.792).

## Gates (set in the plan before the run)

1. Beats both baselines on own-group fit: **fail.** Regression wins (0.659 against 0.794).
2. Specific for most test bases: **fail.** 18% of test bases.
3. Diverse: **pass.** Median 5 different nearest enrichments of 8.

The calibration row shows that gates 1 and 2 were set badly. A real held-out enrichment also loses
to regression (0.787 against 0.659), because a mean sits close to every point of a cloud. A real
enrichment is specific for only 39% of bases. So the useful reading is against the held-out teacher:
own-group fit equals a real enrichment (0.794 against 0.787), but group specificity is about half
(18% against 39%). The samples look like real enrichments in general, with too little dependence on the base.
Later runs compare against the held-out teacher row, not against regression.

## Images

Six test bases, three samples each, rendered on CPU with the diffusers path (4 steps, 512 x 512, noise A,
about 55 s per image). Each sample keeps the base prompt's padding rows. The teacher image uses the
nearest teacher enrichment by descriptor. Sheet: `run1/images/sheet.png` (each row: base, then sample
and nearest teacher for s0, s1 and s2).

| Base | s0 | s1 | s2 |
|---|---|---|---|
| high detail portrait man green knight necromancer ... | near base | closer, more dramatic | framed card art |
| stylized exterior architecture | realistic street photo | colorful cartoon facade | black and white, night |
| overhead view surveillance | fisheye office with people | neon city from above | abstract blue rings (fails) |
| emily blunt in futuristic power armor | near base | stronger city lights | space scene with ships |
| medieval oil painting | figure holds an object | seated king on a throne | flower still life |
| a middle aged man | near base | hat, desert, rugged | office, smiling |

First reading (Claude; the user's check decides):

- 17 of 18 samples give a coherent image that keeps the base subject or style. 1 fails (abstract rings).
- About 11 add a clearly visible detail. About 4 change the scene only a little, and 3 are close to the base.
- Samples differ from each other for every base. That agrees with the diversity result.
- The image a sample gives often differs from its nearest teacher image, as in the prototype.
  The descriptor metric does not find the enrichment the image shows.
- "Medieval oil painting" to "flower still life" keeps the style but drops the scene. The base prompt
  has no subject, so this is a valid enrichment of that text.

## Run 2: classifier-free guidance

Question: does the enricher ignore the base because the base signal is weak at sampling time?
Classifier-free guidance (CFG) tests this. Training drops the base rows for 10% of steps
(`--ctx-drop 0.1`). Sampling runs the model with and without the base and extrapolates:
`free + w * (with_base - free)`. All other settings are the same as run 1 (same data, network, 86,196 steps,
12.9 steps/s on a secure RTX 4090). Outputs: `run2-cfg/`.

Test split, 1,232 groups:

| Guidance w | own/D | Specific fraction | Median specificity | Distinct (of 8) |
|---|---|---|---|---|
| 1 (no guidance) | 0.800 | 0.183 | 0.920 | 5 |
| 1.5 | 0.788 | 0.360 | 0.883 | 5 |
| 2 | 0.791 | 0.444 | 0.862 | 5 |
| 3 | 0.811 | 0.544 | 0.840 | 4 |
| Held-out teacher | 0.787 | 0.392 | 0.879 | 14 |
| Mean | 0.810 | 0.000 | 1.161 | 1 |

Validation (1,295 groups) agrees within 0.02 on every value. Train (300 sampled groups) at w = 1
already reaches 0.433 specific. So run 1 learned the base dependence for training bases but did not
carry it to new bases at w = 1.

Gates (the rule against the held-out teacher, set after run 1):

| w | Fit within 10% of teacher | Specific >= 75% of teacher | Distinct >= 4 |
|---|---|---|---|
| 1 | pass | **fail** (47%) | pass |
| 1.5 | pass | pass (92%) | pass |
| 2 | pass | pass (113%) | pass |
| 3 | pass | pass (139%) | pass (at the limit) |

- Dropping the base for 10% of steps costs nothing: w = 1 equals run 1 (0.183 against 0.184 specific).
- w = 2 is the best point by the vectors: teacher-level fit and more specific than a real enrichment.
- w = 3 moves fit toward the mean (0.811) and loses diversity. Specificity above the teacher is not
  automatically good. Guidance can exaggerate the base signal, and the descriptor metric rewards that.
  The images decide between 1.5, 2 and 3.
- The length head is unchanged (MAE 1.49; validation cross-entropy 2.87, overfit as in run 1).

### Run 2 images

The same six test bases and render settings as run 1, 3 samples each at w = 1.5, 2 and 3 (54 samples,
no teacher renders). Sheet: `run2-cfg/images/sheet.png` (each row: base, then s0-s2 at w = 1.5, 2, 3).

| Base | w = 1.5 | w = 2 | w = 3 |
|---|---|---|---|
| green knight necromancer | green glow; armor; full body | near base; card frame with garbled text; green lightning | near base; dark forest arch; near base |
| stylized exterior architecture | building in a big vine tree; modern flats; pink glass block | cafe with plants; skyscraper street; classical block, wet street | flat modern facade; white parametric block; sunset street |
| overhead view surveillance | roof garden; doctor looks up at camera; car park | street; courtyard with tables; fisheye with CCTV timestamp | park; junkyard; office from above |
| emily blunt in power armor | near base; daytime city; purple armor on white | intense close-up; night city; hangar | flying debris; near base; starry city |
| medieval oil painting | horned barbarian; framed icon; floating bubbles over fields | man in red robe; enthroned woman in red; peasant group outdoors | seated knight; two queens; red ball creature with faces |
| a middle aged man | dark dramatic light; black hair; near base | older, wrinkles; black hair; older, red eyes | stern; very old; glasses |

First reading (Claude; the user's check decides):

- All 54 samples give a coherent image that keeps the base subject or style. Run 1 had 1 failure in 18.
- About 7 of 54 are close to the base (run 1: 3 of 18). The rest add a visible detail or scene change.
- w = 3 shows no oversaturation or broken images. The exaggeration risk from the vectors is not visible here.
- The images do not separate w = 1.5, 2 and 3. Single images show plausibility and variety, not group
  specificity, so the choice rests on the vectors: **w = 2** (teacher-level fit, full diversity).
- Two odd but valid samples: "floating bubbles" (w = 1.5) and "red ball creature" (w = 3) on the
  medieval base. Both match real enrichments of that group ("floating big bubbles").

## Limits

18 images from one noise file; one viewer; no blind rating. The teacher replaces the compact encoder.
No NPU, Phlegm, size or speed result.
