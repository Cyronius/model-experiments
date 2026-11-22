# Short text sequences into the Klein denoiser, 2026-11-17

Question: the denoiser always gets 512 text rows, and about 490 of them are padding. Can it get fewer
rows without retraining? Fewer rows means fewer tokens per denoiser step on the NPU.

Script: `pad_trunc_test.py`. Outputs: `artifacts/flux-conditioning/pad-trunc/` (ignored), with
`sheet-noise-a.png`, `sheet-noise-b.png`, `results.json`.

## Method

10 prompts from the encoder probe (`ENCODER_PROBE.md`), 2 noise files, 512 x 512, 4 steps, CPU
diffusers path. Text positions are `0..L-1` for any length L, so real rows keep their positions.

| Version | Text rows |
|---|---|
| teacher 512 | 512, stock |
| meanpad 512 | real rows + 1 mean padding vector repeated to 512 (noise A only, from the probe run) |
| meanpad 128 | real rows + the same mean vector to 128 |
| real rows only | 15-24 rows, no padding |
| probe, real rows only | probe rows from Qwen3-Embedding-0.6B, no padding |

## Results

Mean pixel MAD (mean absolute pixel difference, 0-255) against the teacher image with the same noise:

| Version | MAD |
|---|---|
| meanpad 512 | 31.4 |
| meanpad 128 | 31.3 |
| real rows only | 45.7 |
| probe, real rows only | 55.0 |
| new noise (teacher A against teacher B) | 50.8 |

Images (Claude's reading; the sheets are the evidence):

- **meanpad 128 equals meanpad 512.** On noise A it is near pixel-identical for 7 of 10 prompts.
  Counts (3 apples), text (OPEN) and Einstein's likeness survive on both noises.
- **Real rows only damages content.** On noise B: 2 apples instead of 3, "OYEN" instead of "OPEN",
  Einstein without glasses and less like him. On noise A: Einstein's likeness is lost, the
  knight's helmet and the man's clothes change. Images stay coherent.
- The probe with real rows only shares the real-rows-only damage, on top of its own changes.

## Reading

The padding rows do work in the denoiser, but 128 rows are enough. The likely cause is attention: with
no padding, the text rows get the attention that padding rows took (a "sink"), so their influence changes.
This is not tested.

Cut to 128: tokens per step at 512 drop from 1,536 to 1,152. Estimate from operation counts: about 27%
less denoiser compute per step. On the NPU (0.797 s per step, Phlegm spec) that is about 0.58 s per
step, if time follows compute. Not measured; the native runtime needs kernels for the new sizes.

## Run 2: 64 and 32 rows

Same prompts, noises and settings. Sheets: `sheet2-noise-a.png`, `sheet2-noise-b.png` (teacher 512,
meanpad 128, 64, 32, real rows only).

| Text rows | MAD | Tokens per step at 512 | Denoiser compute per step (estimate) | Content |
|---|---|---|---|---|
| 512 (stock) | 0 | 1,536 | 10.2 TFLOP | |
| 128 | 31.3 | 1,152 | 7.5 (-27%) | equal to meanpad 512 |
| 64 | 34.9 | 1,088 | 7.0 (-31%) | counts, text, Einstein (noise A) kept; 2 of 10 compositions drift on noise B |
| 32 | 38.7 | 1,056 | 6.8 (-33%) | counts and text kept; Einstein's likeness lost on noise A |
| real only (15-24) | 45.7 | about 1,044 | 6.7 (-34%) | 2 apples, "OYEN", Einstein lost |

The damage grows step by step as rows are removed; there is no sharp edge. 64 rows give 4 more points of
compute than 128 for a small drift. At 1024 x 1024 (4,096 image tokens) the gain from 512 to 128 rows
is only about 8%.

## Limits

10 prompts, 2 noises, one viewer, pixel MAD is a coarse measure. Tested lengths: 512, 128, 64, 32, real only.
Prompts longer than 128 tokens need more rows; the cut is "real rows, padded up to 128".
