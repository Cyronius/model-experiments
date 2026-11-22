# Compact encoder step 0: linear probe on Qwen3-Embedding-0.6B, 2026-11-16

Question: how close is a frozen Qwen3-Embedding-0.6B, plus one linear map, to the Klein teacher
(Qwen3-4B taps 9/18/27 + `context_embedder`)? Plan: `.claude/plans/flux-compact-encoder.md`.
Script: `encoder_probe.py`. Outputs: `artifacts/flux-conditioning/encoder-probe/run1/` (ignored).

## Method

- Same token IDs as the teacher (checked), same chat template, causal. Rows 0-2 are constants and
  are not fitted. Targets are the teacher's real rows, normalized per dimension like the enricher.
- Ridge regression from hidden states of layers 4, 7, 10, 14, 18, 21, 24, 28 (8,192 features)
  to the 3,072-wide rows. Lambda chosen on validation.
- Data: enricher-v2 groups (base prompt + 3 enriched prompts per group), same group split.
  Train 2,000 groups (8,000 prompts, 121,740 rows), validation 150, test 300 (18,245 rows).
  CPU, 41 min for train features.

## Vectors (test)

| | R2 | Median row cosine |
|---|---|---|
| Probe, 8 layers, lambda 1e-3 | 0.927 | 0.969 |
| Probe, layers 7/14/21 only (validation) | 0.915 | |
| Teacher through PCA-256 (enricher accuracy) | 0.946 | 0.981 |

## Images

10 prompts, 3 renders each, noise A, CPU diffusers path: teacher; teacher with one mean padding
vector; probe with the same mean padding. Sheet: `run1/sheet.png`.

| Prompt | Probe against teacher |
|---|---|
| green knight necromancer | same; adds horns to the helmet |
| stylized exterior architecture | same building; plainer background |
| overhead view surveillance | indoor fisheye office; mean padding alone already moves it indoors |
| emily blunt in futuristic power armor | armor kept; **different face**; plain background |
| medieval oil painting | same as mean padding |
| a middle aged man | near identical |
| three red apples on a wooden table | three apples, correct |
| a cat sitting to the left of a dog | correct order |
| a neon sign that says OPEN above a shop door | text correct |
| portrait of albert einstein, oil painting | old man with moustache; **not recognisably Einstein** |

First reading (Claude; the user's check decides): 10 of 10 decent. Counts, spatial order and short
text survive. Named identities weaken (2 of 2 names). Mean padding causes some composition changes on
its own (surveillance, medieval); the probe mostly follows the mean-padding image.

## Limits

One noise file, 10 prompts, one viewer. The probe keeps the full 0.6B model (28 layers).
No speed number.
