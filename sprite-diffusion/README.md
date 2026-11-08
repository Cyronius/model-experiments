# sprite-diffusion

A 24M-parameter rectified-flow DiT that draws 32×32 RGBA pixel-art sprites by category (creature, item, weapon, food, effect, terrain, structure, nature), and runs on the AMD XDNA2 NPU as a sprite-sheet generator.

Status (2026-10-26): version-one training is complete. Keep the original 100k checkpoint and the 150k continuation: neither won every category in the same-seed visual review. **32 sampling steps is the owner's preferred default; 16 and 8 are useful balanced and fast options.** This is a category-conditioned generator, not a free-text prompt model.

The original patch-2 checkpoint generated a 64-sprite sheet on the AMD XDNA2 NPU in approximately **5.0 seconds at 8 steps** and **9.9 seconds at 16 steps** in the October 20 measurements. The 150k continuation and 32-step setting have not been revalidated or timed on the NPU. The full historical record is in [PLAN.md](PLAN.md); old `utilities/sprite-*` paths correspond to `sprite-*` here.

## What it makes

32×32 RGBA sprites conditioned on one of eight categories: creature, item, weapon, food, effect, terrain, structure and nature. Choose a category and seed, generate alternatives, then keep useful designs. Outputs can contain broken silhouettes, detached pixels, smooth colour ramps or partial structures. Creature output still favors people; food includes many potions. The model does not guarantee seamless terrain, coherent animations, production-ready assets, or freedom from resemblance to its training material.

## Completed training and comparison

| Checkpoint | Training | Practical observation |
|---|---|---|
| 100k | Original category-conditioned patch-2 model | Useful reference; more character variety in the inspected seeds |
| 110k | 10k-update continuation | Mixed changes; no clear overall advantage |
| 150k | Another 40k updates, resuming the continuation | More varied nature designs in the inspected seeds; no universal quality win |

The continuation used the expanded 383,660-sprite dataset, category/subtype-balanced sampling, a 3e-5 learning rate and palette augmentation. The original optimizer was reset for the first continuation; the next 40k resumed the pilot's full training state. The two successful jobs took about 48 and 184 minutes on an RTX 4090, with estimated GPU charges of $0.71 and $2.73. These exclude the earlier run whose results remain inaccessible on its stopped host and exclude persistent storage charges.

The comparison generated 64 sprites per setting across all eight categories, using the same initial noise for 100k/110k/150k at **1, 2, 4, 8, 12, 16, 32 and 64 steps** (1,536 sprites total). It used EMA weights, CFG 3, uniform Euler integration, null footprint conditioning and no palette snapping. The two seed sets were 17 and 93, with four samples per category each. One conversational vision review inspected all sheets; it was not an independent blinded benchmark. No held-out quality metric, matched uniform control or new copying analysis was completed.

- 1–2 steps: severely underdeveloped.
- 4 steps: rough drafting.
- 8, 16 and 32 steps: useful operating choices; the owner prefers 32.
- 64 steps: mixed changes and visible regressions in some samples.

Training is finished for this version. Further progress should be driven by specific quality goals and better data, rather than additional updates alone.

Model weights and code are licensed under **Apache 2.0**. All training data is excluded from this release.

- Browser generator: https://cyronius.github.io/model-experiments/sprite-diffusion/
- Inference-only weights: https://huggingface.co/Cyronius/sprite-diffusion


| folder | what's there |
|---|---|
| `sprite-data/` | data harvest and automated labelling: pack cutters, crawlers, SigLIP labeller, bake-off scripts |
| `sprite-train/` | `dit.py` (model), `train.py`, captions, evaluation, palette snap, and the NPU port: `export_npu.py`, `build_npu_sets.py`, `npu_sprite.py` |
| `npu-kernels/` | forks of OpenFlowLM's `dit_fa` (block-diagonal batched attention) and `dit_ew` (row size 384, sprite 2-D RoPE, residual bias) |
| `spec/` | requirements (`SPRITE-*`) and the palette test |

Data and outputs live outside git: `C:\dev\sprite-data` (training set, runs, checkpoints) and `C:\dev\sprite-npu` (kernel sets, exported weights, sheets).

## Training palette augmentation

`train.py` now applies palette variation on the fly to weapon, food, structure and nature
sprites with probability 0.5. Each selected sprite gets one hue rotation of up to 25 degrees
and a saturation multiplier from 0.9 to 1.1. Luminance is preserved to uint8 rounding;
dark outlines, near-neutral colours, highlights, transparency and geometry stay unchanged.
The same source palette colour always maps to the same new colour. Originals stay in the
dataset; augmented copies do not count as additional designs.

Use `--palette-prob 0` for the previous training behaviour, or `--palette-hue` to adjust the
rotation range. Palette augmentation defaults off with `--text`, because changing colours
would contradict caption colour words. Horizontal flips still run independently at 50%.

Preview real training examples before a run:

```
python sprite-train/preview_augmentation.py --data C:/dev/sprite-data/trainset.npz --out palette-preview
```

Sampling choice (2026-10-25): **32 default, 16 balanced, 8 fast**. Palette augmentation ran during the 110k and 150k continuations. It increases colour variation, not distinct designs. Human and monster coverage still needs separate tracking within `creature`.

## NPU sheet

Needs the IRON environment and an OpenFlowLM checkout (for `dit_gemm`, the pyxrt harness and `build_design.py`), named by `OFLM_ROOT`.

```
. C:\dev\mlir-aie\iron_env.ps1
$env:OFLM_ROOT = 'C:\code\OpenFlowLM-Next2'
python sprite-train\export_npu.py --ckpt C:\dev\sprite-data\runs\q-p2-cat\ckpt.pt --out C:\dev\sprite-npu\weights-p2.npz   # igpu-venv (torch)
python sprite-train\build_npu_sets.py --patch 2 --out C:\dev\sprite-npu\sets-p2
python sprite-train\npu_sprite.py --weights C:\dev\sprite-npu\weights-p2.npz --sets C:\dev\sprite-npu\sets-p2 sheet --cat food --steps 8 --out sheet.png
```

## Balanced training exposure (2026-10-23)

`train.py --sampling uniform` retains the previous per-image distribution and is the
compatibility default. `--sampling category` gives 90% of draws equally to the populated
labelled categories, and 10% to unlabelled images. `--unlabelled-prob` changes that share.
With all eight categories present, each receives 11.25%. Category CFG dropout remains
independent at 10%, giving 19% null-category conditioning with the default balanced mix.

`--sampling category-subtype` additionally splits the creature share equally among the
populated trusted human, monster and unknown pools. Unknown annotations stay unknown;
subtypes affect sampling only, not model output categories. This mode is experimental:
the known subtype pools are small, so increased exposure can overfit their designs.
Neither sampling mode creates additional designs.

The expanded 383,660-image dataset's 200,000-draw audit found structure/nature exposure
near 11.25% each (previously about 1.3%), and trusted monster exposure near 3.75%
(previously 0.56%) in subtype mode. Audit without loading sprite pixels:

```
python sprite-train/audit_sampling.py --data C:/code/model-experiments/artifacts/sprite-data-expansion/trainset-expanded.npz --out sampling-audit.json
```

Every training run writes `sampling.json`; checkpoints store the distribution and sampler
RNG state. Changing the distribution during a resume is rejected. For a new category-only
continuation, `--init-from <checkpoint>` loads model and EMA weights into a separate output
directory, resetting optimizer and step. Subsequent slices resume that directory normally,
including when `--init-from` remains on the wrapper command. `--steps` then counts new
updates. Existing output checkpoints from other initializations are rejected.

The planned 10k pilot and subsequent 40k continuation are now complete. The 100k original was preserved, and both completed continuation checkpoints were recovered locally before their pods stopped. The expanded pool overlaps the original training sources: genuinely unseen validation packs and a matched uniform control remain outstanding. Sampling audits establish exposure, not image quality; these runs do not isolate augmentation from sampling or additional training.
