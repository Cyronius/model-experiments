# sprite-diffusion

A 24M-parameter rectified-flow DiT that draws 32×32 RGBA pixel-art sprites by category (creature, item, weapon, food, effect, terrain, structure, nature), and runs on the AMD XDNA2 NPU as a sprite-sheet generator.

Status (2026-10-20): the patch-2 model (256 tokens per sprite, 100k steps) generates a 64-sprite sheet in about 5 s on the NPU at 8 steps. The full history, decisions and measurements are in [PLAN.md](PLAN.md) (written while the work lived in OpenFlowLM, so its paths say `utilities/sprite-*`; here they are `sprite-*`).

Model weights are not published: the training data includes ripped game sprites.

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

Sampling choice (2026-10-23): 16 steps normally, 8 steps for fast mode. More training and
targeted new designs for structure, nature, food, weapon and nonhuman monsters are next.
Human and monster coverage needs separate tracking within `creature`; its combined count
does not establish monster coverage. No new training run has been launched by adding
augmentation.

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

Example configuration for a future pilot (not launched): use the expanded data, a new
output directory, `--init-from C:/dev/sprite-data/runs/q-p2-cat/ckpt.pt`,
`--sampling category-subtype --lr 3e-5 --steps 10000 --sample-steps 16` with `train_dibs.ps1`.
The pilot still needs genuinely unseen validation packs and a matched uniform control;
the recovered source packs overlap the old training set. Keep palette probability equal
between controls to isolate the sampling change. The best 100k-step checkpoint remains
unchanged. CPU smoke runs verify execution and resume, not image quality or iGPU speed.
