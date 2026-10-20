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

## NPU sheet

Needs the IRON environment and an OpenFlowLM checkout (for `dit_gemm`, the pyxrt harness and `build_design.py`), named by `OFLM_ROOT`.

```
. C:\dev\mlir-aie\iron_env.ps1
$env:OFLM_ROOT = 'C:\code\OpenFlowLM-Next2'
python sprite-train\export_npu.py --ckpt C:\dev\sprite-data\runs\q-p2-cat\ckpt.pt --out C:\dev\sprite-npu\weights-p2.npz   # igpu-venv (torch)
python sprite-train\build_npu_sets.py --patch 2 --out C:\dev\sprite-npu\sets-p2
python sprite-train\npu_sprite.py --weights C:\dev\sprite-npu\weights-p2.npz --sets C:\dev\sprite-npu\sets-p2 sheet --cat food --steps 8 --out sheet.png
```
