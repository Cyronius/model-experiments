# Sprite diffusion: 32×32 pixel-art sprite sheets on the NPU

What the sprite sheet generator must do. Directory name gives the prefix: `SPRITE`.

The model is a 24M-parameter rectified-flow DiT (`utilities/sprite-train/dit.py`: hidden 384,
10 single-stream blocks, 3 heads of 128, patch 2 or 4) trained locally on ripped game
sprites. Its weights stay local and are not published. Plan and measurements:
`.claude/plans/sprite-diffusion-plan.md` (Phase 3).

**The reference is the PyTorch model in fp32 on the CPU** (`utilities/sprite-train/npu_goldens.py`).

## Requirements

### SPRITE-SHEET: a sheet of 64 sprites of one category
**Applies to:** OpenFlowLM-Next (sprite sheet runner)
**Verification:** manual

Given a category, a seed and a step count, the runner writes one PNG holding 64 RGBA 32×32
sprites of that category, palette-snapped, with transparent backgrounds.

**Verification (manual):**
1. Run the sheet runner for `food`, seed 1, 16 steps.
2. Open the PNG: 64 cells, each a 32×32 sprite on a transparent background.
3. Most cells read as food (potions and drinks count).

### SPRITE-NPU-PARITY: the NPU velocity matches the reference
**Applies to:** OpenFlowLM-Next (sprite sheet runner)
**Verification:** test

For fixed noise, category and timestep, one model evaluation on the NPU matches the PyTorch
reference velocity.

**Acceptance criteria:**
- rel_fro(NPU, reference) ≤ 3e-2 and every row's cosine > 0.999, at patch 2 and patch 4.
- The test skips, not fails, on a machine with no NPU.

### SPRITE-PALETTE: snapped sprites use a small palette
**Applies to:** OpenFlowLM-Next (`utilities/sprite-train/snap.py`)
**Verification:** test

`snap2` maps a generated sprite to at most 16 opaque colours and fully transparent or fully
opaque alpha.

**Acceptance criteria:**
- Any input in [-1, 1] of shape (4, 32, 32) gives at most 16 distinct RGB values among
  opaque pixels.
- Every output alpha is 0 or 255.
