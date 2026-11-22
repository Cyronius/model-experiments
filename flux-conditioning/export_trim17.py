"""Export the healed 17-block Klein denoiser (heal2.py run4) as safetensors for Hugging Face: kept blocks + bridges (2026-11-21)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

A = Path(__file__).resolve().parents[1] / "artifacts" / "flux-conditioning" / "heal" / "run4"
OUT = Path(r"C:\dev\hf-upload\FLUX.2-klein-4B-Trim17")
TIMESTEPS = [1.0, 0.9609375, 0.8828125, 0.71484375]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ck = torch.load(A / "ckpt.pt", map_location="cpu", weights_only=False)
    blocks = {}
    for k, sd in sorted(ck["blocks"].items()):
        prefix = f"transformer_blocks.{k}." if k < 5 else f"single_transformer_blocks.{k - 5}."
        blocks |= {prefix + n: v.to(torch.bfloat16).contiguous() for n, v in sd.items()}
    save_file(blocks, str(OUT / "trim17_blocks.safetensors"), metadata={"format": "pt"})
    z = np.load(A / "bridge-v2.npz")
    save_file({k.replace("-", "_"): torch.from_numpy(z[k]).float().contiguous() for k in z.files},
              str(OUT / "trim17_bridges.safetensors"), metadata={"format": "pt"})
    kept = sorted(ck["blocks"])
    cfg = {"base_model": "black-forest-labs/FLUX.2-klein-4B", "blocks_total": 25, "blocks_kept": kept,
           "blocks_removed": [i for i in range(25) if i not in kept],
           "gaps": {"13_15": {"removed": [13, 14, 15], "bridge_before_block": 16},
                    "17_21": {"removed": [17, 18, 19, 20, 21], "bridge_before_block": 22}},
           "steps": 4, "timesteps": TIMESTEPS,
           "bridge": "x <- x + (LN(x) * (1 + scale_sgl) + shift_sgl) @ W[gap][step]; LN = single_transformer_blocks[0].norm"}
    (OUT / "trim17_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(len(blocks), "block tensors;", sorted(z.files), "->", OUT)


if __name__ == "__main__":
    main()
