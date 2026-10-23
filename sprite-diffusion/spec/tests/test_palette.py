# Traces: SPRITE-PALETTE
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sprite-train"))
from snap import snap2  # noqa: E402


def test_snap2_palette_and_alpha():
    rng = np.random.default_rng(0)
    for _ in range(20):
        out = snap2(rng.uniform(-1, 1, (4, 32, 32)).astype(np.float32))
        opaque = out[..., 3] == 255
        assert set(np.unique(out[..., 3])) <= {0, 255}
        assert len(np.unique(out[opaque][:, :3], axis=0)) <= 16
