"""c(g) = V^T (teacher(g) - mu) for the T2 bigram and trigram seed vectors."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import seeds as SD  # noqa: E402
mu, V = SD.basis()
for n in (2, 3):
    c = SD.coords(np.load(SD.S2 / f"seedvec_n{n}_T2.npy", mmap_mode="r"), mu, V)
    np.save(SD.S2 / f"c_n{n}_T2.npy", c)
    print(n, c.shape, float(np.linalg.norm(c, axis=1).mean()), int((np.abs(c).sum(1) == 0).sum()), flush=True)
