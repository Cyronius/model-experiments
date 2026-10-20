# Snaps a generated RGBA sprite to a small palette (median cut, no dithering); alpha thresholded; 2026-10-19.
import numpy as np
from PIL import Image


def snap(x, colours=12):
    """x: (4,32,32) floats in [-1,1] -> uint8 RGBA (32,32,4)."""
    a = ((np.clip(x.astype(np.float32), -1, 1) + 1) * 127.5).round().astype(np.uint8).transpose(1, 2, 0)
    mask = a[..., 3] >= 128
    out = np.zeros_like(a)
    if mask.sum() >= 4:
        rgb = Image.fromarray(a[..., :3])
        q = rgb.quantize(colors=colours, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).convert('RGB')
        out[..., :3] = np.array(q)
    out[..., 3] = np.where(mask, 255, 0)
    out[~mask] = 0
    return out


def _kmeans(p, k, iters=12):
    c = p[np.linspace(0, len(p) - 1, k).astype(int)].copy()
    for _ in range(iters):
        lab = ((p[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (lab == j).any():
                c[j] = p[lab == j].mean(0)
    return c, ((p[:, None] - c[None]) ** 2).sum(-1).argmin(1)


def snap2(x, max_colours=16, tol=12.0):
    """Palette from opaque pixels only, fewest colours within tol (rms, 0-255), then strays and lone pixels cleaned."""
    a = ((np.clip(x.astype(np.float32), -1, 1) + 1) * 127.5).round().transpose(1, 2, 0)
    mask = a[..., 3] >= 128
    # Lone opaque pixels (<=1 opaque neighbour) are alpha speckle.
    pad = np.pad(mask, 1)
    nb = sum(np.roll(np.roll(pad, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)[1:-1, 1:-1]
    mask &= nb >= 2
    out = np.zeros((32, 32, 4), np.uint8)
    if mask.sum() < 4:
        return out
    p = a[mask][:, :3]
    for k in range(2, max_colours + 1):
        c, lab = _kmeans(p, k)
        if np.sqrt(((p - c[lab]) ** 2).sum(1).mean()) <= tol:
            break
    idx = np.full((32, 32), -1); idx[mask] = lab
    # A pixel whose colour no neighbour shares takes its neighbours' most common colour.
    fixed = idx.copy()
    for y, xx in zip(*np.nonzero(mask)):
        n = idx[max(0, y - 1):y + 2, max(0, xx - 1):xx + 2].ravel()
        n = n[n >= 0]
        if (n == idx[y, xx]).sum() == 1 and len(n) >= 5:
            fixed[y, xx] = np.bincount(n[n != idx[y, xx]]).argmax()
    out[mask, :3] = np.clip(c[fixed[mask]], 0, 255).round().astype(np.uint8)
    out[mask, 3] = 255
    return out
