"""On-the-fly palette variation for category-conditioned sprite training."""
import math

import torch


# trainset.npz / dit.py order: creature, item, weapon, food, effect, terrain,
# structure, nature. These four categories need more colour variation.
PALETTE_CATEGORIES = (2, 3, 6, 7)


def palette_swap(x, categories, probability=0.5, hue_degrees=25.0, generator=None):
    """Vary a sprite's palette without changing its shape or shading.

    Input: float BCHW RGBA in [-1, 1]. One chroma rotation and saturation
    multiplier per sprite; luminance is preserved to uint8 rounding precision.
    Near-neutral colours, dark outlines, highlights, alpha and hidden RGB stay
    unchanged. One shared gamut scale per sprite avoids clipped colour ramps.
    Applies only to weapon / food / structure / nature; originals stay in X.
    """
    if probability == 0 or hue_degrees == 0:
        return x
    batch = len(x)
    eligible = torch.zeros_like(categories, dtype=torch.bool)
    for category in PALETTE_CATEGORIES:
        eligible |= categories == category
    selected = eligible & (torch.rand(batch, device=x.device, generator=generator) < probability)
    angle = (torch.rand(batch, 1, 1, 1, device=x.device, generator=generator) * 2 - 1)
    angle = angle * (math.pi * hue_degrees / 180)
    saturation = 0.9 + 0.2 * torch.rand(batch, 1, 1, 1, device=x.device, generator=generator)

    rgb = (x[:, :3] + 1) * 0.5
    r, g, b = rgb.split(1, dim=1)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    u, v = (b - luminance) / 1.8556, (r - luminance) / 1.5748
    cosine, sine = angle.cos(), angle.sin()
    rotated_u = (u * cosine - v * sine) * saturation
    rotated_v = (u * sine + v * cosine) * saturation
    dr, db = 1.5748 * rotated_v, 1.8556 * rotated_u
    dg = -(0.2126 * dr + 0.0722 * db) / 0.7152
    delta = torch.cat([dr, dg, db], dim=1)

    chroma = rgb.amax(dim=1, keepdim=True) - rgb.amin(dim=1, keepdim=True)
    editable = ((x[:, 3:4] > -1) & (chroma > 0.06)
                & (luminance > 0.06) & (luminance < 0.94)
                & (rgb.amax(dim=1, keepdim=True) > 48 / 255))
    # For every colour, bound the chroma scale to keep Y + scale * delta in
    # gamut. Use a single scale for the whole palette, not separate RGB clips.
    limit = torch.where(delta > 0, (1 - luminance) / delta.clamp_min(1e-8),
                        luminance / (-delta).clamp_min(1e-8))
    limit = torch.where(editable, limit, torch.ones_like(limit))
    scale = limit.amin(dim=(1, 2, 3), keepdim=True).clamp(0, 1)
    swapped = (luminance + scale * delta).clamp(0, 1)
    swapped = (swapped * 255).round() / 127.5 - 1
    changed = editable & selected[:, None, None, None]
    return torch.cat([torch.where(changed, swapped, x[:, :3]), x[:, 3:4]], dim=1)
