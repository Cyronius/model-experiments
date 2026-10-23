# Shared sprite helpers for the data pipeline: content tests, the 32x32 training canvas,
# and the rendering every labeller sees.
import numpy as np
from PIL import Image

DATA = 'C:/dev/sprite-data'
CANVAS = 32


def rgba(path_or_img):
    im = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    return np.array(im.convert('RGBA'))


def trim(a):
    """Crop to the alpha bounding box; None if fully transparent."""
    ys, xs = np.nonzero(a[..., 3])
    if len(ys) == 0:
        return None
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def has_content(a, min_px=12, min_colours=3):
    m = a[..., 3] > 0
    if m.sum() < min_px:
        return False
    return len(np.unique(a[m][:, :3], axis=0)) >= min_colours


def footprint(a):
    h, w = a.shape[:2]
    if h <= 16 and w <= 16:
        return '16x16'
    if h <= 32 and w <= 16:
        return '16x32'
    return '32x32' if h <= 32 and w <= 32 else None


def on_canvas(a):
    """Bottom-centre a trimmed sprite on the 32x32 transparent training canvas."""
    h, w = a.shape[:2]
    c = np.zeros((CANVAS, CANVAS, 4), np.uint8)
    x = (CANVAS - w) // 2
    c[CANVAS - h:, x:x + w] = a
    return c


def render(a, size=256, fit=224, bg=(128, 128, 128)):
    """What labellers see: the trimmed sprite at the largest integer scale that fits `fit`,
    nearest-neighbour, centred on neutral grey, RGB."""
    a = trim(a)
    h, w = a.shape[:2]
    k = max(1, fit // max(h, w))
    sp = Image.fromarray(a).resize((w * k, h * k), Image.NEAREST)
    out = Image.new('RGB', (size, size), bg)
    out.paste(sp, ((size - w * k) // 2, (size - h * k) // 2), sp)
    return out


def cells(sheet, cw=16, ch=16):
    """Yield (row, col, cell) for every cell of a grid sheet."""
    H, W = sheet.shape[:2]
    for r in range(H // ch):
        for c in range(W // cw):
            yield r, c, sheet[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
