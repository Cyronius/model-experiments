# Stray-pixel count per sprite before/after a small-island cleanup on top of snap2, vs real sprites; 2026-10-19.
import sys
import numpy as np
from snap import snap2

SMALL = 2      # same-colour 4-connected islands of at most this many pixels count as strays
CONTRAST = 60  # and only when they differ from every neighbour colour by more than this (rgb distance)


def islands(a):
    """Small high-contrast same-colour islands: list of (pixels, neighbour colours)."""
    m = a[..., 3] > 0
    key = np.where(m, a[..., 0].astype(np.int64) << 16 | a[..., 1].astype(np.int64) << 8 | a[..., 2], -1)
    seen = np.zeros((32, 32), bool); out = []
    for y0, x0 in zip(*np.nonzero(m)):
        if seen[y0, x0]:
            continue
        comp, st = [], [(y0, x0)]; seen[y0, x0] = True
        while st:
            y, x = st.pop(); comp.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v, u = y + dy, x + dx
                if 0 <= v < 32 and 0 <= u < 32 and not seen[v, u] and key[v, u] == key[y0, x0]:
                    seen[v, u] = True; st.append((v, u))
        if len(comp) > SMALL:
            continue
        nb = {(v, u) for y, x in comp for v in range(y - 1, y + 2) for u in range(x - 1, x + 2)
              if 0 <= v < 32 and 0 <= u < 32 and m[v, u] and key[v, u] != key[y0, x0]}
        if len(nb) < 4:
            continue
        c0 = a[y0, x0, :3].astype(int)
        cols = np.array([a[v, u, :3] for v, u in nb], int)
        if np.sqrt(((cols - c0) ** 2).sum(1)).min() > CONTRAST:
            out.append((comp, cols))
    return out


def clean(a):
    a = a.copy()
    for comp, cols in islands(a):
        vals, cnt = np.unique(cols, axis=0, return_counts=True)
        for y, x in comp:
            a[y, x, :3] = vals[cnt.argmax()]
    return a


if __name__ == '__main__':
    z = np.load(sys.argv[1])
    sn = [snap2(x) for x in z['x']]
    before = [len(islands(a)) for a in sn]; after = [len(islands(clean(a))) for a in sn]
    d = np.load('C:/dev/sprite-data/trainset.npz')
    ix = np.random.default_rng(0).choice(np.nonzero(d['cat'] >= 0)[0], 400, replace=False)
    real = [len(islands(a)) for a in d['sprites'][ix]]
    print(f'snapped samples: mean {np.mean(before):.2f} strays/sprite (median {np.median(before):.0f}, max {max(before)})')
    print(f'after cleanup:   mean {np.mean(after):.2f} (median {np.median(after):.0f}, max {max(after)})')
    print(f'real sprites:    mean {np.mean(real):.2f} (median {np.median(real):.0f}, max {max(real)})')
    if len(sys.argv) > 2:
        from PIL import Image
        cols, S = 10, 96; im = Image.new('RGB', (cols * (2 * S + 12), 5 * (S + 4)), (118, 118, 118))
        for i, a in enumerate(sn):
            cx, cy = (i % cols) * (2 * S + 12) + 4, (i // cols) * (S + 4) + 2
            for j, p in enumerate((a, clean(a))):
                q = Image.fromarray(p).resize((S, S), Image.NEAREST); im.paste(q, (cx + j * (S + 2), cy), q)
        im.save(sys.argv[2])
