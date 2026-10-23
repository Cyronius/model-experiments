# Text for each training sprite from real metadata (file name, pack tags, TSR sheet title) plus pixel-derived colours; 2026-10-19.
import re
import numpy as np

STEM_GENERIC = set('sprite sprites spritesheet sheet image img icon icons tile tiles tileset untitled new frame frames layer layers final preview '
                   'example test copy free asset assets pack atlas texture output export png gif bmp v x big small final'.split())
TAG_GENERIC = {'pixel art', 'pixelart', 'pixel', 'sprite', 'sprites', 'spritesheet', '2d', 'rpg', 'free', 'cc0', 'asset', 'assets', 'assets pack', 'pack',
               'png', 'raster art', 'art', '2d art', 'no ai', 'game assets', 'tileset', 'tiles', 'tile', 'icon', 'icons', 'gui', 'ui', 'pixel-art'}
ROLE_WORD = {'boss': 'boss', 'enemy': 'enemy', 'npc': 'npc', 'hero': 'hero'}
PLATFORM = {'snes': 'snes', 'nes': 'nes', 'game_boy_advance': 'gba', 'sega_genesis': 'genesis', 'game_boy_gbc': 'game boy color'}


def tokens(text, limit=4):
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    out = []
    for t in re.split(r'[^A-Za-z]+', text):
        t = t.lower()
        if len(t) >= 2 and t not in STEM_GENERIC and t not in out:
            out.append(t)
    return out[:limit]


def subject_from_path(path):
    parts = re.split(r'[\\/]', path)
    base = re.sub(r'\.(png|gif|bmp)$', '', parts[-1], flags=re.I)
    t = tokens(base)
    if len(parts) > 1 and len(t) < 2:
        t = (tokens(parts[-2], 2) + t)[:4]
    return ' '.join(t)


def tags_from(tags, limit=3):
    out = []
    for t in tags:
        t = t.lower().replace('-', ' ').strip()
        if t and t not in TAG_GENERIC and not re.fullmatch(r'\d+x\d+', t) and t not in out:
            out.append(t)
    return out[:limit]


def colour_words(a, top=2, min_share=0.2):
    m = a[..., 3] > 0
    if m.sum() == 0:
        return []
    rgb = a[m][:, :3].astype(np.float32) / 255
    mx, mn = rgb.max(1), rgb.min(1)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    h = np.zeros(len(rgb), np.float32)
    nz = d > 0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    h[nz & (mx == r)] = ((g - b)[nz & (mx == r)] / d[nz & (mx == r)]) % 6
    h[nz & (mx == g)] = (b - r)[nz & (mx == g)] / d[nz & (mx == g)] + 2
    h[nz & (mx == b)] = (r - g)[nz & (mx == b)] / d[nz & (mx == b)] + 4
    h = h * 60
    name = np.full(len(rgb), 'gray', object)
    chroma = s >= 0.2
    for lo, hi, w in [(0, 15, 'red'), (15, 40, 'orange'), (40, 70, 'yellow'), (70, 165, 'green'), (165, 200, 'teal'), (200, 255, 'blue'), (255, 300, 'purple'), (300, 345, 'pink'), (345, 361, 'red')]:
        name[chroma & (h >= lo) & (h < hi)] = w
    name[chroma & np.isin(name, ['orange', 'red']) & (mx < 0.6)] = 'brown'
    name[(~chroma) & (mx >= 0.8)] = 'white'
    name[mx < 0.2] = 'black'
    vals, cnt = np.unique(name, return_counts=True)
    order = np.argsort(-cnt)
    return [vals[i] for i in order[:top] if cnt[i] / len(name) >= min_share]


def build(category=None, subject='', style='', tags=(), colours=()):
    parts = [category, subject, style, *tags, ' '.join(colours)]
    return ', '.join(p for p in parts if p)[:96]
