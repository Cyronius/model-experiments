# Shared helpers for The Spriters Resource (TSR): a polite cached fetcher, the game-page
# parser (section + sheet title for every sheet), category/role seeds from section and title
# keywords, and the sheet -> sprite cutter.
import hashlib, html, os, re, time, urllib.error, urllib.request
import numpy as np
from scipy import ndimage
from spritekit import DATA, has_content

BASE = 'https://www.spriters-resource.com'
CACHE = f'{DATA}/tsr_cache'
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
DELAY = 0.8          # seconds between network requests; cached reads are free
_last = [0.0]


def cached(path):
    fn = os.path.join(CACHE, hashlib.md5(path.encode()).hexdigest())
    return open(fn, 'rb').read() if os.path.exists(fn) else None


def get(path, binary=False):
    """GET a site path through the on-disk cache. Every response is cached before use, so
    an interrupted crawl resumes by re-running: cached paths cost nothing."""
    b = cached(path)
    if b is None:
        os.makedirs(CACHE, exist_ok=True)
        for attempt in range(4):
            wait = DELAY - (time.time() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
            try:
                b = urllib.request.urlopen(urllib.request.Request(BASE + path, headers=UA), timeout=30).read()
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if e.headers.get('cf-mitigated'):   # the site opted out of bots; never bypass it
                    raise RuntimeError(f'TSR serves a Cloudflare challenge for {path}; crawl stopped')
                time.sleep(5 * (attempt + 1))
            except Exception:
                time.sleep(5 * (attempt + 1))
        else:
            return None
        open(os.path.join(CACHE, hashlib.md5(path.encode()).hexdigest()), 'wb').write(b)
    return b if binary else b.decode('utf8', 'replace')


def text(x):
    return ' '.join(html.unescape(re.sub('<[^>]+>', ' ', x)).split())


def platform_games(plat):
    """(title, game path) for every game listed on a platform's letter pages."""
    out = {}
    for letter in ['0'] + [chr(c) for c in range(ord('A'), ord('Z') + 1)]:
        s = get(f'/{plat}/{letter}/') or ''
        for u, t in re.findall(rf'href="(/{plat}/[a-z0-9]+/)"[^>]*>(.*?)</a>', s, re.S):
            t = text(t)
            if t and len(u.strip('/').split('/')) == 2:
                out[u] = t
    return [(t, u) for u, t in out.items()]


def sheets(game_path, page=None):
    """(section, title, asset path) for every sheet on a game page, in page order."""
    page = page or get(game_path) or ''
    tok = re.compile(r'<div class="section[^"]*" id="section-\d+">(.*?)</div>|'
                     rf'href="({re.escape(game_path)}asset/\d+/)"[^>]*>(.*?)</a>', re.S)
    section, seen = None, set()
    for m in tok.finditer(page):
        if m.group(1) is not None:
            section = re.sub(r'^arrow_drop_down\s*\[\d+\]\s*', '', text(m.group(1)))
        elif m.group(2) not in seen:
            seen.add(m.group(2))
            yield section, text(m.group(3)), m.group(2)


def sheet_png(asset_path):
    """The sheet image bytes for an asset page, or None."""
    p = get(asset_path)
    m = re.search(r'src="(/media/assets/[^"?]+\.png)', p or '')
    return get(m.group(1), binary=True) if m else None


# Seeds: (regex on the sheet title, regex on the section, category, role); first match wins.
# Character and monster are one category, "creature" (2026-10-04); the section gives the role.
SEED_RULES = [
    (r'\b(tile ?sets?|tiles)\b', None, 'tile', None),
    (r'^(?!.*\b(fonts?|hud)\b).*\bitems?\b.*\bicons?\b', None, 'item', None),    # "Item Icons" beats "icons -> ui"
    (r'^(?!.*\b(fonts?|hud)\b).*\bweapons?\b.*\bicons?\b', None, 'weapon', None),
    (r'\bmaps?\b|\bbackgrounds?\b|\bstages?\b|\blevels?\b|\bportraits?\b|\bmugshots?\b|\bcutscenes?\b', None, 'map', None),
    (r'\bfonts?\b|\bhud\b|\bmenus?\b|\binterface\b|\bicons?\b|\btitle screen', None, 'ui', None),
    (r'\bfood\b', None, 'food', None),
    (r'\beffects?\b|\bprojectiles?\b|\bparticles?\b|\bexplosions?\b|\bspells?\b', None, 'effect', None),
    (r'\bweapons?\b', None, 'weapon', None),
    (r'\bitems?\b|\btreasures?\b|\bpick-?ups?\b|\bpower-?ups?\b', None, 'item', None),
    (None, r'\bfood\b', 'food', None),
    (None, r'\beffects?\b|\bprojectiles?\b', 'effect', None),
    (None, r'\bweapons?\b', 'weapon', None),
    (None, r'\bitems?\b', 'item', None),
    (None, r'\bboss(es)?\b(?!.*enem)', 'creature', 'boss'),
    (None, r'\benem(y|ies)\b', 'creature', 'enemy'),
    (None, r'\bnon-playable\b|\bnpcs?\b', 'creature', 'npc'),
    (None, r'\bplayable characters?\b', 'creature', 'hero'),
    (r'\bboss(es)?\b', None, 'creature', 'boss'),
    (r'\benem(y|ies)\b|\bmonsters?\b|\bcreatures?\b', None, 'creature', 'enemy'),
    (r'\bnpcs?\b|\btownspeople\b|\bvillagers?\b', None, 'creature', 'npc'),
]


def seed(title, section):
    t, s = (title or '').lower(), (section or '').lower()
    for rt, rs, cat, role in SEED_RULES:
        if (rt is None or re.search(rt, t)) and (rs is None or re.search(rs, s)):
            return cat, role
    return None, None


def bg_mask(a):
    if (a[..., 3] == 0).mean() > 0.05:
        return a[..., 3] == 0
    rgb = a[..., :3].reshape(-1, 3).astype(np.int64); keys = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    vals, cnt = np.unique(keys, return_counts=True)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])[:, :3].astype(np.int64)
    bk = (border[:, 0] << 16) | (border[:, 1] << 8) | border[:, 2]
    bv, bc = np.unique(bk, return_counts=True)
    return np.isin(keys, list(set(vals[cnt > 0.15 * len(keys)]) | {bv[bc.argmax()]})).reshape(a.shape[:2])


def strip_box(crop):
    """A sprite drawn on its own coloured box (a second background colour on the sheet)
    comes out as the whole box. If the crop's outer ring is almost all one colour, make that
    colour transparent where it touches the border, then re-trim."""
    m = crop[..., 3] > 0
    ring = np.concatenate([crop[0], crop[-1], crop[1:-1, 0], crop[1:-1, -1]])
    ring_rgb = ring[ring[..., 3] > 0][:, :3]
    if len(ring_rgb) < 0.8 * len(ring):
        return crop
    vals, cnt = np.unique(ring_rgb, axis=0, return_counts=True)
    if cnt.max() < 0.8 * len(ring):
        return crop
    same = (crop[..., :3] == vals[cnt.argmax()]).all(-1) & m
    lab, _ = ndimage.label(same)
    edge = set(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist()) - {0}
    out = crop.copy(); out[np.isin(lab, list(edge)), 3] = 0
    ys, xs = np.nonzero(out[..., 3])
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(ys) else None


def is_text(crop):
    """Labels drawn on the sheet: one ink colour, thin strokes."""
    m = crop[..., 3] > 0
    _, cnt = np.unique(crop[m][:, :3], axis=0, return_counts=True)
    return cnt.max() >= 0.7 * m.sum() and m.sum() < 0.5 * m.size


def sprites(a, lo=6, hi=32, min_px=30):
    """Connected components on the sheet background that fit the 32x32 canvas, with
    coloured backing boxes stripped and drawn-on text dropped."""
    lab, _ = ndimage.label(~bg_mask(a), structure=np.ones((3, 3)))
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if not (lo <= h <= hi and lo <= w <= hi):
            continue
        m = lab[sl] == i
        crop = a[sl].copy(); crop[~m] = 0; crop[m, 3] = 255
        crop = strip_box(crop)
        if crop is None or min(crop.shape[:2]) < 4:
            continue
        if (crop[..., 3] > 0).sum() >= min_px and has_content(crop) and not is_text(crop):
            yield crop
