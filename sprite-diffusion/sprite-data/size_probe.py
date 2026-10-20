# Probe: how big are sprites on The Spriters Resource? Crawls up to PER_GAME sprite
# sheets (tile/map/UI sheets skipped by title) for each game in GAMES, cuts each sheet into
# connected components on its background, dedupes them, and prints how many fit 16x16,
# 16x32, 32x32 and 64x64. Writes size_rows.json and SNES contact sheets gain_*.png to cwd.
# Requests are cached in TSR_CACHE and spaced 0.6 s apart.
import re, time, os, html, json, hashlib, urllib.request, urllib.error, io, sys
import numpy as np
from PIL import Image
from scipy import ndimage
BASE = 'https://www.spriters-resource.com'
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
C = os.environ.get('TSR_CACHE', 'C:/dev/sprite-data/tsr_cache')
os.makedirs(C, exist_ok=True)
def get(path, binary=False):
    fn = os.path.join(C, hashlib.md5(path.encode()).hexdigest())
    if os.path.exists(fn):
        b = open(fn, 'rb').read()
    else:
        time.sleep(0.6)
        for attempt in range(3):
            try:
                b = urllib.request.urlopen(urllib.request.Request(BASE + path, headers=UA), timeout=30).read(); break
            except urllib.error.HTTPError as e:
                return None
            except Exception as e:
                time.sleep(2)
        else:
            return None
        open(fn, 'wb').write(b)
    return b if binary else b.decode('utf8', 'replace')
def norm(x): return re.sub(r'[^a-z0-9]', '', x.lower().replace('é', 'e'))
def slug(plat, title):
    guess = f'/{plat}/{norm(re.sub(r"^the ", "", title.lower()))}/'
    p = get(guess)
    if p and '<title>' in p and norm(title.split(':')[0].replace('The ', '')) in norm(p.split('<title>')[1].split('</title>')[0]): return guess
    t = re.sub(r'^the ', '', title.lower()); letter = t[0].upper() if t[0].isalpha() else '0'
    s = get(f'/{plat}/{letter}/')
    for u, txt in re.findall(rf'href="(/{plat}/[a-z0-9]+/)"[^>]*>(.*?)</a>', s, re.S):
        txt = ' '.join(html.unescape(re.sub('<[^>]+>', ' ', txt)).split())
        if norm(txt) == norm(title): return u
    print(f'  !! no slug for {plat} {title}'); return None
SKIP = re.compile(r'tile|map|level|stage|area|world|background|\bbg\b|screen|font|hud|menu|text|portrait|cutscene|ending|intro|credit|logo|palette|vram|interface|window|dialog', re.I)
GAMES = {
 'nes': ['The Legend of Zelda', 'Super Mario Bros. 3', 'Mega Man 2', 'Final Fantasy', 'Castlevania'],
 'snes': ['The Legend of Zelda: A Link to the Past', 'Chrono Trigger', 'Final Fantasy VI', 'Secret of Mana', 'Super Mario World', 'EarthBound'],
 'game_boy_gbc': ['Pokémon Red / Blue', "The Legend of Zelda: Link's Awakening DX"],
}
PER_GAME = 18
def bg_mask(a):
    if (a[..., 3] == 0).mean() > 0.05:
        return a[..., 3] == 0
    rgb = a[..., :3].reshape(-1, 3); keys = (rgb[:, 0].astype(np.int64) << 16) | (rgb[:, 1].astype(np.int64) << 8) | rgb[:, 2]
    vals, cnt = np.unique(keys, return_counts=True)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])[:, :3]
    bk = (border[:, 0].astype(np.int64) << 16) | (border[:, 1].astype(np.int64) << 8) | border[:, 2]
    bv, bc = np.unique(bk, return_counts=True)
    bgs = set(vals[cnt > 0.15 * len(keys)]) | {bv[bc.argmax()]}
    return np.isin(keys, list(bgs)).reshape(a.shape[:2])
seen = set(); rows = []; samples = {}
for plat, titles in GAMES.items():
    for title in titles:
        g = slug(plat, title)
        if not g: continue
        page = get(g)
        assets = [(u, ' '.join(html.unescape(re.sub('<[^>]+>', ' ', t)).split()))
                  for u, t in re.findall(rf'href="({re.escape(g)}asset/\d+/)"[^>]*>(.*?)</a>', page, re.S)]
        keep = [(u, t) for u, t in dict(assets).items() if t and not SKIP.search(t)][:PER_GAME]
        print(f'{plat:13s} {title[:40]:40s} sheets {len(assets):3d} kept {len(keep)}', flush=True)
        for u, t in keep:
            sp = get(u); m = re.search(r'src="(/media/assets/[^"?]+\.png)', sp or '')
            if not m: continue
            b = get(m.group(1), binary=True)
            if not b: continue
            a = np.array(Image.open(io.BytesIO(b)).convert('RGBA'))
            if a.shape[0] * a.shape[1] > 6e6: continue
            fg = ~bg_mask(a)
            lab, n = ndimage.label(fg, structure=np.ones((3, 3)))
            for i, sl in enumerate(ndimage.find_objects(lab), 1):
                h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
                m_ = lab[sl] == i; area = m_.sum()
                if area < 24: continue
                px = a[sl][m_][:, :3]; ncol = len(np.unique(px, axis=0))
                if ncol < 3: continue                      # glyphs / outlines / swatches
                if area > 0.97 * h * w and h * w > 64 and ncol < 6: continue  # solid boxes
                crop = np.where(m_[..., None], a[sl], 0)
                k = hashlib.md5(crop.tobytes() + bytes([h, w]) if h < 256 and w < 256 else crop.tobytes()).hexdigest()
                if k in seen: continue
                seen.add(k)
                bucket = ('16x16' if w <= 16 and h <= 16 else '16x32' if w <= 16 and h <= 32 else
                          '32x32' if w <= 32 and h <= 32 else '64x64' if w <= 64 and h <= 64 else 'bigger')
                rows.append((plat, title, t, h, w, bucket))
                if bucket in ('16x32', '32x32') and plat == 'snes':
                    samples.setdefault(bucket, []).append(crop)
json.dump(rows, open('size_rows.json', 'w'))
import collections
print()
for plat in GAMES:
    c = collections.Counter(r[5] for r in rows if r[0] == plat); n = sum(c.values())
    cum = 0; out = []
    for b in ['16x16', '16x32', '32x32', '64x64', 'bigger']:
        cum += c[b]; out.append(f'{b}: {c[b]:5d} (cum {100*cum/max(n,1):3.0f}%)')
    print(f'{plat:13s} n={n:5d}  ' + '  '.join(out))
# contact sheets of what 16x32 and 32x32 would add on SNES
for b, L in samples.items():
    L = L[::max(1, len(L) // 48)][:48]; H = 34 if b == '16x32' else 34; W = 18 if b == '16x32' else 34
    sheet = np.full((H * 4, W * 12, 4), (90, 90, 90, 255), np.uint8)
    for j, cimg in enumerate(L):
        r, cidx = divmod(j, 12); h, w = cimg.shape[:2]
        tile = sheet[r * H + 1 + (H - 2 - h): r * H + 1 + (H - 2), cidx * W + 1 + (W - 2 - w) // 2: cidx * W + 1 + (W - 2 - w) // 2 + w]
        mm = cimg[..., 3] > 0; tile[mm] = cimg[mm]
    Image.fromarray(sheet).resize((sheet.shape[1] * 4, sheet.shape[0] * 4), Image.NEAREST).save(f'gain_{b}.png')
