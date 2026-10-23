# Bake-off TSR set: sprites cut from The Spriters Resource sheets already in the crawl cache
# (no new requests), each with its game, section and sheet title and a category seed from
# high-precision keyword rules. Frozen with the 2026-10-03 bake-off: its seed rules predate
# the character+monster merge (tsrkit.SEED_RULES has the current ones). TSR has no answer key; the bake-off scores agreement between
# the seed, the SigLIP probe and the VLM. Writes DATA/bakeoff/tsr.jsonl and render/tsr-*.png.
import html, io, json, os, random, re
import numpy as np
from PIL import Image
from spritekit import DATA, trim, footprint, render
import tsrkit

CACHE = tsrkit.CACHE
B = f'{DATA}/bakeoff'
PER_SHEET, TOTAL = 4, 240


cached, text, sheets = tsrkit.cached, tsrkit.text, tsrkit.sheets


SEED_RULES = [   # (regex on "title | section", seed); first match wins
    (r'\b(tile ?sets?|tiles)\b', 'tile'),
    (r'\bmaps?\b|\bbackgrounds?\b|\bstages?\b|\blevels?\b', None),
    (r'\bfonts?\b|\bhud\b|\bmenus?\b|\binterface\b|\bicons?\b|\btitle screen', 'ui'),
    (r'\bweapons?\b', 'weapon'),
    (r'\bfood\b', 'food'),
    (r'\beffects?\b|\bprojectiles?\b|\bparticles?\b|\bexplosions?\b|\bmagic\b|\bspells?\b', 'effect'),
    (r'\bitems?\b|\btreasures?\b|\bpick-?ups?\b|\bpower-?ups?\b', 'item'),
    (r'\bboss(es)?\b', 'boss'),
    (r'\benem(y|ies)\b|\bmonsters?\b|\bcreatures?\b', 'monster'),
    (r'\bplayable characters?\b|\bnpcs?\b|\bnon-playable\b|\bcharacters?\b|\btownspeople\b|\bvillagers?\b', 'character'),
]


def seed(title, section):
    s = f'{title} | {section or ""}'.lower()
    for rx, cat in SEED_RULES:
        if re.search(rx, s):
            return cat
    return None


sprites = tsrkit.sprites


games = []
for f in os.listdir(CACHE):
    s = open(os.path.join(CACHE, f), 'rb').read(4096).decode('utf8', 'replace')
    m = re.search(r'<title>(.+?) - (NES|SNES|Game Boy / Game Boy Color|Game Boy Advance) - The Spriters Resource', s)
    if m:
        full = open(os.path.join(CACHE, f), 'rb').read().decode('utf8', 'replace')
        g = re.search(r'href="/addasset(/[a-z_0-9]+/[a-z0-9]+/)"', full)
        if g and cached(g.group(1)):
            games.append((html.unescape(m.group(1)), m.group(2), g.group(1)))
random.seed(0); recs = []
for game, plat, gp in sorted(set(games)):
    for section, title, ap in sheets(gp):
        sp = cached(ap)
        mm = re.search(r'src="(/media/assets/[^"?]+\.png)', sp.decode('utf8', 'replace')) if sp else None
        png = cached(mm.group(1)) if mm else None
        if not png:
            continue
        cand = list(sprites(np.array(Image.open(io.BytesIO(png)).convert('RGBA'))))
        random.shuffle(cand)
        for a in cand[:PER_SHEET]:
            a = trim(a)
            recs.append(dict(id=f'tsr-{len(recs):05d}', game=game, platform=plat, section=section, title=title,
                             seed=seed(title, section), footprint=footprint(a), _a=a))
random.shuffle(recs); recs = recs[:TOTAL]
with open(f'{B}/tsr.jsonl', 'w') as fh:
    for r in recs:
        render(r.pop('_a')).save(f"{B}/render/{r['id']}.png")
        fh.write(json.dumps(r) + '\n')
from collections import Counter
print(len(recs), 'sprites from', len({r['title'] + r['game'] for r in recs}), 'sheets;', Counter(r['seed'] for r in recs))
print(Counter(r['section'] for r in recs).most_common(12))
