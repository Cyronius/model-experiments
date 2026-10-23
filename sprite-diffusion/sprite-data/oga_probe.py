# Measures how many unique <=32x32 sprites one OpenGameArt HF archive yields, per category.
import csv, glob, hashlib, io, json, os, random, re, sys, zipfile, zstandard
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spritekit import DATA, trim, render
import tsrkit

O = f'{DATA}/oga'
ZIP = sys.argv[1]
LIC = os.path.basename(ZIP).split('-2D_Art')[0]
CATS = [('food', r'\bfood|fruits?\b|vegetables?|\bmeat\b|bread|cake|cheese'),
        ('weapon', r'\bweapons?\b|swords?\b|\baxes?\b|\bbows?\b|spears?|daggers?|\bguns?\b|firearm|\bclub'),
        ('item', r'\bitems?\b|potions?|treasure|loot|inventory|\bcoins?\b|\bgems?\b|currency|\bbooks?\b|headgear|armou?r'),
        ('effect', r'\beffects?\b|explosions?|spells?\b|particles?|projectiles?'),
        ('creature', r'characters?|monsters?|enem(y|ies)|creatures?|animals?|\bhero|\bnpcs?\b|zombies?|slimes?'),
        ('terrain', r'tile ?sets?|\btiles?\b|terrain'),
        ('structure', r'buildings?|hous(e|es)\b|\bwalls?\b|doors?|furniture|castles?'),
        ('nature', r'\btrees?\b|bush(es)?|plants?|\brocks?\b|flowers?')]

rec = {}
for l in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(f'{O}/meta/{LIC}-2D_Art.jsonl.zst', 'rb')), encoding='utf8'):
    r = json.loads(l)
    for f in r['files']:
        rec[f['name']] = r


def category(r):
    t = (r['title'] + ' ' + ' '.join(r['tags'])).lower()
    return next((c for c, p in CATS if re.search(p, t)), 'other')


def unscale(a):
    for s in (8, 6, 5, 4, 3, 2):
        if a.shape[0] % s == 0 and a.shape[1] % s == 0 and a.shape[0] >= 8 * s:
            b = a[::s, ::s]
            if np.array_equal(np.repeat(np.repeat(b, s, 0), s, 1), a):
                return b, s
    return a, 1


def images(name, data, depth=0):
    ext = name.rsplit('.', 1)[-1].lower()
    if ext in ('png', 'gif', 'bmp'):
        yield name, data
    elif ext == 'zip' and depth < 2:
        try:
            z = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            return
        for n in z.namelist():
            if '__MACOSX' not in n and z.getinfo(n).file_size < 50e6:
                yield from images(n, z.read(n), depth + 1)


z = zipfile.ZipFile(ZIP)
seen, stats, samples = set(), defaultdict(Counter), defaultdict(list)
for k, top in enumerate(z.namelist()):
    r = rec.get(top)
    cat = category(r) if r else 'unmatched'
    stats[cat]['downloads'] += 1
    for n, data in images(top, z.read(top)):
        try:
            a = np.array(Image.open(io.BytesIO(data)).convert('RGBA'))
        except Exception:
            continue
        if a.shape[0] * a.shape[1] > 16e6:
            continue
        a, s = unscale(a)
        stats[cat]['images'] += 1
        stats[cat][f'upscaled x{s}'] += s > 1
        big = 0
        for c in tsrkit.sprites(a):
            c = trim(c)
            h = hashlib.md5(c.tobytes() + bytes(c.shape[:2])).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            stats[cat]['sprites'] += 1
            stats[cat]['<=16' if max(c.shape[:2]) <= 16 else '17-32'] += 1
            if len(samples[cat]) < 400:
                samples[cat].append(c)
    if k % 50 == 0:
        print(f'{k}/{len(z.namelist())}', flush=True)

lines = [f'# OGA probe: {os.path.basename(ZIP)}', '', '| category | downloads | images | unique sprites <=32 | <=16 | 17-32 | upscaled images |',
         '|---|---:|---:|---:|---:|---:|---:|']
for cat in sorted(stats, key=lambda c: -stats[c]['sprites']):
    s = stats[cat]
    up = sum(v for k, v in s.items() if k.startswith('upscaled'))
    lines.append(f"| {cat} | {s['downloads']} | {s['images']} | {s['sprites']} | {s['<=16']} | {s['17-32']} | {up} |")
lines.append(f"| total | {sum(s['downloads'] for s in stats.values())} | {sum(s['images'] for s in stats.values())} | "
             f"{sum(s['sprites'] for s in stats.values())} | | | |")
open(f'{O}/probe-{os.path.basename(ZIP)}.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
random.seed(0)
rows = [c for c in samples if samples[c]]
sheet = Image.new('RGB', (20 * 66, len(rows) * 80), (40, 40, 40))
from PIL import ImageDraw
d = ImageDraw.Draw(sheet)
for i, cat in enumerate(rows):
    for j, c in enumerate(random.sample(samples[cat], min(20, len(samples[cat])))):
        sheet.paste(render(c).resize((64, 64), Image.NEAREST), (j * 66 + 1, i * 80 + 14))
    d.text((2, i * 80 + 1), cat, fill=(0, 255, 255))
sheet.save(f'{O}/probe-{os.path.basename(ZIP)}.png')
