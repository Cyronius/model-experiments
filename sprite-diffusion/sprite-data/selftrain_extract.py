# Cuts the crawled self-training sheets (DATA/selftrain/sheets.jsonl) into sprites:
# components that fit the 32x32 canvas, at most --per-sheet per sheet, deduplicated across
# the whole set. Each sprite carries its sheet's seed (category) and role.
# Safe to rerun at any point: it rebuilds from the cache and skips sheets it has done.
# Writes DATA/selftrain/<--out> (default sprites.jsonl), sprites/<id>.png and render/<id>.png.
import argparse, hashlib, io, json, os, random, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spritekit import DATA, trim, footprint, on_canvas, render
import tsrkit

ap = argparse.ArgumentParser()
ap.add_argument('--per-sheet', type=int, default=30)
ap.add_argument('--out', default='sprites.jsonl', help='a new name re-cuts from scratch (ids are content hashes)')
args = ap.parse_args()
D = f'{DATA}/selftrain'
os.makedirs(f'{D}/sprites', exist_ok=True); os.makedirs(f'{D}/render', exist_ok=True)
out = f'{D}/{args.out}'
done_sheets, seen = set(), set()
if os.path.exists(out):
    for l in open(out):
        r = json.loads(l); done_sheets.add(r['asset']); seen.add(r['hash'])
sheets = [json.loads(l) for l in open(f'{D}/sheets.jsonl')]
n_new = 0
with open(out, 'a') as fh:
    for k, s in enumerate(sheets):
        if s['asset'] in done_sheets:
            continue
        png = tsrkit.sheet_png(s['asset'])          # cached by the crawl
        if not png:
            continue
        a = np.array(Image.open(io.BytesIO(png)).convert('RGBA'))
        if a.shape[0] * a.shape[1] > 8e6:
            continue
        cand = list(tsrkit.sprites(a))
        random.Random(s['asset']).shuffle(cand)
        kept = 0
        for c in cand:
            c = trim(c)
            h = hashlib.md5(c.tobytes() + bytes(c.shape[:2])).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            sid = f'st-{h[:12]}'
            Image.fromarray(on_canvas(c)).save(f'{D}/sprites/{sid}.png')
            render(c).save(f'{D}/render/{sid}.png')
            fh.write(json.dumps(dict(id=sid, hash=h, asset=s['asset'], platform=s['platform'], game=s['game'],
                                     section=s['section'], title=s['title'], seed=s['seed'], role=s['role'],
                                     footprint=footprint(c))) + '\n')
            kept += 1; n_new += 1
            if kept >= args.per_sheet:
                break
        if kept == 0:     # remember empty sheets too, so a rerun skips them
            fh.write(json.dumps(dict(id=None, hash=f'empty:{s["asset"]}', asset=s['asset'])) + '\n')
        if k % 200 == 0:
            print(f'{k}/{len(sheets)} sheets, {n_new} new sprites', flush=True)
print(f'done: {n_new} new sprites')
