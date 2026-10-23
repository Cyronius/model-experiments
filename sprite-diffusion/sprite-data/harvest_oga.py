# Cuts the OpenGameArt HF archives into 32x32 sprites (`cut`, system python) and labels them with so400m (`label`, igpu-venv); 2026-10-16.
import argparse, glob, hashlib, io, json, os, random, re, shutil, sys, time, zipfile
from collections import Counter, defaultdict
from urllib.parse import unquote
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spritekit import DATA, trim, footprint, on_canvas, render

O = f'{DATA}/oga'
CUT = f'{O}/cut'
FLOOR_GB = 10
PACK_CAP = 500
CATS = [('food', r'\bfood|fruits?\b|vegetables?|\bmeat\b|bread|cake|cheese|potions?|drinks?|beverages?|elixirs?|flasks?|vials?|juice|\btea\b|coffee|\bwine\b|\bbeer\b|\bmilk\b'),
        ('weapon', r'\bweapons?\b|swords?\b|\baxes?\b|\bbows?\b|spears?|daggers?|\bguns?\b|firearm|\bclubs?\b'),
        ('item', r'\bitems?\b|treasure|loot|inventory|\bcoins?\b|\bgems?\b|currency|\bbooks?\b|headgear|armou?r|\bkeys?\b'),
        ('effect', r'\beffects?\b|explosions?|spells?\b|particles?|projectiles?'),
        ('creature', r'characters?|monsters?|enem(y|ies)|creatures?|animals?|\bhero|\bnpcs?\b|zombies?|slimes?'),
        ('terrain', r'tile ?sets?|\btiles?\b|terrain'),
        ('structure', r'buildings?|hous(e|es)\b|\bwalls?\b|doors?|furniture|castles?'),
        ('nature', r'\btrees?\b|bush(es)?|plants?|\brocks?\b|flowers?'),
        ('ui', r'\bui\b|\bgui\b|interface|buttons?|\bhud\b|\bfonts?\b|cursors?|\bframes?\b')]
LABELS = {'creature', 'item', 'weapon', 'food', 'effect', 'terrain', 'structure', 'nature'}


def cats_in(text):
    t = text.lower()
    return {c for c, p in CATS if re.search(p, t)}


def seed(pack, path):
    # A pack names one category, or failing that the file/folder name does (plan 1e).
    c = cats_in(pack['title'] + ' ' + ' '.join(pack['tags']))
    if len(c) == 1:
        return c.pop()
    c2 = cats_in(re.sub(r'[_\-./]+', ' ', path))
    if len(c2) == 1:
        return c2.pop()
    return None


def free_gb():
    return shutil.disk_usage('C:/').free / 1e9


def wait_for_disk():
    while free_gb() < FLOOR_GB + 1:
        print(f'{time.strftime("%H:%M")} paused: {free_gb():.1f} GB free, floor {FLOOR_GB} GB', flush=True)
        time.sleep(600)


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
            for n in z.namelist():
                if '__MACOSX' not in n and z.getinfo(n).file_size < 50e6:
                    yield from images(n, z.read(n), depth + 1)
        except (zipfile.BadZipFile, NotImplementedError, RuntimeError, OSError):
            return


def load_packs(lic):
    import zstandard
    by_name = {}
    f = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(f'{O}/meta/{lic}-2D_Art.jsonl.zst', 'rb')), encoding='utf8')
    for l in f:
        r = json.loads(l)
        for fl in r['files']:
            by_name.setdefault(fl['name'], r)
            by_name.setdefault(unquote(fl['url'].rsplit('/', 1)[-1]), r)   # archive members use the URL's file name
    return by_name


def cut():
    import tsrkit
    os.makedirs(CUT, exist_ok=True)
    manifest = [l.split() for l in open(f'{O}/zips/manifest.txt') if l.strip()]
    seen, per_pack = set(), Counter()
    for f in (f for f in glob.glob(f'{CUT}/*.jsonl') if not f.endswith('.labels.jsonl')):
        for r in map(json.loads, open(f, encoding='utf8')):
            seen.add(r['hash']); per_pack[r['pack']] += 1
    packs_lic = {}
    for lic, part, _ in manifest:
        shard = f'{CUT}/{lic}-{part[:-4]}'
        if os.path.exists(shard + '.npz'):
            continue
        wait_for_disk()
        if lic not in packs_lic:
            packs_lic = {lic: load_packs(lic)}
        packs = packs_lic[lic]
        z = zipfile.ZipFile(f'{O}/zips/{lic}-{part}')
        arrs, recs, unmatched = [], [], 0
        for top in z.namelist():
            pack = packs.get(top)
            if pack is None:
                unmatched += 1
                continue
            if per_pack[pack['url']] >= PACK_CAP:
                continue
            cand = []
            for path, data in images(top, z.read(top)):
                try:
                    im = Image.open(io.BytesIO(data))
                    if im.width * im.height > 16e6:
                        continue
                    a = np.array(im.convert('RGBA'))
                except Exception:
                    continue
                a, s = unscale(a)
                for c in tsrkit.sprites(a):
                    cand.append((path, s, trim(c)))
            random.Random(top).shuffle(cand)   # so the cap samples the whole pack, not its first sheet
            for path, s, c in cand:
                if per_pack[pack['url']] >= PACK_CAP:
                    break
                h = hashlib.md5(c.tobytes() + bytes(c.shape[:2])).hexdigest()
                if h in seen:
                    continue
                seen.add(h); per_pack[pack['url']] += 1
                arrs.append(on_canvas(c))
                recs.append(dict(id=f'oga-{h[:12]}', hash=h, lic=lic, archive=part, member=top, path=path,
                                 asset=f'{part}:{top}:{path}', pack=pack['url'], title=pack['title'],
                                 tags=pack['tags'], author=pack['author'], seed=seed(pack, top + ' ' + path),
                                 upscale=s, footprint=footprint(c)))
        with open(shard + '.jsonl', 'w', encoding='utf8') as fh:
            for r in recs:
                fh.write(json.dumps(r) + '\n')
        # Written last: the .npz marks the archive done.
        np.savez_compressed(shard + '.npz', sprites=np.stack(arrs) if arrs else np.zeros((0, 32, 32, 4), np.uint8))
        print(f'{time.strftime("%H:%M")} {lic}-{part}: {len(recs)} sprites, {unmatched} unmatched files, '
              f'{dict(Counter(r["seed"] for r in recs))}, {free_gb():.1f} GB free', flush=True)
    print('cut complete', flush=True)


DIBS = os.environ.get('DIBS_EXE', 'dibs')


def gpu_wanted():
    # Another session waits on the gpu lock we hold: the user asked us to yield (exit 75, resumes mid-archive).
    import subprocess
    try:
        out = subprocess.run([DIBS, 'status'], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return False
    return any(l.startswith('gpu') and 'waiting:' in l for l in out.splitlines())


def label(wait, once=False, yield_gpu=False):
    os.environ.setdefault('SIGLIP', 'google/siglip2-so400m-patch16-256')
    import torch, siglipkit
    CAT_PROMPTS, MERGE = siglipkit.CAT_PROMPTS, siglipkit.MERGE
    text_p = ['text', 'letters', 'a word written in a font', 'a single letter of the alphabet', 'a number digit']
    obj_p = ['a person', 'a monster', 'an item', 'a weapon', 'food', 'a magic effect', 'a tree', 'a wall',
             'a creature', 'a sword', 'a potion']
    cats = list(CAT_PROMPTS)
    T = np.stack([siglipkit.embed_text([f'pixel art sprite of {p}' for p in CAT_PROMPTS[c]]).mean(0) for c in cats])
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    Tt = siglipkit.embed_text([f'pixel art sprite of {p}' for p in text_p])
    To = siglipkit.embed_text([f'pixel art sprite of {p}' for p in obj_p])
    m, proc = siglipkit.model()
    n_arch = len([l for l in open(f'{O}/zips/manifest.txt') if l.strip()])
    while True:
        todo = [f[:-4] for f in sorted(glob.glob(f'{CUT}/*.npz')) if '.emb-' not in f and not os.path.exists(f[:-4] + '.labels.jsonl')]
        if not todo:
            if not wait or len(glob.glob(f'{CUT}/*.labels.jsonl')) >= n_arch:
                break
            time.sleep(120)
            continue
        for shard in todo:
            wait_for_disk()
            S = np.load(shard + '.npz')['sprites']
            recs = [json.loads(l) for l in open(shard + '.jsonl', encoding='utf8')]
            E = []
            part = shard + '.emb-partial.npy'
            if os.path.exists(part):
                E = [np.load(part)]
            for j in range(sum(len(e) for e in E), len(S), 32):
                if yield_gpu and j % 256 == 0 and j and gpu_wanted():
                    np.save(part, np.concatenate(E)) if E else None
                    print(f'{time.strftime("%H:%M")} yielding the gpu at {os.path.basename(shard)} {j}/{len(S)}', flush=True)
                    sys.exit(75)
                x = proc(images=[render(a) for a in S[j:j + 32]], return_tensors='pt').to(siglipkit.dev)
                with torch.no_grad():
                    E.append(torch.nn.functional.normalize(siglipkit._feat(m.get_image_features(
                        pixel_values=x['pixel_values'].half())).float(), dim=-1).cpu().numpy())
                if j % 2048 == 0:
                    print(f'{time.strftime("%H:%M")}   {os.path.basename(shard)} {j}/{len(S)}', flush=True)
            E = np.concatenate(E) if E else np.zeros((0, T.shape[1]), np.float32)
            np.savez_compressed(shard + f'.emb-{siglipkit.TAG}.npz', E=E.astype(np.float16))
            Sc = E @ T.T
            top = Sc.argsort(1)[:, ::-1]
            text_margin = (E @ Tt.T).max(1) - (E @ To.T).max(1)
            with open(shard + '.labels.jsonl.tmp', 'w', encoding='utf8') as fh:
                for i, r in enumerate(recs):
                    zs = MERGE.get(cats[top[i, 0]], cats[top[i, 0]])
                    fh.write(json.dumps(dict(id=r['id'], seed=r['seed'], zs=zs,
                                             margin=round(float(Sc[i, top[i, 0]] - Sc[i, top[i, 1]]), 4),
                                             text_margin=round(float(text_margin[i]), 4),
                                             label=r['seed'] if r['seed'] in LABELS and r['seed'] == zs else None)) + '\n')
            os.replace(shard + '.labels.jsonl.tmp', shard + '.labels.jsonl')
            if os.path.exists(part):
                os.replace(part, part + '.done')
            print(f'{time.strftime("%H:%M")} labelled {os.path.basename(shard)}: {len(recs)} sprites', flush=True)
            if once:
                sys.exit(0)
    print('label complete', flush=True)
    sys.exit(3 if once else 0)


def report():
    n, lab, seeds, des = Counter(), Counter(), Counter(), 0
    for f in sorted(glob.glob(f'{CUT}/*.labels.jsonl')):
        for r in map(json.loads, open(f, encoding='utf8')):
            n['all'] += 1
            seeds[r['seed']] += 1
            if r['label']:
                lab[r['label']] += 1
    lines = ['# OpenGameArt harvest', '', f"Sprites labelled so far: {n['all']}.", '',
             '| category | seeded | labelled (seed = so400m) |', '|---|---:|---:|']
    for c in sorted(set(seeds) | set(lab), key=lambda c: -seeds[c]):
        lines.append(f'| {c} | {seeds[c]} | {lab[c]} |')
    open(f'{O}/report-harvest.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['cut', 'label', 'report'])
    ap.add_argument('--wait', action='store_true', help='label: keep polling until every archive is cut and labelled')
    ap.add_argument('--once', action='store_true', help='label: one archive, then exit (3 = nothing left)')
    ap.add_argument('--yield-gpu', action='store_true', help='label: exit 75 when another session waits on the gpu lock')
    a = ap.parse_args()
    {'cut': cut, 'label': lambda: label(a.wait, a.once, a.yield_gpu), 'report': report}[a.stage]()
