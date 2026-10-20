# Free itch.io food packs -> sprite shards in oga/cut (labelled later by harvest_oga.py label); skips packs that opt out of AI; 2026-10-18.
import argparse, glob, hashlib, html, io, json, os, random, re, sys, time, zipfile
from collections import Counter
import numpy as np
import requests
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spritekit import DATA, trim, footprint, on_canvas
from harvest_oga import images, unscale, cats_in, free_gb, PACK_CAP

I = f'{DATA}/itch'
CUT = f'{DATA}/oga/cut'
TAGS = ['food', 'fruit', 'vegetables', 'cooking', 'ingredients', 'sweets', 'candy', 'bread', 'drinks', 'kitchen']
FOODISH = re.compile(r'\bfoods?\b|fruits?|vegetables?|veg(gie)?s?\b|\bmeat\b|bread|cake|cheese|cooking|ingredients?|sweets?|candy|dessert|pizza|burger|'
                     r'cookies?|donuts?|drinks?|snacks?|meals?|pastr(y|ies)|berries|restaurant|kitchen|\bpixelfood\b', re.I)
NO_AI = re.compile(r'\bno[ -]?ai\b|\bai[ -]?free\b|not (be )?used (to|for) (train|ai)|ai training|machine[ -]learning|generative|'
                   r'(do not|don.t|no) (use|scrape)[^.]{0,40}\b(ai|dataset)', re.I)
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
DELAY = 2.5
_last = [0.0]
S = requests.Session(); S.headers.update(UA)


def get(url, **kw):
    wait = DELAY - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    method = kw.pop('method', 'GET')
    for attempt in range(6):
        _last[0] = time.time()
        r = S.request(method, url, timeout=60, **kw)
        if r.headers.get('cf-mitigated'):
            raise RuntimeError(f'Cloudflare challenge on {url}; stopped')
        if r.status_code != 429:
            return r
        time.sleep(30 * (attempt + 1))   # itch rate-limits; back off rather than push
        kw.get('stream') and r.close()
    return r


def cached_page(url):
    fn = f'{I}/cache/{hashlib.md5(url.encode()).hexdigest()}.html'
    if os.path.exists(fn):
        return open(fn, encoding='utf8').read()
    r = get(url)
    if r.status_code != 200:
        return None
    os.makedirs(f'{I}/cache', exist_ok=True)
    open(fn, 'w', encoding='utf8').write(r.text)
    return r.text


def discover():
    packs = {}
    out = f'{I}/packs.jsonl'
    if os.path.exists(out):
        packs = {r['url']: r for r in map(json.loads, open(out, encoding='utf8'))}
    for tag in TAGS:
        for page in range(1, 40):
            t = cached_page(f'https://itch.io/game-assets/free/tag-{tag}' + (f'?page={page}' if page > 1 else ''))
            urls = [m.group(1) for a in re.findall(r'<a ([^>]*class="title game_link"[^>]*)>', t or '') if (m := re.search(r'href="(https://[^"/]+\.itch\.io/[^"]+)"', a))]
            if not urls:
                break
            new = [u for u in urls if u not in packs]
            for u in new:
                packs[u] = dict(url=u, found_by=tag)
            print(f'tag {tag} page {page}: {len(urls)} packs, {len(new)} new', flush=True)
    os.makedirs(I, exist_ok=True)
    with open(out, 'w', encoding='utf8') as fh:
        for r in packs.values():
            fh.write(json.dumps(r) + '\n')
    print(f'{len(packs)} candidate packs')


def pack_info(url):
    t = cached_page(url)
    if t is None:
        return None
    title = html.unescape(re.sub(r'\s+', ' ', (re.search(r'<title>(.*?)</title>', t, re.S) or [0, ''])[1])).strip()
    tags = sorted(set(re.findall(r'href="https://itch\.io/game-assets/tag-([a-z0-9-]+)"', t)))
    desc = re.search(r'<div class="formatted_description[^"]*">(.*?)</div>\s*</div>', t, re.S)
    text = html.unescape(re.sub(r'<[^>]+>', ' ', desc.group(1))) if desc else ''
    lic = re.search(r'(Creative Commons[^<\n]{0,60}|CC0[^<\n]{0,40}|CC[- ]BY[^<\n]{0,30}|MIT License)', t)
    return dict(url=url, title=title, tags=tags, text=text[:4000], license=lic.group(1).strip() if lic else 'unspecified (free download page)',
                no_ai=bool(('no-ai' in tags) or NO_AI.search(title + ' ' + text)), csrf=(re.search(r'name="csrf_token" value="([^"]+)"', t) or [0, None])[1])


def download(info):
    base = info['url'].rstrip('/')
    slug = re.sub(r'[^a-z0-9]+', '-', base.split('//')[1].lower()).strip('-')
    d = f'{I}/raw/{slug}'
    if os.path.exists(d + '/.done'):
        return d
    if not info['csrf']:
        return None
    r = get(base + '/download_url', method='POST', data={'csrf_token': info['csrf']})
    if r.status_code != 200 or 'url' not in r.json():
        return None
    dl = get(r.json()['url']).text
    csrf = re.search(r'name="csrf_token" value="([^"]+)"', dl)
    uploads = re.findall(r'data-upload_id="(\d+)"[^>]*>Download.*?title="([^"]+)"', dl, re.S)
    if not csrf or not uploads:
        return None
    os.makedirs(d, exist_ok=True)
    for uid, name in uploads:
        if not name.lower().endswith(('.zip', '.png', '.gif')):
            continue
        j = get(f'{base}/file/{uid}', method='POST', params={'source': 'game_download', 'as_props': 1, 'after_download_lightbox': 'true'},
                data={'csrf_token': csrf.group(1)}).json()
        if 'url' not in j:
            continue
        fn = f'{d}/{re.sub(r"[^A-Za-z0-9._-]+", "_", name)}'
        with get(j['url'], stream=True) as resp:
            size = 0
            with open(fn, 'wb') as fh:
                for chunk in resp.iter_content(1 << 20):
                    size += len(chunk); fh.write(chunk)
                    if size > 200e6:
                        break
    open(d + '/.done', 'w').close()
    return d


def cut_pack(info, d, seen):
    import tsrkit
    shard = f'{CUT}/ITCH-{os.path.basename(d)}'
    if os.path.exists(shard + '.npz'):
        return 0
    cand = []
    for fn in glob.glob(f'{d}/*'):
        if fn.endswith('.done'):
            continue
        for path, data in images(os.path.basename(fn), open(fn, 'rb').read()):
            try:
                im = Image.open(io.BytesIO(data))
                if im.width * im.height > 16e6:
                    continue
                a = np.array(im.convert('RGBA'))
            except Exception:
                continue
            a, s = unscale(a)
            cand += [(path, s, trim(c)) for c in tsrkit.sprites(a)]
    random.Random(d).shuffle(cand)
    arrs, recs = [], []
    for path, s, c in cand:
        if len(recs) >= PACK_CAP:
            break
        h = hashlib.md5(c.tobytes() + bytes(c.shape[:2])).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        arrs.append(on_canvas(c))
        recs.append(dict(id=f'itch-{h[:12]}', hash=h, lic=info['license'], archive='itch', member=os.path.basename(d), path=path,
                         asset=f'itch:{os.path.basename(d)}:{path}', pack=info['url'], title=info['title'], tags=info['tags'],
                         author=info['url'].split('//')[1].split('.')[0], seed='food', upscale=s, footprint=footprint(c)))
    with open(shard + '.jsonl', 'w', encoding='utf8') as fh:
        for r in recs:
            fh.write(json.dumps(r) + '\n')
    np.savez_compressed(shard + '.npz', sprites=np.stack(arrs) if arrs else np.zeros((0, 32, 32, 4), np.uint8))
    return len(recs)


def run():
    discover()
    seen = set()
    for f in (f for f in glob.glob(f'{CUT}/*.jsonl') if not f.endswith('.labels.jsonl')):
        seen.update(r['hash'] for r in map(json.loads, open(f, encoding='utf8')))
    stats, rows = Counter(), []
    for r in map(json.loads, open(f'{I}/packs.jsonl', encoding='utf8')):
        if free_gb() < 20:
            print('stopping: under 20 GB free'); break
        info = pack_info(r['url'])
        if info is None:
            stats['page missing'] += 1; continue
        if info['no_ai']:
            stats['skipped: opts out of AI'] += 1; rows.append((info['url'], 'no-ai')); continue
        if not FOODISH.search(info['title'] + ' ' + ' '.join(info['tags'])):
            stats['skipped: not food by title/tags'] += 1; continue
        d = download(info)
        if d is None:
            stats['skipped: not a free download'] += 1; continue
        n = cut_pack(info, d, seen)
        stats['packs cut'] += 1; stats['sprites'] += n
        print(f"{info['title'][:60]:60s} {n:4d} sprites  {info['license'][:30]}", flush=True)
    json.dump(dict(stats), open(f'{I}/stats.json', 'w'), indent=1)
    print(dict(stats))


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args()
    run()
