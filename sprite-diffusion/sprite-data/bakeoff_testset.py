# Bake-off test set: 20 course sprites with known class + 24 sprites cut from TSR sheets
# (sheet title as the hint). Writes 256x256 PNGs, meta.json and a numbered contact sheet
# to OUT. Reuses size_probe.py's crawler/cache and background mask.
import io, os, re, html, json, numpy as np
D = 'C:/dev/sprite-data'; OUT = f'{D}/bakeoff/ftest'; os.makedirs(OUT, exist_ok=True)
from PIL import Image
from scipy import ndimage
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'size_probe.py'), encoding='utf8').read().split('seen = set()')[0]
exec(src)
rng = np.random.default_rng(1)
items = []
# gold: course set, 4 unique per class
x = np.load(f'{D}/course/sprites.npy'); y = np.load(f'{D}/course/labels.npy').argmax(1)
names = ['front-facing character', 'monster/creature', 'food', 'item', 'side-facing character with weapon']
for c in range(5):
    idx = np.where(y == c)[0]; _, first = np.unique(x[idx].reshape(len(idx), -1), axis=0, return_index=True)
    for i in rng.choice(idx[first], 4, replace=False):
        items.append(dict(src='course', truth=names[c], rgba=np.dstack([x[i], np.full((16, 16), 255, np.uint8)])))
# TSR: 2-3 sprites from chosen sheets
want = [('nes', 'The Legend of Zelda', 'Overworld Enemies'), ('nes', 'Super Mario Bros. 3', 'Items and Objects'),
        ('snes', 'The Legend of Zelda: A Link to the Past', 'Hylian Knights'), ('snes', 'Chrono Trigger', 'Frog'),
        ('snes', 'EarthBound', 'Ness'), ('snes', 'Secret of Mana', 'Weapons'),
        ('snes', 'The Legend of Zelda: A Link to the Past', 'Taros & Moblin'), ('snes', 'Secret of Mana', 'Common NPCs')]
for plat, game, title in want:
    g = slug(plat, game); page = get(g)
    assets = {t: u for u, t in [(u, ' '.join(html.unescape(re.sub('<[^>]+>', ' ', t)).split()))
              for u, t in re.findall(rf'href="({re.escape(g)}asset/\d+/)"[^>]*>(.*?)</a>', page, re.S)]}
    u = assets.get(title)
    if not u: print('missing', game, title); continue
    m = re.search(r'src="(/media/assets/[^"?]+\.png)', get(u)); a = np.array(Image.open(io.BytesIO(get(m.group(1), binary=True))).convert('RGBA'))
    fg = ~bg_mask(a); lab, n = ndimage.label(fg, structure=np.ones((3, 3)))
    cands = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start; mm = lab[sl] == i
        if not (8 <= h <= 32 and 8 <= w <= 32) or mm.sum() < 40: continue
        if len(np.unique(a[sl][mm][:, :3], axis=0)) < 3: continue
        crop = a[sl].copy(); crop[~mm] = 0; crop[mm, 3] = 255; cands.append(crop)
    for k in rng.choice(len(cands), min(3, len(cands)), replace=False):
        items.append(dict(src=f'{plat}:{game[:22]}', truth=f'sheet "{title}"', rgba=cands[k]))
# render each on a 32x32 canvas, bottom-centred, 8x nearest on grey
meta = []
for j, it in enumerate(items):
    r = it['rgba']; h, w = r.shape[:2]; can = np.zeros((32, 32, 4), np.uint8)
    can[32 - h:, (32 - w) // 2:(32 - w) // 2 + w] = r if it['src'] != 'course' else r
    if it['src'] == 'course':  # course sprites are 16x16 opaque on white: show at 16x scale instead
        im = Image.fromarray(r[..., :3]).resize((256, 256), Image.NEAREST)
    else:
        bg = Image.new('RGB', (32, 32), (128, 128, 128)); bg.paste(Image.fromarray(can), (0, 0), Image.fromarray(can))
        im = bg.resize((256, 256), Image.NEAREST)
    im.save(f'{OUT}/{j:02d}.png'); meta.append(dict(id=j, src=it['src'], truth=it['truth']))
json.dump(meta, open(f'{OUT}/meta.json', 'w'), indent=0)
# contact sheet with numbers
N = len(meta); cols = 10; rows = (N + cols - 1) // cols
sheet = Image.new('RGB', (cols * 132, rows * 132), (60, 60, 60))
from PIL import ImageDraw
d = ImageDraw.Draw(sheet)
for j in range(N):
    im = Image.open(f'{OUT}/{j:02d}.png').resize((128, 128), Image.NEAREST)
    sheet.paste(im, ((j % cols) * 132 + 2, (j // cols) * 132 + 2)); d.text(((j % cols) * 132 + 4, (j // cols) * 132 + 3), str(j), fill=(255, 255, 0))
sheet.save(f'{D}/bakeoff/ftest_sheet.png'); print(N, 'sprites')
