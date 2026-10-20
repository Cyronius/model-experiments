# Builds the automatic answer keys for the labeller bake-off from data that labels itself:
#   - the course set (sprites_1788_16x16.npy): category, and facing for its front/side classes
#   - DawnLike: category (and subject where a sheet holds one kind of thing) from sheet names
#   - Ninja Adventure: category/subject from folders and file names; facing and pose from
#     the walk/idle/attack sheets, whose columns are front/back/left/right
#   - 0x72 DungeonTilesetII: category, subject and pose from frame names; its heroes face right
# Writes DATA/bakeoff/corpus/<id>.png (32x32 canvas), .../render/<id>.png (what labellers
# see), corpus.jsonl (every sprite) and marks a balanced evaluation sample with "eval": true.
import os, re, glob, json, hashlib, random
import numpy as np
from collections import defaultdict
from spritekit import DATA, rgba, trim, has_content, footprint, on_canvas, render, cells

P = f'{DATA}/packs'
OUT = f'{DATA}/bakeoff'
NINJA = f'{P}/ninja/Ninja Adventure - Asset Pack'
DT2 = f'{P}/dt2/0x72_DungeonTilesetII_v1.7/frames'
DAWN = f'{P}/dawnlike'
FACING4 = ['front', 'back', 'left', 'right']

recs, seen = [], set()


def words(name):
    name = re.sub(r'\d+$', '', name).replace('_', ' ')
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name).lower().strip()


def add(a, source, group, category, subject=None, facing=None, pose=None):
    a = trim(a)
    if a is None or not has_content(a):
        return
    fp = footprint(a)
    if fp is None:
        return
    key = hashlib.md5(a.tobytes() + bytes(a.shape[:2])).hexdigest()
    if key in seen:
        return
    seen.add(key)
    recs.append(dict(id=f'{source}-{len(recs):05d}', source=source, group=group, category=category,
                     subject=subject, facing=facing, pose=pose, footprint=fp, _a=a))


# --- course set -------------------------------------------------------------------------
x = np.load(f'{DATA}/course/sprites.npy'); y = np.load(f'{DATA}/course/labels.npy').argmax(1)
_, first = np.unique(x.reshape(len(x), -1), axis=0, return_index=True)
course = [('character', 'front'), ('monster', None), ('food', None), ('item', None), ('character', 'side')]
for i in sorted(first):
    im = x[i]
    alpha = np.where((im > 245).all(-1), 0, 255).astype(np.uint8)   # white background -> transparent
    cat, fac = course[y[i]]
    add(np.dstack([im, alpha]), 'course', f'course:{i}', cat, facing=fac)

# --- DawnLike ---------------------------------------------------------------------------
DAWN_CAT = {
    'Player': 'character', 'Humanoid': 'character',
    **{k: 'monster' for k in ['Aquatic', 'Avian', 'Cat', 'Dog', 'Demon', 'Elemental', 'Misc', 'Pest',
                              'Plant', 'Quadraped', 'Reptile', 'Rodent', 'Slime', 'Undead']},
    'Food': 'food', 'Flesh': 'food',
    **{k: 'weapon' for k in ['LongWep', 'MedWep', 'ShortWep', 'Ammo']},
    **{k: 'item' for k in ['Shield', 'Armor', 'Amulet', 'Book', 'Boot', 'Chest', 'Glove', 'Hat', 'Key',
                           'Light', 'Money', 'Music', 'Potion', 'Ring', 'Rock', 'Scroll', 'Tool', 'Wand']},
    **{k: 'terrain' for k in ['Floor', 'Ground', 'Tile', 'Hill', 'Pit', 'Map']},
    **{k: 'structure' for k in ['Wall', 'Door', 'Fence', 'Decor', 'Trap']},
    'Tree': 'nature', 'Effect': 'effect',   # Ore cells are scattered sparkle fragments
}
DAWN_SUBJ = {'Slime': 'slime', 'Cat': 'cat', 'Dog': 'dog', 'Avian': 'bird', 'Rodent': 'rat',
             'Potion': 'potion', 'Scroll': 'scroll', 'Ring': 'ring', 'Key': 'key', 'Boot': 'boot',
             'Hat': 'hat', 'Shield': 'shield', 'Book': 'book', 'Amulet': 'amulet', 'Money': 'coin',
             'Glove': 'glove', 'Armor': 'armor', 'Chest': 'chest', 'Tree': 'tree', 'Door': 'door',
             'Fence': 'fence', 'Wall': 'wall'}
for f in sorted(glob.glob(f'{DAWN}/*/*.png')):
    base = re.sub(r'\d$', '', os.path.splitext(os.path.basename(f))[0])
    if base not in DAWN_CAT or f.endswith('1.png'):      # *1.png is the second animation frame
        continue
    for r, c, cell in cells(rgba(f)):
        add(cell, 'dawnlike', f'dawnlike:{base}', DAWN_CAT[base], subject=DAWN_SUBJ.get(base))

# --- Ninja Adventure --------------------------------------------------------------------
def strip_frames(a, h=None):
    """Split a horizontal animation strip into square-ish frames of its height."""
    h = h or a.shape[0]
    if a.shape[1] % h:
        return [a]
    return [a[:, i * h:(i + 1) * h] for i in range(a.shape[1] // h)]

for d in sorted(glob.glob(f'{NINJA}/Actor/Character/*/SeparateAnim')):
    name = os.path.basename(os.path.dirname(d))
    for anim, pose in [('Walk', 'walk'), ('Idle', 'idle'), ('Attack', 'attack')]:
        f = f'{d}/{anim}.png'
        if not os.path.exists(f):
            continue
        for r, c, cell in cells(rgba(f)):
            if r < 2:
                add(cell, 'ninja', f'ninja:{name}', 'character', facing=FACING4[c], pose=pose)
for f in sorted(glob.glob(f'{NINJA}/Actor/Monster/*/SpriteSheet.png')):
    name = os.path.basename(os.path.dirname(f))
    for r, c, cell in cells(rgba(f)):
        if r < 2:
            add(cell, 'ninja', f'ninja:{name}', 'monster', subject=words(name), facing=FACING4[c], pose='walk')
for f in sorted(glob.glob(f'{NINJA}/Actor/Animal/*/*.png')):
    if 'Faceset' in f:
        continue
    name = os.path.basename(os.path.dirname(f))
    a = rgba(f)
    for fr in strip_frames(a, 16)[:2] if a.shape[0] == 16 else [a]:
        add(fr, 'ninja', f'ninja:{name}', 'monster', subject=words(name))
ITEM_CAT = {'Food': 'food', 'Projectile': 'weapon', 'Tool': 'item', 'Potion': 'item', 'Treasure': 'item',
            'Resource': 'item', 'Scroll': 'item', 'Object': 'item'}
for sub, cat in ITEM_CAT.items():
    for f in sorted(glob.glob(f'{NINJA}/Items/{sub}/*.png')):
        stem = os.path.splitext(os.path.basename(f))[0]
        add(strip_frames(rgba(f))[0], 'ninja', f'ninja:items-{sub}', cat, subject=words(stem))
for f in sorted(glob.glob(f'{NINJA}/Items/Weapons/*/Sprite*.png')):
    name = os.path.basename(os.path.dirname(f))
    add(rgba(f), 'ninja', 'ninja:items-Weapons', 'weapon', subject=words(name))
for f in sorted(glob.glob(f'{NINJA}/FX/*/*/*.png')):
    name = os.path.basename(os.path.dirname(f))
    frames = strip_frames(rgba(f))
    add(frames[len(frames) // 2], 'ninja', f'ninja:fx-{name}', 'effect', subject=words(name))
TILE_CAT = {'TilesetField': 'terrain', 'TilesetFloor': 'terrain', 'TilesetFloorB': 'terrain',
            'TilesetDesert': 'terrain', 'TilesetWater': 'terrain', 'TilesetHole': 'terrain',
            'TilesetRelief': 'terrain', 'TilesetHouse': 'structure', 'TilesetTowers': 'structure',
            'TilesetVillageAbandoned': 'structure', 'TilesetDungeon': 'structure', 'TilesetNature': 'nature'}
for name, cat in TILE_CAT.items():
    f = f'{NINJA}/Backgrounds/Tilesets/{name}.png'
    if os.path.exists(f):
        for r, c, cell in cells(rgba(f)):
            add(cell, 'ninja', f'ninja:{name}', cat)

# --- 0x72 DungeonTilesetII --------------------------------------------------------------
HEROES = {'knight': 'knight', 'elf': 'elf', 'wizzard': 'wizard', 'lizard': 'lizard man', 'dwarf': 'dwarf'}
MONSTERS = {'big_demon': 'demon', 'big_zombie': 'zombie', 'chort': 'chort', 'goblin': 'goblin', 'imp': 'imp',
            'ice_zombie': 'ice zombie', 'masked_orc': 'masked orc', 'muddy': 'mud monster',
            'necromancer': 'necromancer', 'ogre': 'ogre', 'orc_shaman': 'orc shaman', 'orc_warrior': 'orc warrior',
            'pumpkin_dude': 'pumpkin head', 'skelet': 'skeleton', 'slug': 'slug', 'swampy': 'swamp monster',
            'tiny_slug': 'slug', 'tiny_zombie': 'zombie', 'wogol': 'wogol', 'zombie': 'zombie',
            'doc': 'plague doctor', 'angel': 'angel'}
POSE = {'idle': 'idle', 'run': 'walk', 'hit': 'hurt'}
for f in sorted(glob.glob(f'{DT2}/*.png')):
    n = os.path.splitext(os.path.basename(f))[0]
    a = rgba(f)
    m = re.match(r'(.+?)(?:_([mf]))?_(idle|run|hit)_anim_f(\d)$', n)
    if m:
        base, _, act, fr = m.groups()
        if base in HEROES:
            add(a, 'dt2', f'dt2:{base}', 'character', subject=HEROES[base], facing='right', pose=POSE[act])
        elif base in MONSTERS:
            add(a, 'dt2', f'dt2:{base}', 'monster', subject=MONSTERS[base], pose=POSE[act])
        continue
    base = re.sub(r'(_anim)?(_f\d)?$', '', n)
    if base in MONSTERS:
        add(a, 'dt2', f'dt2:{base}', 'monster', subject=MONSTERS[base])
    elif n.startswith('weapon_'):
        add(a, 'dt2', 'dt2:weapon', 'weapon', subject=words(n[7:]))
    elif n.startswith('flask_'):
        add(a, 'dt2', 'dt2:flask', 'item', subject='potion')
    elif n.startswith('chest_'):
        add(a, 'dt2', 'dt2:chest', 'item', subject='chest')
    elif n.startswith('coin'):
        add(a, 'dt2', 'dt2:coin', 'item', subject='coin')
    elif n in ('skull',):
        add(a, 'dt2', 'dt2:skull', 'item', subject='skull')
    elif n == 'bomb':
        add(a, 'dt2', 'dt2:bomb', 'weapon', subject='bomb')
    elif n.startswith('floor_'):
        add(a, 'dt2', 'dt2:floor', 'terrain')
    elif n.startswith(('wall_', 'doors_', 'column', 'crate', 'lever', 'button', 'hole', 'edge')):
        add(a, 'dt2', 'dt2:wall', 'structure')

# --- write ------------------------------------------------------------------------------
os.makedirs(f'{OUT}/corpus', exist_ok=True); os.makedirs(f'{OUT}/render', exist_ok=True)
random.seed(0)
by_cat = defaultdict(list)
for r in recs:
    by_cat[r['category']].append(r)
# Balanced evaluation sample: up to 50 per category, round-robin across sources and groups.
for cat, rs in by_cat.items():
    random.shuffle(rs)
    per_group = defaultdict(list)
    for r in rs:
        per_group[r['group']].append(r)
    groups = list(per_group.values()); picked = []
    while len(picked) < 50 and any(groups):
        for g in groups:
            if g and len(picked) < 50:
                picked.append(g.pop())
    for r in picked:
        r['eval'] = True
# Facing sample: 30 per direction from Ninja, 20 right-facing DT2 heroes, 20 front + 20 side course.
fac = defaultdict(list)
for r in recs:
    if r['facing']:
        fac[(r['source'], r['facing'])].append(r)
for (src, f), rs in fac.items():
    random.shuffle(rs)
    n = 30 if src == 'ninja' else 20
    for r in rs[:n]:
        r['eval'] = True
with open(f'{OUT}/corpus.jsonl', 'w') as fh:
    for r in recs:
        a = r.pop('_a')
        from PIL import Image
        Image.fromarray(on_canvas(a)).save(f"{OUT}/corpus/{r['id']}.png")
        render(a).save(f"{OUT}/render/{r['id']}.png")
        r.setdefault('eval', False)
        fh.write(json.dumps(r) + '\n')
from collections import Counter
print('corpus', len(recs), Counter(r['source'] for r in recs))
print('categories', Counter(r['category'] for r in recs))
ev = [r for r in recs if r['eval']]
print('eval', len(ev), Counter(r['category'] for r in ev))
print('eval facing', Counter(r['facing'] for r in ev if r['facing']))
print('eval subject', sum(1 for r in ev if r['subject']), 'pose', Counter(r['pose'] for r in ev if r['pose']))
