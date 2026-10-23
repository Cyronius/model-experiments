# Scores samples-final.npz from the 25/50/100% runs: so400m recognition, copying (pixel nearest neighbour), diversity, vs real sprites; igpu-venv; 2026-10-19.
import glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sprite-data'))
os.environ.setdefault('SIGLIP', 'google/siglip2-so400m-patch16-256')
from spritekit import DATA, render, trim
import siglipkit

RUNS = sys.argv[1] if len(sys.argv) > 1 else f'{DATA}/runs'
PATTERN = sys.argv[2] if len(sys.argv) > 2 else 'scale-*'
CATS = ['creature', 'item', 'weapon', 'food', 'effect', 'terrain', 'structure', 'nature']
COPY_THRESH = 0.02      # mean |difference| per value, in [-1, 1]: near-identical
dev = siglipkit.dev
d = np.load(f'{DATA}/trainset.npz')
SP, CAT, GROUP = d['sprites'], d['cat'], d['group']


def subset(group, frac, seed=0):          # same rule as train.py
    order = np.random.default_rng(seed).permutation(group.max() + 1)
    counts = np.bincount(group, minlength=len(order))
    keep = order[:np.searchsorted(np.cumsum(counts[order]), frac * len(group)) + 1]
    return np.isin(group, keep)


def to_rgba(x):                           # (N,4,32,32) in [-1,1] -> uint8 RGBA, alpha thresholded
    a = ((np.clip(x.astype(np.float32), -1, 1) + 1) * 127.5).round().astype(np.uint8).transpose(0, 2, 3, 1)
    a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
    return a


def to_float(a):                          # uint8 RGBA (N,32,32,4) -> (N,4096) floats in [-1,1]
    return (a.astype(np.float32) / 127.5 - 1).transpose(0, 3, 1, 2).reshape(len(a), -1)


m, proc = siglipkit.model()
cats_all = list(siglipkit.CAT_PROMPTS)
T = np.stack([siglipkit.embed_text([f'pixel art sprite of {p}' for p in siglipkit.CAT_PROMPTS[c]]).mean(0) for c in cats_all])
T /= np.linalg.norm(T, axis=1, keepdims=True)


def recognise(arrs):                      # so400m zero-shot category per sprite; None for an empty sprite
    out = []
    for j in range(0, len(arrs), 32):
        batch = [a for a in arrs[j:j + 32]]
        ok = [i for i, a in enumerate(batch) if (a[..., 3] > 0).sum() >= 8 and trim(a) is not None]
        pred = [None] * len(batch)
        if ok:
            x = proc(images=[render(batch[i]) for i in ok], return_tensors='pt').to(dev)
            with torch.no_grad():
                e = torch.nn.functional.normalize(siglipkit._feat(m.get_image_features(pixel_values=x['pixel_values'].half())).float(), dim=-1).cpu().numpy()
            for i, z in zip(ok, (e @ T.T).argmax(1)):
                pred[i] = siglipkit.MERGE.get(cats_all[z], cats_all[z])
        out += pred
    return out


def nn_dist(samples, bank_idx):           # mean |diff| to the nearest training sprite, in chunks on the gpu
    q = torch.tensor(to_float(samples), device=dev, dtype=torch.float16)
    best = torch.full((len(q),), 1e9, device=dev)
    for i in range(0, len(bank_idx), 40000):
        b = torch.tensor(to_float(SP[bank_idx[i:i + 40000]]), device=dev, dtype=torch.float16)
        dist = (q.float().pow(2).sum(1)[:, None] + b.float().pow(2).sum(1)[None] - 2 * (q @ b.T).float()).clamp(min=0).sqrt()
        best = torch.minimum(best, dist.min(1).values)
    return (best / np.sqrt(q.shape[1])).cpu().numpy()   # rms difference per value


def diversity(a):                         # mean pairwise rms difference
    f = torch.tensor(to_float(a), device=dev)
    return float((torch.cdist(f, f).sum() / (len(f) * (len(f) - 1))) / np.sqrt(f.shape[1]))


rng = np.random.default_rng(0)
lines = ['# Scaling test scores', '',
         'recognised = so400m zero-shot agrees with the requested category (empty samples count as wrong);',
         f'copy = share of samples whose nearest training sprite is within {COPY_THRESH} rms; div = mean pairwise rms difference. "real" = the same measure on real sprites.', '']
real_rec, real_div = {}, {}
for c in CATS:
    ix = rng.choice(np.nonzero(CAT == CATS.index(c))[0], 256, replace=False)
    real_rec[c] = np.mean([p == c for p in recognise(SP[ix])]); real_div[c] = diversity(SP[ix])
lines += ['| run | data frac | ' + ' | '.join(CATS) + ' | mean |', '|---|---:|' + '---:|' * (len(CATS) + 1)]
lines.append('| real sprites | - | ' + ' | '.join(f'{100 * real_rec[c]:.0f}%' for c in CATS) + f' | {100 * np.mean(list(real_rec.values())):.0f}% |')
detail = ['', '## Per run: recognised / copy / diversity (real diversity in brackets)', '']
for npz in sorted(glob.glob(f'{RUNS}/{PATTERN}/samples-final.npz')):
    name = os.path.basename(os.path.dirname(npz)); frac = float(name.split('-f')[-1]) if '-f' in name else 1.0
    z = np.load(npz); x, cat = z['x'], z['cat']
    bank = np.nonzero(subset(GROUP, frac))[0] if frac < 1 else np.arange(len(SP))
    rec, rows = [], []
    for ci, c in enumerate(CATS):
        a = to_rgba(x[cat == ci])
        r = np.mean([p == c for p in recognise(a)]); nd = nn_dist(a, bank)
        ncol = float(np.median([len(np.unique(p[p[..., 3] > 0][:, :3], axis=0)) if (p[..., 3] > 0).any() else 0 for p in a]))
        rec.append(r); rows.append((c, r, float((nd < COPY_THRESH).mean()), float(np.median(nd)), diversity(a), ncol))
    lines.append(f'| {name} | {frac} | ' + ' | '.join(f'{100 * r:.0f}%' for r in rec) + f' | {100 * np.mean(rec):.0f}% |')
    detail.append(f'**{name}** (training sprites available: {len(bank)})')
    detail += [f'- {c}: recognised {100 * r:.0f}%, copy {100 * cp:.0f}%, median NN rms {md:.3f}, div {dv:.3f} (real {real_div[c]:.3f}), median colours {nc:.0f}' for c, r, cp, md, dv, nc in rows]
    detail.append('')
open(f'{RUNS}/report-{PATTERN.replace("*", "X")}.md', 'w').write('\n'.join(lines + detail) + '\n')
print('\n'.join(lines + detail))
