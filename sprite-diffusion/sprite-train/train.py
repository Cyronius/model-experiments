# Rectified-flow DiT for 32x32 RGBA sprites, klein-style single-stream blocks, tag conditioning (plan Phase 2); 2026-10-17.
import argparse, json, math, os, re, sys, time
from collections import Counter
import numpy as np, torch, torch.nn.functional as F
import dit
from augment import palette_swap
from sampling import SpriteSampler
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sprite-data'))
try:
    from dibsutil import gpu_wanted
except ImportError:   # not needed off this machine
    gpu_wanted = lambda: False
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='C:/dev/sprite-data/trainset.npz')
ap.add_argument('--out', required=True)
ap.add_argument('--device', default='cuda')
ap.add_argument('--sampling', choices=['uniform', 'category', 'category-subtype'], default='uniform')
ap.add_argument('--unlabelled-prob', type=float, default=0.1,
                help='unlabelled image share in balanced modes, before independent CFG dropout')
ap.add_argument('--sampling-seed', type=int, default=0)
ap.add_argument('--init-from', help='model and EMA weights for a NEW run; resets optimizer and step')
ap.add_argument('--hidden', type=int, default=384)
ap.add_argument('--depth', type=int, default=10)
ap.add_argument('--patch', type=int, default=2)
ap.add_argument('--batch', type=int, default=256)
ap.add_argument('--steps', type=int, default=100_000)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--frac', type=float, default=1.0, help='nested subset by group for the scaling test (0.25 within 0.5 within 1)')
ap.add_argument('--split-seed', type=int, default=0)
ap.add_argument('--ckpt-every', type=int, default=2000)
ap.add_argument('--sample-steps', type=int, default=16)
ap.add_argument('--cfg', type=float, default=3.0)
ap.add_argument('--slice-steps', type=int, default=0, help='stop after this many steps this run (exit 0); 0 = run to --steps')
ap.add_argument('--yield-gpu', action='store_true', help='checkpoint and exit 75 when another session waits on the gpu lock')
ap.add_argument('--text', action='store_true', help='also condition on the caption words in the training set')
ap.add_argument('--vocab-min', type=int, default=20)
ap.add_argument('--max-words', type=int, default=12)
ap.add_argument('--palette-prob', type=float, default=None,
                help='palette augmentation chance for weapon/food/structure/nature (default: 0.5; 0 with --text)')
ap.add_argument('--palette-hue', type=float, default=25.0, help='maximum palette hue rotation in degrees')
args = ap.parse_args()
if args.palette_prob is None:
    args.palette_prob = 0.0 if args.text else 0.5
if not 0 <= args.palette_prob <= 1 or not 0 <= args.palette_hue <= 180:
    ap.error('--palette-prob must be in [0,1] and --palette-hue in [0,180]')
if args.text and args.palette_prob:
    ap.error('palette swaps contradict caption colour words; use --palette-prob 0 with --text')
if not 0 <= args.unlabelled_prob <= 1:
    ap.error('--unlabelled-prob must be in [0,1]')
if args.init_from and args.text:
    ap.error('--init-from only supports category-only training')

dev = args.device
os.makedirs(args.out, exist_ok=True)
NCAT, NFP = dit.NCAT, dit.NFP
DiT = dit.DiT


def subset(group, frac, seed):
    order = np.random.default_rng(seed).permutation(group.max() + 1)
    counts = np.bincount(group, minlength=len(order))
    keep = order[:np.searchsorted(np.cumsum(counts[order]), frac * len(group)) + 1]
    return np.isin(group, keep)


@torch.no_grad()
def sample(model, cat, fp, txt, steps, cfg, gen):
    x = torch.randn(len(cat), 4, 32, 32, device=dev, generator=gen)
    null_c, null_f = torch.full_like(cat, NCAT), torch.full_like(fp, NFP)
    ts = torch.linspace(1, 0, steps + 1, device=dev)
    for i in range(steps):
        t = ts[i].expand(len(cat))
        with torch.autocast(dev, dtype=torch.bfloat16):
            v = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([cat, null_c]), torch.cat([fp, null_f]),
                      None if txt is None else torch.cat([txt, torch.zeros_like(txt)])).float()
        vc, vu = v.chunk(2)
        x = x + (ts[i + 1] - ts[i]) * (vu + cfg * (vc - vu))
    return x


def to_png(x, path, cols):
    a = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
    rows = math.ceil(len(a) / cols)
    sheet = Image.new('RGB', (cols * 66, rows * 66), (128, 128, 128))
    for i, s in enumerate(a):
        im = Image.fromarray(s).resize((64, 64), Image.NEAREST)
        sheet.paste(im, ((i % cols) * 66 + 1, (i // cols) * 66 + 1), im)
    sheet.save(path)


d = np.load(args.data)
keep = subset(d['group'], args.frac, args.split_seed) if args.frac < 1 else np.ones(len(d['group']), bool)
X = torch.from_numpy(d['sprites'][keep]).permute(0, 3, 1, 2).contiguous().to(dev)   # uint8 on the GPU: 4 KB a sprite
C = torch.from_numpy(d['cat'][keep].astype(np.int64)).to(dev)
FP = torch.from_numpy(d['fp'][keep].astype(np.int64)).to(dev)
C[C < 0] = NCAT
sampler = SpriteSampler(d['cat'][keep], mode=args.sampling,
                        subtypes=d['creature_subtype'][keep] if 'creature_subtype' in d else None,
                        unlabelled_prob=args.unlabelled_prob, device=dev, seed=args.sampling_seed)
sampling_report = dict(mode=args.sampling, seed=args.sampling_seed,
                       unlabelled_prob=args.unlabelled_prob, pools=sampler.report)
print('sampling: ' + json.dumps(sampling_report), flush=True)
T = None
if args.text:
    allcaps = [re.split(r'[^a-z0-9]+', c.lower()) for c in d['caption']]
    counts = Counter(w for ws in allcaps for w in set(ws) if w)
    vocab = {w: i + 1 for i, w in enumerate(sorted(w for w, n in counts.items() if n >= args.vocab_min))}
    json.dump(vocab, open(f'{args.out}/vocab.json', 'w'))
    ids = np.zeros((len(allcaps), args.max_words), np.int32)
    for i, ws in enumerate(allcaps):
        row = [vocab[w] for w in ws if w in vocab][:args.max_words]
        ids[i, :len(row)] = row
    T = torch.from_numpy(ids[keep].astype(np.int64)).to(dev)
    print(f'caption vocabulary {len(vocab)} words, {float((T > 0).any(1).float().mean()) * 100:.0f}% of sprites have text', flush=True)
print(f'{len(X)} sprites ({args.frac:.0%} of {len(keep)}), labelled {int((C < NCAT).sum())}', flush=True)
print(f'palette augmentation: probability {args.palette_prob:g}, hue +/-{args.palette_hue:g} degrees; weapon/food/structure/nature', flush=True)

V = len(vocab) + 1 if args.text else 0
model = DiT(args.hidden, args.depth, args.patch, V).to(dev)
ema = DiT(args.hidden, args.depth, args.patch, V).to(dev).eval().requires_grad_(False)
ema.load_state_dict(model.state_dict())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99), fused=dev == 'cuda' or None)
mp, ep = list(model.parameters()), list(ema.parameters())
print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, {(32 // args.patch) ** 2} tokens', flush=True)
step, ck = 0, f'{args.out}/ckpt.pt'


def save():
    torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, args=vars(args), sampling=sampling_report, sampler_state=sampler.get_state()), ck + '.tmp')
    os.replace(ck + '.tmp', ck)


if os.path.exists(ck):
    s = torch.load(ck, map_location=dev)
    if args.init_from and s.get('args', {}).get('init_from') != args.init_from:
        ap.error('output checkpoint belongs to a different initialization; choose a fresh output directory')
    if s.get('sampling', dict(mode='uniform')) != sampling_report:
        # Legacy checkpoints used uniform draws and did not record sampler state.
        if 'sampling' in s or args.sampling != 'uniform':
            ap.error('sampling changed on resume; use --init-from with a fresh output directory')
    if 'sampler_state' in s:
        sampler.set_state(s['sampler_state'])
    model.load_state_dict(s['model']); ema.load_state_dict(s['ema']); opt.load_state_dict(s['opt']); step = s['step']
    print(f'resumed at step {step}', flush=True)
elif args.init_from:
    s = torch.load(args.init_from, map_location=dev)
    if s.get('args', {}).get('text', False):
        ap.error('--init-from only supports category-only checkpoints')
    model.load_state_dict(s['model']); ema.load_state_dict(s['ema'])
    print(f'initialized weights from {args.init_from} (source step {s.get("step", "unknown")}); new optimizer, step 0', flush=True)
with open(f'{args.out}/sampling.json', 'w') as sampling_file:
    json.dump(sampling_report, sampling_file, indent=2)

def prompts(per_cat, seed):
    # Real captions of random training sprites per category (+ unlabelled), so the prompts are in-distribution.
    g = np.random.default_rng(seed); cats = C.cpu().numpy(); ix = []
    for c in range(NCAT + 1):
        pool = np.nonzero(cats == c)[0]
        if len(pool):
            ix += list(g.choice(pool, per_cat, replace=len(pool) < per_cat))
    ix = torch.tensor(ix, device=dev)
    return C[ix], torch.full_like(ix, NFP), (T[ix] if T is not None else None)


grid_cat, grid_fp, grid_txt = prompts(12, 0)
log = open(f'{args.out}/log.txt', 'a')
t0, acc = time.time(), torch.zeros((), device=dev)
start_step = step
while step < args.steps:
    idx = sampler.draw(args.batch)
    x = X[idx].float() / 127.5 - 1
    flip = torch.rand(args.batch, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    if args.palette_prob:
        x = palette_swap(x, C[idx], probability=args.palette_prob, hue_degrees=args.palette_hue)
    cat, fp = C[idx], FP[idx]
    txt = None if T is None else torch.where(torch.rand(args.batch, 1, device=dev) < 0.1, 0, T[idx])
    cat = torch.where(torch.rand(args.batch, device=dev) < 0.1, NCAT, cat)
    fp = torch.where(torch.rand(args.batch, device=dev) < 0.1, NFP, fp)
    t = torch.sigmoid(torch.randn(args.batch, device=dev))
    n = torch.randn_like(x)
    xt = (1 - t[:, None, None, None]) * x + t[:, None, None, None] * n
    for g in opt.param_groups:
        g['lr'] = args.lr * min(1.0, (step + 1) / 1000)
    with torch.autocast(dev, dtype=torch.bfloat16):
        loss = F.mse_loss(model(xt, t, cat, fp, txt).float(), n - x)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(mp, 1.0, foreach=True); opt.step()
    with torch.no_grad():
        torch._foreach_lerp_(ep, mp, 1 - 0.999)
    step += 1; acc += loss.detach()   # no .item() per step: a host sync stalls the kernel queue
    if step % 100 == 0:
        msg = f'step {step} loss {acc.item() / 100:.4f} {100 * args.batch / (time.time() - t0):.0f} samples/s'
        print(msg, flush=True); log.write(msg + '\n'); log.flush(); t0 = time.time(); acc.zero_()
    if args.yield_gpu and step % 25 == 0 and gpu_wanted():
        save(); print(f'yielding the gpu at step {step}', flush=True); sys.exit(75)
    if step % args.ckpt_every == 0 or step == args.steps:
        save()
        gen = torch.Generator(dev).manual_seed(0)
        to_png(sample(ema, grid_cat, grid_fp, grid_txt, args.sample_steps, args.cfg, gen), f'{args.out}/grid-{step:07d}.png', 12)
        t0 = time.time()
    if args.slice_steps and step - start_step >= args.slice_steps and step < args.steps:
        save(); print(f'slice done at step {step}', flush=True); sys.exit(0)
if step >= args.steps:
    # 256 EMA samples per category (+ unlabelled) for the scaling-test scoring.
    gen = torch.Generator(dev).manual_seed(1)
    cats, fps, txts = prompts(256, 1)
    out = torch.cat([sample(ema, cats[i:i + 256], fps[i:i + 256], None if txts is None else txts[i:i + 256], args.sample_steps, args.cfg, gen).cpu() for i in range(0, len(cats), 256)])
    np.savez_compressed(f'{args.out}/samples-final.npz', x=out.clamp(-1, 1).numpy().astype(np.float16), cat=np.where(cats.cpu().numpy() == NCAT, NCAT, cats.cpu().numpy()),
                        txt=(txts.cpu().numpy() if txts is not None else np.zeros(0)))
    print(f'wrote {len(out)} samples', flush=True)
if args.slice_steps or args.yield_gpu:
    sys.exit(3)
