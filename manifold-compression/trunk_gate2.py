"""Gate 2 of the 9B-experts-on-a-small-trunk plan (.claude/plans/9b-experts-on-small-trunk.md): a
frozen-trunk pilot. Each of the 24 trunk layers gets the routed experts of one 9B layer (the S0_K24
carve: 96 experts of 128 neurons, top-24, its learned router), folded into the trunk width through a
ridge map from the trunk's MLP input to the 9B's MLP input (gate 1's fit). The experts read the same
normed input as the trunk's own MLP, which stays as the shared expert, and add to its output through
a per-channel output gate that starts at zero, so step 0 is exactly the stock trunk.

Trained: LoRA on the folded gate, up and down projections, the output gates and the routers. The
map back to the trunk width (B, the least-squares inverse of the input map) is folded into the down
projection and trains through its LoRA and the per-channel gate. Teacher: the dense 9B's stored
top-20 log-probs on its own graded answers (heal_shards.py pool); loss KL + 0.5 CE at answer
positions. Routers train on the experts' activation mass, as in the carve (Slim.select).

Control (arm "random"): the same model with random experts of the same shapes, row and column norms
and partition, and a random router of the same row norms, trained the same way for the same steps.

    trunk_gate2.py prep  --trunk Qwen3.5-2B             (CPU) layer pairing and the folding maps
    trunk_gate2.py train --trunk Qwen3.5-2B --arm fold --hours 1.5 --device cuda
    trunk_gate2.py train --trunk Qwen3.5-2B --arm random --steps-from fold --device cuda
    trunk_gate2.py check --trunk Qwen3.5-2B --arm fold --thinking off --device cuda
    trunk_gate2.py check --trunk Qwen3.5-0.8B --arm stock --thinking off --device cuda
    trunk_gate2.py bench --trunk Qwen3.5-0.8B --arm fold --device cuda
    trunk_gate2.py bench --trunk Qwen3.5-4B --arm stock --device cuda   (a size-up yardstick, stock only)

bench: the multiple-choice panel slim_bench.py prepared (results/moe-carve-v2-20261029/bench/rows.jsonl:
MMLU 20 per subject, ARC-Challenge, ARC-Easy, HellaSwag; zero-shot, raw text, slim_bench's prompts and
pick rules), read in the full tokenizer and scored with the model's own head. Writes each item's
choice log-probs and whether the pick is right, and which items share a 13-token window with this
pool's prompts or answers (slim_bench's audit was against the carve's healing data, not this pool).

Writes results/trunk-gate2-20261105/<trunk>/...; never overwrites a finished file.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'trunk-gate2-20261105'
GATE1 = ROOT.parent / 'trunk-gate1-20261105'
ACTS = GATE1 / 'acts'
HUB = Path.home() / '.cache/huggingface/hub'
BIG = 'Qwen3.5-9B'
TRUNK_PATHS = {
    'Qwen3.5-0.8B': GATE1 / 'models' / 'Qwen3.5-0.8B',
    'Qwen3.5-2B': HUB / 'models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc',
}
REFERENCE_PATHS = {'Qwen3.5-4B': next((HUB / 'models--Qwen--Qwen3.5-4B/snapshots').iterdir())}
BENCH = ROOT / 'results/moe-carve-v2-20261029/bench'
BIG_PATH = next((HUB / 'models--Qwen--Qwen3.5-9B/snapshots').iterdir())
CARVE = ROOT / 'results/moe-carve-v2-20261029/sweep'
POOL = ROOT / 'results/heal-shards-pod-20261105/mc-shard-ec420510e5b4/shards'
T, ROWS = 512, 64                 # gate 1's capture: 64 rows of 512, position 0 dropped
WINDOW = 3
CFG = {
    'lora_rank': 64, 'lora_alpha': 64, 'lr': 2e-4, 'gate_lr': 1e-3, 'router_lr': 1e-3,
    'betas': [0.9, 0.99], 'clip': 1.0, 'warmup_steps': 10, 'final_lr_share': 0.1,
    'seq_len': 4096, 'batch_sequences': 4, 'micro_batch': 1, 'ce_weight': 0.5,
    'router_weight': 1.0, 'router_stride': 4, 'loss_chunk': 1024,
    'held_answers': 160, 'pool_seed': 26110503, 'random_seed': 26110504, 'lora_seed': 26110505,
    'select': {'alpha': 1.0, 'low': 1 / 3, 'high': 2.0, 'seed': 26110502},
}
DEVICE = 'cpu'


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def save_json(path, obj, overwrite=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ValueError(f'{path} exists; refusing to overwrite')
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


# ---------------------------------------------------------------- prep: pairing and maps

def monotone_pairs(score, n_trunk, n_big):
    """For each trunk layer i a distinct 9B layer j(i), strictly increasing in i, maximizing the
    summed score[i, j] (-inf where not allowed)."""
    best = np.full((n_trunk + 1, n_big + 1), -np.inf)
    best[0, :] = 0
    take = np.zeros((n_trunk + 1, n_big + 1), bool)
    for i in range(1, n_trunk + 1):
        for j in range(1, n_big + 1):
            skip = best[i, j - 1]
            use = best[i - 1, j - 1] + score[i - 1, j - 1]
            best[i, j], take[i, j] = (use, True) if use > skip else (skip, False)
    pairs, i, j = [], n_trunk, n_big
    while i > 0:
        if take[i, j]:
            pairs.append((i - 1, j - 1))
            i -= 1
        j -= 1
    return pairs[::-1], float(best[n_trunk, n_big])


@torch.no_grad()
def stage_prep(args):
    torch.set_num_threads(args.threads)
    path = OUT / args.trunk / 'fold.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f'{path} exists')
    seq_of = torch.arange(ROWS).repeat_interleave(T - 1)
    te, tr = seq_of % 4 == 3, seq_of % 4 != 3
    X_all = np.load(ACTS / f'{args.trunk}_in.npy', mmap_mode='r')
    Y_all = np.load(ACTS / f'{BIG}_in.npy', mmap_mode='r')
    LT, LB = X_all.shape[0], Y_all.shape[0]
    solv = {}

    def solver(i):
        if i not in solv:
            X = torch.from_numpy(np.asarray(X_all[i])).float()
            mx = X[tr].mean(0)
            Xc = X[tr] - mx
            G = (Xc.T @ Xc).double()
            G += 1e-3 * G.diagonal().mean() * torch.eye(G.shape[0], dtype=G.dtype)
            solv[i] = (X, mx, Xc, torch.linalg.cholesky(G))
        return solv[i]

    r2 = np.full((LT, LB), -np.inf)
    maps = {}
    for j in range(LB):
        Y = torch.from_numpy(np.asarray(Y_all[j])).float()
        my = Y[tr].mean(0)
        Yc = Y[tr] - my
        d = Y[te] - Y[te].mean(0)
        centre = round(j * (LT - 1) / (LB - 1))
        for i in range(max(0, centre - WINDOW), min(LT, centre + WINDOW + 1)):
            X, mx, Xc, Lc = solver(i)
            W = torch.cholesky_solve((Xc.T @ Yc).double(), Lc).float()      # (d, 4096)
            pred = (X[te] - mx) @ W + my
            r2[i, j] = float(1 - (Y[te] - pred).pow(2).sum() / d.pow(2).sum())
            maps[(i, j)] = (W, my - mx @ W)
        log(f'{args.trunk} 9B layer {j}: best trunk layer {int(np.argmax(r2[:, j]))}, R2 {r2[:, j].max():.3f}')
        # Keep only maps that can still be chosen; memory.
    pairs, total = monotone_pairs(r2, LT, LB)
    A, b, B = [], [], []
    for i, j in pairs:
        W, bias = maps[(i, j)]
        a = W.T.contiguous()                                                  # (4096, d): y = a x + bias
        A.append(a)
        b.append(bias)
        B.append(torch.linalg.pinv(a.double()).float())                       # (d, 4096)
    torch.save({'pairs': pairs, 'A': torch.stack(A), 'b': torch.stack(b), 'B': torch.stack(B),
                'r2_input': torch.from_numpy(r2)}, path)
    report = {'trunk': args.trunk, 'pairs': pairs, 'R2_input_at_pairs': [round(float(r2[i, j]), 3) for i, j in pairs],
              'R2_input_mean': round(total / LT, 3), 'unused_9b_layers': sorted(set(range(LB)) - {j for _, j in pairs}),
              'note': 'Strictly increasing one-to-one pairing of the 24 trunk layers with 24 of the 32 9B layers '
                      '(dynamic programme over held-out R2 of the 9B MLP input from the trunk MLP input, within '
                      f'+-{WINDOW} trunk layers of proportional depth). Maps fit on gate 1 rows 0,1,2 of every 4.'}
    save_json(OUT / args.trunk / 'fold.json', report)
    log(json.dumps(report))


# ---------------------------------------------------------------- the model

class Experts(nn.Module):
    """The trunk MLP plus routed experts that read its input; output = mlp(x) + gate * experts(x)."""

    def __init__(self, mlp, wg, bg, wu, bu, wd, router_w, router_b, cluster, k, rank, alpha, gen):
        super().__init__()
        self.mlp = mlp
        dt = torch.bfloat16
        self.register_buffer('wg', wg.to(dt), persistent=False)
        self.register_buffer('bg', bg.to(dt), persistent=False)
        self.register_buffer('wu', wu.to(dt), persistent=False)
        self.register_buffer('bu', bu.to(dt), persistent=False)
        self.register_buffer('wd', wd.to(dt), persistent=False)
        self.register_buffer('wdnorm', wd.float().norm(dim=0), persistent=False)
        self.register_buffer('cluster', cluster.long(), persistent=False)
        self.k, self.scale = k, alpha / rank
        n, d = wg.shape
        self.n_experts = int(cluster.max()) + 1

        def down(shape):        # LoRA A: kaiming-uniform like peft, from a fixed generator
            bound = 1 / math.sqrt(shape[1])
            return nn.Parameter((torch.rand(shape, generator=gen) * 2 - 1) * bound)
        self.la_g, self.lb_g = down((rank, d)), nn.Parameter(torch.zeros(n, rank))
        self.la_u, self.lb_u = down((rank, d)), nn.Parameter(torch.zeros(n, rank))
        self.la_d, self.lb_d = down((rank, n)), nn.Parameter(torch.zeros(d, rank))
        self.router = nn.Linear(d, self.n_experts)
        self.router.weight.data.copy_(router_w)
        self.router.bias.data.copy_(router_b)
        self.gate = nn.Parameter(torch.zeros(d))
        self.on = True
        self.records = None       # a list: training passes append (router loss) here
        self.offset = 0

    def lora(self, x, la, lb):
        return F.linear(F.linear(x, la.to(x.dtype)), lb.to(x.dtype)) * self.scale

    def forward(self, x):
        base = self.mlp(x)
        if not self.on:
            return base
        logits = self.router(x.detach().float())     # the router loss trains the router only
        chosen = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, logits.detach().topk(self.k, -1).indices, True)
        keep = chosen[..., self.cluster]
        g = F.linear(x, self.wg, self.bg) + self.lora(x, self.la_g, self.lb_g)
        u = F.linear(x, self.wu, self.bu) + self.lora(x, self.la_u, self.lb_u)
        a = F.silu(g) * u
        am = a * keep
        out = F.linear(am, self.wd) + self.lora(am, self.la_d, self.lb_d)
        if self.records is not None and torch.is_grad_enabled():
            # Router target: each expert's share of the activation mass (|a| x down-column norm), as
            # in the carve. Runs again, identically, in the checkpoint recompute; those appends are dropped.
            flat = a.detach().flatten(0, 1)
            rows = torch.arange(self.offset % CFG['router_stride'], flat.shape[0], CFG['router_stride'], device=flat.device)
            score = flat[rows].float().abs() * self.wdnorm
            mass = torch.zeros(len(rows), self.n_experts, device=flat.device).index_add_(1, self.cluster, score)
            target = mass / mass.sum(-1, keepdim=True).clamp_min(1e-12)
            lp = logits.flatten(0, 1)[rows].log_softmax(-1)
            self.records.append(-(target * lp).sum(-1).mean())
        return base + self.gate.to(out.dtype) * out


def load_trunk(name):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    path = TRUNK_PATHS[name] if name in TRUNK_PATHS else REFERENCE_PATHS[name]
    config = AutoConfig.from_pretrained(path)
    config = getattr(config, 'text_config', config)
    model, info = AutoModelForCausalLM.from_pretrained(path, config=config, dtype=torch.bfloat16, output_loading_info=True)
    unexpected = [k for k in info['unexpected_keys'] if not k.startswith(('model.visual', 'visual', 'mtp', 'model.mtp'))]
    if info['missing_keys'] or info.get('mismatched_keys') or unexpected:
        raise RuntimeError(f'Invalid model load: {info}')
    tok = AutoTokenizer.from_pretrained(path)
    return model.to(DEVICE).eval().requires_grad_(False), tok


def big_mlp(j):
    from safetensors import safe_open
    idx = json.loads((BIG_PATH / 'model.safetensors.index.json').read_text())['weight_map']
    w = {}
    for part in ('gate_proj', 'up_proj', 'down_proj'):
        key = f'model.language_model.layers.{j}.mlp.{part}.weight'
        with safe_open(str(BIG_PATH / idx[key]), 'pt') as f:
            w[part] = f.get_tensor(key).float()
    return w


def like_rows(w, gen):
    """A random matrix with w's row norms."""
    r = torch.randn(w.shape, generator=gen)
    return r * (w.norm(dim=1, keepdim=True) / r.norm(dim=1, keepdim=True).clamp_min(1e-12))


def like_vec(v, gen):
    r = torch.randn(v.shape, generator=gen)
    return r * (v.norm() / r.norm().clamp_min(1e-12))


@torch.no_grad()
def install(model, trunk, arm):
    """Wraps every trunk MLP in Experts; returns them."""
    fold = torch.load(OUT / trunk / 'fold.pt', weights_only=True, map_location='cpu')
    s = torch.load(CARVE / 'set_S0_K24.pt', weights_only=True, map_location='cpu')
    routers = torch.load(CARVE / 'routers_S0_K24.pt', weights_only=True, map_location='cpu')
    assert s['S'] == 0 and s['experts'] == 96 and s['K'] == 24, s
    lgen = torch.Generator().manual_seed(CFG['lora_seed'])
    experts = []
    for n, (i, j) in enumerate(fold['pairs']):
        assert n == i
        A, b, B = fold['A'][n].to(DEVICE), fold['b'][n].to(DEVICE), fold['B'][n].to(DEVICE)
        w = {k: v.to(DEVICE) for k, v in big_mlp(j).items()}
        wg, bg = w['gate_proj'] @ A, w['gate_proj'] @ b
        wu, bu = w['up_proj'] @ A, w['up_proj'] @ b
        wd = B @ w['down_proj']
        rw = routers[j]['weight'].to(DEVICE).float()
        router_w, router_b = rw @ A, rw @ b + routers[j]['bias'].to(DEVICE).float()
        cluster = s['cluster'][j]
        if arm == 'random':
            g = torch.Generator().manual_seed(CFG['random_seed'] * 100 + n)
            wg, bg = like_rows(wg.cpu(), g), like_vec(bg.cpu(), g)
            wu, bu = like_rows(wu.cpu(), g), like_vec(bu.cpu(), g)
            wd = like_rows(wd.T.cpu(), g).T.contiguous()                    # column norms of the down
            router_w, router_b = like_rows(router_w.cpu(), g), router_b.cpu()[torch.randperm(len(router_b), generator=g)]
        layer = model.model.layers[i]
        e = Experts(layer.mlp, wg, bg, wu, bu, wd, router_w, router_b, cluster, s['K'], CFG['lora_rank'],
                    CFG['lora_alpha'], lgen).to(DEVICE)
        layer.mlp = e
        experts.append(e)
        del w
    return experts


def trainable(experts):
    lora = [p for e in experts for p in (e.la_g, e.lb_g, e.la_u, e.lb_u, e.la_d, e.lb_d)]
    gates = [e.gate for e in experts]
    routers = [p for e in experts for p in e.router.parameters()]
    return lora, gates, routers


def adapter_state(experts):
    return [{k: v.detach().cpu() for k, v in e.state_dict().items() if not k.startswith('mlp.')} for e in experts]


def load_adapter(experts, state):
    for e, sd in zip(experts, state, strict=True):
        missing, unexpected = e.load_state_dict({k: v for k, v in sd.items()}, strict=False)
        if unexpected or any(not k.startswith('mlp.') for k in missing):
            raise RuntimeError(f'adapter mismatch: {missing} {unexpected}')


# ---------------------------------------------------------------- data

class Answers:
    """The pool's kept answers (heal_sessions.select), shuffled once; the last held_answers are held out."""

    def __init__(self, eot):
        import heal_sessions as hs
        shards = sorted(p for p in POOL.iterdir() if (p / 'samples.jsonl').exists())
        self.items, self.teacher = [], {}
        for sh in shards:
            rows = [json.loads(l) for l in (sh / 'samples.jsonl').open()]
            for f in sorted(p for p in sh.glob('teacher*.npz') if not p.stem.endswith('.tmp')):
                z = np.load(f)
                self.teacher.update({(sh.name, k): z for k in z.files})
            for k in hs.select(rows, CFG['select']):
                self.items.append((sh.name, k, rows[k]))
        random.Random(CFG['pool_seed']).shuffle(self.items)
        self.held = self.items[-CFG['held_answers']:]
        self.train = self.items[:-CFG['held_answers']]
        self.eot = eot

    def arrays(self, item):
        sh, key, r = item
        ids = self.teacher[(sh, f'ids_{key}')][f'ids_{key}']
        lp = self.teacher[(sh, f'logprobs_{key}')][f'logprobs_{key}']
        return r['prompt_ids'], r['answer_ids'], ids, lp

    def pack(self, items, cursor, rows_wanted, length):
        """Packed sequences from items[cursor:], as heal_sessions.Pool.batch. Returns (arrays, cursor) or None."""
        ids = np.full((rows_wanted, length), self.eot, np.int64)
        mask = np.zeros((rows_wanted, length - 1), bool)
        tids = np.full((rows_wanted, length - 1, 20), -1, np.int32)
        tlp = np.full((rows_wanted, length - 1, 20), -np.inf, np.float32)
        k = cursor
        for b in range(rows_wanted):
            pos = 0
            while True:
                if k >= len(items):
                    return ((ids, mask, tids, tlp), k) if b > 0 or pos > 0 else None
                prompt, answer, a_ids, a_lp = self.arrays(items[k])
                item = list(prompt) + list(answer) + [self.eot]
                if len(item) > length:
                    k += 1
                    continue
                if pos + len(item) > length:
                    break
                ids[b, pos:pos + len(item)] = item
                first = pos + len(prompt) - 1
                n = len(answer)
                mask[b, first:first + n] = True
                tids[b, first:first + n] = a_ids
                tlp[b, first:first + n] = a_lp.astype(np.float32)
                pos += len(item)
                k += 1
        return (ids, mask, tids, tlp), k


def answer_loss(model, batch, index):
    """Summed KL (stored top-20) and CE over the answer positions of a packed batch, and their count."""
    from torch.utils.checkpoint import checkpoint
    import heal_sessions as hs
    ids, mask, tids, tlp = batch
    hs_ = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1]
    where = mask.nonzero(as_tuple=True)
    h, target, ti, tl = hs_[where], ids[:, 1:][where], tids[where], tlp[where]
    head = model.lm_head.weight
    kl = ce = 0
    for c in range(0, len(target), CFG['loss_chunk']):
        part = slice(c, c + CFG['loss_chunk'])
        if torch.is_grad_enabled():
            k, e = checkpoint(hs.stored_loss, h[part], target[part], ti[part], tl[part], head, index, use_reentrant=False)
        else:
            k, e = hs.stored_loss(h[part], target[part], ti[part], tl[part], head, index)
        kl, ce = kl + k, ce + e
    return kl, ce, len(target)


@torch.no_grad()
def held_out(model, data, index):
    model.eval()
    kl = ce = n = 0
    cursor = 0
    while True:
        got = data.pack(data.held, cursor, 1, CFG['seq_len'])
        if got is None:
            break
        arrays, cursor = got
        t = [torch.from_numpy(a).to(DEVICE) for a in arrays]
        k, e, m = answer_loss(model, t, index)
        kl, ce, n = kl + float(k), ce + float(e), n + m
        if cursor >= len(data.held):
            break
    return {'kl': kl / n, 'ce': ce / n, 'answer_tokens': n}


# ---------------------------------------------------------------- train

def stage_train(args):
    folder = OUT / args.trunk / args.arm
    if (folder / 'state.pt').exists():
        raise ValueError(f'{folder}/state.pt exists')
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(CFG['lora_seed'])
    model, tok = load_trunk(args.trunk)
    eot = tok.convert_tokens_to_ids('<|endoftext|>')
    data = Answers(eot)
    log(f'{len(data.items)} kept answers: {len(data.train)} train, {len(data.held)} held out')
    index = torch.arange(model.config.vocab_size, device=DEVICE)

    # The stock trunk's held-out numbers, and the step-0 identity check, before installing.
    probe = torch.from_numpy(data.pack(data.held, 0, 1, 512)[0][0]).to(DEVICE)
    with torch.no_grad():
        stock_logits = model(input_ids=probe).logits.float()
    stock_path = OUT / args.trunk / 'stock_held_out.json'
    if not stock_path.exists():
        save_json(stock_path, held_out(model, data, index))
    stock = json.loads(stock_path.read_text())
    log(f'stock held-out: {stock}')
    experts = install(model, args.trunk, args.arm)
    with torch.no_grad():
        diff = float((model(input_ids=probe).logits.float() - stock_logits).abs().max())
    log(f'step 0 against the stock trunk: max logit difference {diff}')
    if diff != 0:
        raise RuntimeError('step 0 is not the stock trunk')
    del stock_logits

    lora, gates, routers = trainable(experts)
    for p in lora + gates + routers:
        p.requires_grad_(True)
    opt = torch.optim.AdamW([{'params': lora, 'lr': CFG['lr'], 'base': CFG['lr']},
                             {'params': gates, 'lr': CFG['gate_lr'], 'base': CFG['gate_lr']}],
                            betas=tuple(CFG['betas']), weight_decay=0.0)
    ropt = torch.optim.Adam(routers, lr=CFG['router_lr'])
    params = lora + gates
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False

    if args.steps_from:
        total = json.loads((OUT / args.trunk / args.steps_from / 'train.json').read_text())['steps']
        until = math.inf
    else:
        total, until = None, time.time() + 3600 * args.hours
    log(f'trainable: LoRA {sum(p.numel() for p in lora) / 1e6:.1f}M, gates {sum(p.numel() for p in gates)}, '
        f'routers {sum(p.numel() for p in routers) / 1e6:.2f}M; steps {total or "by time"}')
    step, cursor, losses, t_start, times = 0, 0, [], time.time(), []
    while True:
        if total is not None and step >= total:
            break
        if total is None and len(times) >= 3:
            # Fix the step count from the measured pace, so the schedule can decay and the control can match it.
            pace = float(np.mean(times[1:]))
            total = step + max(1, int((until - time.time()) / pace))
            log(f'pace {pace:.1f} s/step: {total} steps in all')
        frac = step / total if total else 0.0
        f = min(1.0, (step + 1) / CFG['warmup_steps']) * (1 - (1 - CFG['final_lr_share']) * frac)
        for g in opt.param_groups:
            g['lr'] = g['base'] * f
        for g in ropt.param_groups:
            g['lr'] = CFG['router_lr'] * f
        got = data.pack(data.train, cursor, CFG['batch_sequences'], CFG['seq_len'])
        if got is None:
            log('pool exhausted')
            break
        arrays, cursor = got
        t0 = time.time()
        tensors = [torch.from_numpy(a).to(DEVICE) for a in arrays]
        positions = int(tensors[1].sum())
        model.train()
        tot = [0.0, 0.0, 0, []]
        for j in range(0, CFG['batch_sequences'], CFG['micro_batch']):
            part = [x[j:j + CFG['micro_batch']] for x in tensors]
            for e in experts:
                e.records, e.offset = [], step + j
            kl, ce, n = answer_loss(model, part, index)
            rl = torch.stack([r for e in experts for r in e.records[:1]]).mean()
            n_micro = CFG['batch_sequences'] // CFG['micro_batch']
            loss = (kl + CFG['ce_weight'] * ce) / max(1, positions) + CFG['router_weight'] * rl / n_micro
            for e in experts:
                e.records = []      # checkpoint recomputes append here during backward; dropped
            loss.backward()
            tot[0] += float(kl)
            tot[1] += float(ce)
            tot[2] += n
            tot[3].append(float(rl))
        for e in experts:
            e.records = None
        norm = float(torch.nn.utils.clip_grad_norm_(params, CFG['clip']))
        torch.nn.utils.clip_grad_norm_(routers, CFG['clip'])
        opt.step()
        ropt.step()
        opt.zero_grad(set_to_none=True)
        ropt.zero_grad(set_to_none=True)
        step += 1
        times.append(time.time() - t0)
        n = max(1, tot[2])
        gate_abs = float(torch.stack([e.gate.detach().abs().mean() for e in experts]).mean())
        losses.append({'step': step, 'positions': tot[2], 'kl': tot[0] / n, 'ce': tot[1] / n,
                       'router': float(np.mean(tot[3])), 'grad_norm': norm, 'lr_factor': f,
                       'gate_abs_mean': gate_abs, 'seconds': times[-1], 'cursor': cursor})
        if step % 5 == 0 or step <= 3:
            last = losses[-1]
            log(f"step {step}/{total}: KL {last['kl']:.4f}, CE {last['ce']:.4f}, router {last['router']:.3f}, "
                f"norm {norm:.2f}, |gate| {gate_abs:.4f}, {CFG['batch_sequences'] * CFG['seq_len'] / last['seconds']:.0f} tok/s")
        if not math.isfinite(losses[-1]['kl']):
            raise FloatingPointError('non-finite loss')
    model.gradient_checkpointing_disable()
    for p in lora + gates + routers:
        p.requires_grad_(False)
    torch.save({'trunk': args.trunk, 'arm': args.arm, 'step': step, 'adapter': adapter_state(experts)}, folder / 'state.pt')
    held = held_out(model, data, index)
    record = {'trunk': args.trunk, 'arm': args.arm, 'steps': step, 'cursor_answers': cursor,
              'train_answer_tokens': sum(l['positions'] for l in losses),
              'packed_tokens': step * CFG['batch_sequences'] * CFG['seq_len'],
              'hours': (time.time() - t_start) / 3600, 'cfg': CFG, 'held_out_stock': stock, 'held_out': held,
              'losses': losses}
    save_json(folder / 'train.json', record)
    log(f"done: {step} steps, held-out KL {stock['kl']:.4f} -> {held['kl']:.4f}, CE {stock['ce']:.4f} -> {held['ce']:.4f}")


# ---------------------------------------------------------------- held-out KL again (CPU re-check)

def stage_heldout(args):
    """The held-out KL and CE of one arm, recomputed (written 2026-11-07 for a CPU re-check: the local
    GPU has computed wrong values silently since 2026-11-05, so the training run's own numbers need a
    second source). Writes <arm>/held_out_<device>.json."""
    torch.set_num_threads(args.threads)
    out = OUT / args.trunk / args.arm / f'held_out_{DEVICE.split(":")[0]}.json'
    if out.exists():
        raise ValueError(f'{out} exists')
    model, tok = load_trunk(args.trunk)
    if args.arm != 'stock':
        experts = install(model, args.trunk, args.arm)
        state = torch.load(OUT / args.trunk / args.arm / 'state.pt', weights_only=False, map_location='cpu')
        load_adapter(experts, state['adapter'])
        for e in experts:
            e.to(DEVICE)
    data = Answers(tok.convert_tokens_to_ids('<|endoftext|>'))
    t0 = time.time()
    held = held_out(model, data, torch.arange(model.config.vocab_size, device=DEVICE))
    held.update(trunk=args.trunk, arm=args.arm, device=DEVICE, dtype='bfloat16', seconds=time.time() - t0)
    save_json(out, held)
    log(f"{args.trunk} {args.arm} held-out on {DEVICE}: KL {held['kl']:.4f}, CE {held['ce']:.4f}, {held['seconds'] / 60:.0f} min")


# ---------------------------------------------------------------- check (GSM8K 96, as dense_check.py)

def stage_check(args):
    import heal_probe as hp
    import moe_carve as mc
    import slim_batch as sb
    import slim_opd as so
    mc.DEVICE = DEVICE
    mc.configure(sb.CARVE_RULES)
    rules = json.loads((ROOT / 'geometry_check_rules.json').read_text())
    cfg = rules['eval']
    thinking = {'on': True, 'off': False}[args.thinking]
    out = OUT / args.trunk / args.arm / f'check_thinking_{args.thinking}.json'
    if out.exists():
        raise ValueError(f'{out} exists')
    model, tok = load_trunk(args.trunk)
    if args.arm != 'stock':
        experts = install(model, args.trunk, args.arm)
        state = torch.load(OUT / args.trunk / args.arm / 'state.pt', weights_only=False, map_location='cpu')
        load_adapter(experts, state['adapter'])
        for e in experts:
            e.to(DEVICE)
    model.eval()
    prompts = json.loads((ROOT / cfg['prompts']).read_text())['prompts'][:cfg['count']]
    ids = [tok.encode(so.template(tok, p['question'], thinking), add_special_tokens=False) for p in prompts]
    vocab = type('Vocab', (), {})()
    vocab.stop = [tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|endoftext|>')]
    vocab.removed = torch.zeros(model.config.vocab_size, device=DEVICE)
    Decode, _ = so.stopping_classes()
    x, mask, _ = so.left_pad(ids, vocab.stop[1])
    seed = cfg['seed'] + 1
    torch.manual_seed(seed)
    processor = Decode(vocab.removed, x.shape[1], torch.tensor([False] * len(ids), device=DEVICE), cfg['sampling'])
    t0 = time.time()
    with torch.no_grad(), sb.StoredState(model, [cfg['state']] * len(ids), seed) as stored:
        gen = so.compact(model, vocab, x, mask, processor, stored, cfg['max_new_tokens'], False)
    seconds = time.time() - t0
    rows = []
    for g, p in zip(gen, prompts, strict=True):
        answer = sb.final_number(tok.decode(g['tokens']))
        rows.append({'id': p['id'], 'tokens': g['tokens'], 'length': len(g['tokens']), 'finished': g['stop'] is not None,
                     'answer': answer, 'correct': answer is not None and abs(answer - p['answer']) < 1e-6})
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / f'gen_thinking_{args.thinking}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    summary = hp.generations(rows, rules['loop'])
    at = so.truncated(rows, [dict(p) for p in prompts], tok, cfg['also_at'])
    summary_at = {k: v for k, v in hp.generations(at, rules['loop']).items() if k != 'looping'}
    record = {'trunk': args.trunk, 'arm': args.arm, 'thinking': args.thinking, 'decoding': 'sampled',
              'sampling': cfg['sampling'], 'max_new_tokens': cfg['max_new_tokens'], 'seed': seed,
              'summary': summary, f'at_{cfg["also_at"]}': summary_at, 'seconds': seconds}
    save_json(out, record)
    log(f"{args.trunk} {args.arm} thinking {args.thinking}: {summary['correct']}/{len(rows)} right, "
        f"{summary['finished']} finished, {summary['loops']} loops, {seconds / 60:.0f} min")


def pool_windows(n):
    """The sorted, distinct hashes of every n-token window of the pool's prompts and answers (all shards, kept or not)."""
    import slim_heal as sh
    hashes = []
    for shard in sorted(p for p in POOL.iterdir() if (p / 'samples.jsonl').exists()):
        for line in (shard / 'samples.jsonl').open():
            r = json.loads(line)
            for ids in (r['prompt_ids'], r['answer_ids']):
                if len(ids) >= n:
                    hashes.append(sh.window_hashes(ids, n))
    return np.unique(np.concatenate(hashes))


def stage_bench(args):
    import moe_carve as mc
    import slim_bench as bench
    import slim_heal as sh
    torch.set_num_threads(args.threads)
    if args.trunk in REFERENCE_PATHS and args.arm != 'stock':
        raise ValueError(f'{args.trunk} is a yardstick; only its stock arm is scored')
    out = OUT / args.trunk / args.arm / 'bench.json'
    if out.exists():
        raise ValueError(f'{out} exists')
    mc.DEVICE = DEVICE
    rules = json.loads(bench.RULES.read_text())
    rows = [json.loads(l) for l in (BENCH / 'rows.jsonl').open()]
    n = rules['overlap_tokens']
    ref = pool_windows(n)
    shared = []
    for r in rows:
        h = sh.window_hashes(r['full']['context'][1:] + r['full']['choices'][r['answer']], n)
        at = np.searchsorted(ref, h).clip(max=len(ref) - 1)
        shared.append(bool(len(h)) and bool((ref[at] == h).any()))
    log(f'{len(rows)} rows, {sum(shared)} share a {n}-token window with the pool')
    model, tok = load_trunk(args.trunk)
    if args.arm != 'stock':
        experts = install(model, args.trunk, args.arm)
        state = torch.load(OUT / args.trunk / args.arm / 'state.pt', weights_only=False, map_location='cpu')
        load_adapter(experts, state['adapter'])
        for e in experts:
            e.to(DEVICE)
    model.eval()
    head = model.lm_head.weight.float()
    index = torch.arange(head.shape[0], device=DEVICE)
    pad = tok.convert_tokens_to_ids('<|endoftext|>')
    t0 = time.time()
    lps = bench.choice_logprobs(model, rows, 'full', head, index, pad, args.batch_tokens)
    seconds = time.time() - t0
    rule = {s['name']: s['rule'] for s in rules['sets']}
    items = [{'id': r['id'], 'set': r['set'], 'group': r['group'], 'shared': s, 'logprobs': lp,
              'right': bench.picks(r, lp, rule[r['set']]) == r['answer']} for r, lp, s in zip(rows, lps, shared)]
    sets = {}
    for name in rule:
        mine = [i for i in items if i['set'] == name]
        clean = [i for i in mine if not i['shared']]
        sets[name] = {'items': len(mine), 'accuracy': float(np.mean([i['right'] for i in mine])),
                      'clean_items': len(clean), 'clean_accuracy': float(np.mean([i['right'] for i in clean])) if clean else None}
    save_json(out, {'trunk': args.trunk, 'arm': args.arm, 'device': DEVICE, 'rows': str(BENCH / 'rows.jsonl'), 'overlap_tokens': n,
                    'sets': sets, 'seconds': seconds, 'items': items})
    log(f"{args.trunk} {args.arm} bench: " + ', '.join(f"{k} {v['accuracy']:.3f}" for k, v in sets.items())
        + f', {seconds / 60:.0f} min')


def main():
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prep', 'train', 'check', 'bench', 'heldout'])
    ap.add_argument('--trunk', required=True, choices=list(TRUNK_PATHS) + list(REFERENCE_PATHS))
    ap.add_argument('--arm', default='fold', choices=['fold', 'random', 'stock'])
    ap.add_argument('--hours', type=float, default=1.5)
    ap.add_argument('--steps-from', help='train: take the step count of this arm\'s finished run')
    ap.add_argument('--thinking', choices=['on', 'off'], default='off')
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-tokens', type=int, default=16384, help='bench: padded tokens per forward pass')
    args = ap.parse_args()
    DEVICE = args.device
    {'prep': stage_prep, 'train': stage_train, 'check': stage_check, 'bench': stage_bench, 'heldout': stage_heldout}[args.stage](args)


if __name__ == '__main__':
    main()
