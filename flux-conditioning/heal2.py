"""Heal Klein with removed block gaps: full fine-tune of the kept blocks + one bridge per gap, half on the student's own path (2026-11-19)."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from heal import BF, DEV, bridge, head, load_pipe, log, prep, rel, run_blocks, sample_prompts

GAPS = [(13, 22)]
KEEP = [k for k in range(25) if not any(a <= k <= b for a, b in GAPS)]
N_DBL = 5


def set_gaps(spec: str) -> None:
    global GAPS, KEEP
    GAPS = [tuple(int(x) for x in g.split("-")) for g in spec.split(",")]
    KEEP = [k for k in range(25) if not any(a <= k <= b for a, b in GAPS)]


def gkey(g) -> str:
    return f"{g[0]}-{g[1]}"


def block_call(blk, i: int, state, P: dict):
    if i < N_DBL:
        return blk(hidden_states=state[1], encoder_hidden_states=state[0], temb_mod_img=P["img"], temb_mod_txt=P["txt"],
                   image_rotary_emb=P["rope"], joint_attention_kwargs=None)
    return blk(hidden_states=state, encoder_hidden_states=None, temb_mod=P["sgl"], image_rotary_emb=P["rope"],
               joint_attention_kwargs=None)


class Student:
    def __init__(self, tr, train_set: set[int], W: dict[str, list[torch.Tensor]]):
        self.tr = tr
        allb = list(tr.transformer_blocks) + list(tr.single_transformer_blocks)
        self.blocks, self.params = {}, []
        for k in KEEP:
            if k in train_set:
                b = copy.deepcopy(allb[k])
                for p in b.parameters():
                    p.requires_grad_(True)
                    self.params.append(p)
                self.blocks[k] = b
            else:
                self.blocks[k] = allb[k]
        self.train_set = train_set
        self.W = {g: [nn.Parameter(w.float().to(DEV)) for w in ws] for g, ws in W.items()}
        self.ends = {b + 1: gkey((a, b)) for a, b in GAPS}

    def __call__(self, ctx, hs, t, ids, s: int, ckpt: bool = False):
        tr = self.tr
        P = prep(tr, t, ids, hs.shape[0])
        state = (ctx, tr.x_embedder(hs))
        h12 = None
        for k in KEEP:
            if k == N_DBL:
                state = torch.cat(state, dim=1)
            if k in self.ends:
                if h12 is None:
                    h12 = state
                state = bridge(tr, state, P, self.W[self.ends[k]][s])
            blk = self.blocks[k]
            if ckpt and k in self.train_set:
                if k < N_DBL:
                    state = checkpoint(lambda a, b, blk=blk, k=k: block_call(blk, k, (a, b), P), *state, use_reentrant=False)
                else:
                    state = checkpoint(lambda a, blk=blk, k=k: block_call(blk, k, a, P), state, use_reentrant=False)
            else:
                state = block_call(blk, k, state, P)
        return h12, state, head(tr, state, P, ctx.shape[1])


@torch.no_grad()
def teacher(tr, ctx, hs, t, ids):
    P = prep(tr, t, ids, hs.shape[0])
    h12 = run_blocks(tr, (ctx, tr.x_embedder(hs)), 0, GAPS[0][0], P)
    h24 = run_blocks(tr, h12, GAPS[0][0], 25, P)
    return h12, h24, head(tr, h24, P, ctx.shape[1])


class InBackwardAdam:
    """Adam with fp32 master weights and bf16 moments; each step runs as its gradient lands, so no gradient buffer stays."""

    def __init__(self, groups: list[tuple[list[nn.Parameter], float]], b1=0.9, b2=0.999, eps=1e-8):
        self.b1, self.b2, self.eps, self.t, self.scale = b1, b2, eps, 0, 1.0
        self.st = {}
        for params, lr in groups:
            for p in params:
                self.st[p] = {"lr": lr, "m": torch.zeros_like(p, dtype=BF), "v": torch.zeros_like(p, dtype=BF),
                              "w": None if p.dtype == torch.float32 else p.detach().float().clone()}
                p.register_post_accumulate_grad_hook(self.hook)

    def begin(self, scale: float) -> None:
        self.t += 1
        self.scale = scale

    @torch.no_grad()
    def hook(self, p) -> None:
        st = self.st[p]
        g = p.grad.float()
        p.grad = None
        m = st["m"].float().mul_(self.b1).add_(g, alpha=1 - self.b1)
        v = st["v"].float().mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
        st["m"].copy_(m)
        st["v"].copy_(v)
        w = p.data if st["w"] is None else st["w"]
        step = (m / (1 - self.b1 ** self.t)) / ((v / (1 - self.b2 ** self.t)).sqrt_().add_(self.eps))
        w.add_(step, alpha=-st["lr"] * self.scale)
        if st["w"] is not None:
            p.data.copy_(w)


def nmse(a, b):
    a, b = a.float(), b.float()
    return (a - b).pow(2).mean() / b.pow(2).mean()


def data(args) -> None:
    probe = json.loads(args.probe.read_text(encoding="utf-8"))[: args.n_probe]
    prompts = sample_prompts(args.n_train + args.n_val, {c["prompt"] for c in probe}, 0)
    log(f"prompts {len(prompts)}")
    pipe = load_pipe(True)
    tr = pipe.transformer
    meta = {"t": []}
    orig = tr.forward

    def spy(**kw):
        meta.setdefault("img", kw["img_ids"][0].cpu())
        meta.setdefault("txt", kw["txt_ids"][0].cpu())
        meta["t"].append(float(kw["timestep"][0]))
        return orig(**kw)

    tr.forward = spy
    with torch.no_grad():
        pipe(prompt=prompts[0], height=512, width=512, num_inference_steps=4, guidance_scale=1.0,
             generator=torch.Generator(DEV).manual_seed(0), output_type="latent")
        del tr.forward
        ctx = []
        for i in range(0, len(prompts), 16):
            emb = pipe.encode_prompt(prompts[i:i + 16], device=DEV)[0]
            ctx.append(tr.context_embedder(emb).to(BF).cpu())
            if i % 800 == 0:
                log(f"encoded {i}/{len(prompts)}")
        probe_emb = {c["id"]: pipe.encode_prompt(c["prompt"], device=DEV)[0][0].cpu() for c in probe}
    torch.save({"ctx": torch.cat(ctx), "meta": meta, "prompts": prompts, "probe_emb": probe_emb}, args.work / "data2.pt")
    log(f"data saved, timesteps {meta['t']}")


def load_init(student: Student, path: Path) -> None:
    """Merge a heal.py LoRA checkpoint into the student copies."""
    st = torch.load(path)
    with torch.no_grad():
        for key, a in st["lora"].items():
            if not key.endswith(".a"):
                continue
            b = st["lora"][key[:-2] + ".b"]
            part, idx, rest = key[:-2].split(".", 2)
            k = int(idx) + (N_DBL if part == "single_transformer_blocks" else 0)
            if k not in student.train_set:
                raise SystemExit(f"LoRA on block {k}, which is not trainable here")
            lin = student.blocks[k].get_submodule(rest)
            lin.weight.add_((b.float().to(DEV) @ a.float().to(DEV)).to(lin.weight.dtype))


def rollout(fn, ctx, x, ids, sig, upto: int):
    for k in range(upto):
        v = fn(ctx, x, torch.full((x.shape[0],), sig[k], device=DEV), k)
        x = (x.float() + (sig[k + 1] - sig[k]) * v.float()).to(BF)
    return x


def tv(tr, ids):
    return lambda c, x, t, k: teacher(tr, c, x, t, ids)[2]


def sv(student, ids):
    return lambda c, x, t, k: student(c, x, t, ids, k)[2]


@torch.no_grad()
def validate(tr, student, CTX, val, ids, sig, bsz) -> dict:
    tf, op, fin = [], [], []
    for j in range(0, len(val), bsz):
        pick = val[j:j + bsz]
        ctx = CTX[pick].to(DEV)
        g = torch.Generator().manual_seed(5000 + j)
        x0 = torch.randn(len(pick), 1024, 128, generator=g).to(DEV, BF)
        xt, xs = x0, x0
        for k in range(4):
            t = torch.full((len(pick),), sig[k], device=DEV)
            vt = teacher(tr, ctx, xt, t, ids)[2]
            tf.append(rel(student(ctx, xt, t, ids, k)[2], vt))
            vs = student(ctx, xs, t, ids, k)[2]
            op.append(rel(vs, teacher(tr, ctx, xs, t, ids)[2]))
            xt = (xt.float() + (sig[k + 1] - sig[k]) * vt.float()).to(BF)
            xs = (xs.float() + (sig[k + 1] - sig[k]) * vs.float()).to(BF)
        fin.append(rel(xs, xt))
    return {"tf_vel_err": round(float(np.mean(tf)), 4), "op_vel_err": round(float(np.mean(op)), 4),
            "final_latent_err": round(float(np.mean(fin)), 4)}


@torch.no_grad()
def render(pipe, student, probe_emb, ids, ts, out: Path, tag: str, seeds) -> float:
    from PIL import Image
    tr = pipe.transformer
    out.mkdir(exist_ok=True)

    def img(cid, seed):
        return pipe(prompt_embeds=probe_emb[cid][None].to(DEV), height=512, width=512, num_inference_steps=4,
                    guidance_scale=1.0, generator=torch.Generator(DEV).manual_seed(seed)).images[0]

    for cid in probe_emb:
        for seed in seeds:
            if not (out / f"{cid}-s{seed}-teacher.png").exists():
                img(cid, seed).save(out / f"{cid}-s{seed}-teacher.png")

    def fwd(**kw):
        t = kw["timestep"]
        k = int(np.argmin([abs(float(t[0]) - x) for x in ts]))
        return (student(tr.context_embedder(kw["encoder_hidden_states"]), kw["hidden_states"], t, ids, k)[2],)

    tr.forward = fwd
    mads = []
    try:
        for cid in probe_emb:
            for seed in seeds:
                p = out / f"{cid}-s{seed}-{tag}.png"
                img(cid, seed).save(p)
                a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
                b = np.asarray(Image.open(out / f"{cid}-s{seed}-teacher.png").convert("RGB"), dtype=np.float32)
                mads.append(float(np.abs(a - b).mean()))
    finally:
        del tr.forward
    return float(np.mean(mads))


def train(args) -> None:
    blob = torch.load(args.work / "data2.pt", weights_only=False)
    CTX, meta, probe_emb = blob["ctx"], blob["meta"], blob["probe_emb"]
    ids = {"img": meta["img"].to(DEV), "txt": meta["txt"].to(DEV)}
    ts = meta["t"][:4]
    sig = ts + [0.0]
    n = CTX.shape[0]
    train_idx, val = list(range(n - args.n_val)), list(range(n - args.n_val, n))
    pipe = load_pipe(False)
    tr = pipe.transformer
    train_set = set(KEEP) if args.train_blocks == "all" else {int(x) for x in args.train_blocks.split(",")}
    ridge = np.load(args.bridges) if args.bridges else None
    if args.init:
        W0 = {gkey(GAPS[0]): torch.load(args.init)["W"]}
    else:
        W0 = {gkey(g): [torch.from_numpy(ridge[f"{gkey(g)}_s{s}"]) for s in range(4)] for g in GAPS}
    student = Student(tr, train_set, W0)
    if args.init:
        load_init(student, args.init)
    Wall = [w for ws in student.W.values() for w in ws]
    opt = InBackwardAdam([(student.params, args.lr), (Wall, args.lr_bridge)])
    log(f"gaps {GAPS}, kept {len(KEEP)} blocks")
    log(f"trainable {sum(p.numel() for p in student.params) / 1e9:.2f}B in blocks {sorted(train_set)}")
    seeds = (1, 2) if not args.smoke else (1,)
    evals = []

    def evaluate(tag: str, it: int) -> None:
        r = validate(tr, student, CTX, val, ids, sig, args.batch) | {"tag": tag, "it": it}
        r["mad"] = round(render(pipe, student, probe_emb, ids, ts, args.work / "png", tag, seeds), 2)
        evals.append(r)
        (args.work / "evals.json").write_text(json.dumps(evals, indent=1))
        log(f"eval {r}")

    def save() -> None:
        torch.save({"blocks": {k: student.blocks[k].state_dict() for k in sorted(train_set)}, "gaps": GAPS,
                    "W": {g: [w.detach().cpu() for w in ws] for g, ws in student.W.items()}}, args.work / "ckpt.tmp")
        # Rename last so a fetch never copies a half-written file.
        (args.work / "ckpt.tmp").replace(args.work / "ckpt.pt")
        np.savez(args.work / "bridge-v2.npz", **{f"{g}_s{s}": ws[s].detach().cpu().numpy() for g, ws in student.W.items()
                                                 for s in range(4)},
                 **({k: ridge[k] for k in ridge.files if k.startswith("gate_")} if ridge is not None else {}))
        log("checkpoint saved")

    evaluate("e0", 0)
    rng = random.Random(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    end = args.until if args.until else t0 + args.minutes * 60
    next_eval, it, hist, ne = t0 + args.eval_minutes * 60, 0, [], 1
    while time.time() < end and (not args.smoke or it < 2):
        frac = (time.time() - t0) / (end - t0)
        s = rng.randrange(4)
        onp = it % 2 == 1
        pick = rng.sample(train_idx, args.batch)
        ctx = CTX[pick].to(DEV)
        x = torch.randn(args.batch, 1024, 128, generator=gen).to(DEV, BF)
        with torch.no_grad():
            x = rollout(sv(student, ids) if onp else tv(tr, ids), ctx, x, ids, sig, s)
        t = torch.full((args.batch,), sig[s], device=DEV)
        th12, th24, tvel = teacher(tr, ctx, x, t, ids)
        h12, h24, vel = student(ctx, x, t, ids, s, ckpt=True)
        loss = nmse(vel, tvel) + args.feat * (nmse(h12, th12) + nmse(h24, th24))
        if torch.isfinite(loss):
            opt.begin(min(1.0, (it + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac))))
            loss.backward()
            hist.append((onp, float(loss.detach())))
        else:
            log(f"skip non-finite loss at it {it}")
        it += 1
        if it % 100 == 0:
            a = [l for o, l in hist[-100:] if not o]
            b = [l for o, l in hist[-100:] if o]
            log(f"it {it} loss tf {np.mean(a):.4f} op {np.mean(b):.4f} ({time.time() - t0:.0f}s)")
        if time.time() > next_eval:
            evaluate(f"e{ne}", it)
            save()
            ne += 1
            next_eval = time.time() + args.eval_minutes * 60
    evaluate("final", it)
    save()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=("data", "train"))
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--probe", type=Path, help="json list of {id, prompt}")
    p.add_argument("--n-probe", type=int, default=10)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-val", type=int, default=40)
    p.add_argument("--init", type=Path, help="heal.py checkpoint (bridge + LoRA) to start from, first gap only")
    p.add_argument("--bridges", type=Path, help="ridge npz with {a}-{b}_s* per gap and gate_s*, used without --init")
    p.add_argument("--gaps", default="13-22", help="removed blocks, e.g. 13-15,17-21")
    p.add_argument("--train-blocks", default="all", help="'all' or comma list of block numbers")
    p.add_argument("--until", type=float, help="unix time to stop training")
    p.add_argument("--minutes", type=float, default=60)
    p.add_argument("--eval-minutes", type=float, default=120)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--lr-bridge", type=float, default=1e-5)
    p.add_argument("--feat", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="2 iterations, 1 render seed")
    args = p.parse_args()
    set_gaps(args.gaps)
    args.work.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    {"data": data, "train": train}[args.cmd](args)


if __name__ == "__main__":
    main()
