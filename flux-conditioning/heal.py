"""Heal Klein with blocks 13-22 removed: start from the ridge bridge, train bridge + LoRA on blocks 12 and 23, local vs global loss (2026-11-19)."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

GAP = (13, 22)
LORA_BLOCKS = (12, 23)
REPO = "black-forest-labs/FLUX.2-klein-4B"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BF = torch.bfloat16


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class LoRA(nn.Module):
    on = True

    def __init__(self, base: nn.Linear, r: int):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.randn(r, base.in_features, device=DEV) / math.sqrt(base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, r, device=DEV))

    def forward(self, x):
        y = self.base(x)
        if not LoRA.on:
            return y
        return y + ((x.float() @ self.a.T) @ self.b.T).to(y.dtype)


def add_lora(tr, r: int) -> list[nn.Parameter]:
    params = []
    blocks = list(tr.transformer_blocks) + list(tr.single_transformer_blocks)
    for k in LORA_BLOCKS:
        for name, mod in list(blocks[k].named_modules()):
            if isinstance(mod, nn.Linear):
                parent = blocks[k].get_submodule(name.rsplit(".", 1)[0]) if "." in name else blocks[k]
                lo = LoRA(mod, r)
                setattr(parent, name.rsplit(".", 1)[-1], lo)
                params += [lo.a, lo.b]
    return params


def load_pipe(text_encoder: bool):
    from diffusers import Flux2KleinPipeline
    kw = {} if text_encoder else {"text_encoder": None, "tokenizer": None}
    pipe = Flux2KleinPipeline.from_pretrained(REPO, revision=REVISION, torch_dtype=BF, **kw).to(DEV)
    pipe.set_progress_bar_config(disable=True)
    for p in pipe.transformer.parameters():
        p.requires_grad_(False)
    return pipe


def prep(tr, t: torch.Tensor, ids: dict, batch: int) -> dict:
    temb = tr.time_guidance_embed(t.to(BF).expand(batch) * 1000, None)
    img_rope, txt_rope = tr.pos_embed(ids["img"]), tr.pos_embed(ids["txt"])
    return {"temb": temb, "img": tr.double_stream_modulation_img(temb), "txt": tr.double_stream_modulation_txt(temb),
            "sgl": tr.single_stream_modulation(temb),
            "rope": (torch.cat([txt_rope[0], img_rope[0]]), torch.cat([txt_rope[1], img_rope[1]]))}


def run_blocks(tr, state, lo: int, hi: int, P: dict):
    n_dbl = len(tr.transformer_blocks)
    blocks = list(tr.transformer_blocks) + list(tr.single_transformer_blocks)
    for i in range(lo, hi):
        if i == n_dbl and isinstance(state, tuple):
            state = torch.cat(state, dim=1)
        if i < n_dbl:
            state = blocks[i](hidden_states=state[1], encoder_hidden_states=state[0], temb_mod_img=P["img"],
                              temb_mod_txt=P["txt"], image_rotary_emb=P["rope"], joint_attention_kwargs=None)
        else:
            state = blocks[i](hidden_states=state, encoder_hidden_states=None, temb_mod=P["sgl"],
                              image_rotary_emb=P["rope"], joint_attention_kwargs=None)
    return state


def bridge(tr, x, P: dict, W: torch.Tensor):
    from diffusers.models.transformers.transformer_flux2 import Flux2Modulation
    shift, scale, _ = Flux2Modulation.split(P["sgl"], 1)[0]
    zn = (1 + scale) * tr.single_transformer_blocks[0].norm(x) + shift
    return x + (zn.float() @ W).to(x.dtype)


def head(tr, state, P: dict, ntxt: int):
    return tr.proj_out(tr.norm_out(state[:, ntxt:], P["temb"]))


def student(tr, ctx, hs, t, ids, W, upto_velocity=True):
    """Returns (state after block 23, velocity or None)."""
    P = prep(tr, t, ids, hs.shape[0])
    with torch.no_grad():
        s = run_blocks(tr, (ctx, tr.x_embedder(hs)), 0, 12, P)
    s = run_blocks(tr, s, 12, 13, P)
    s = bridge(tr, s, P, W)
    s = run_blocks(tr, s, GAP[1] + 1, 24, P)
    v = head(tr, run_blocks(tr, s, 24, 25, P), P, ctx.shape[1]) if upto_velocity else None
    return s, v


@torch.no_grad()
def teacher_state(tr, ctx, hs, t, ids):
    LoRA.on = False
    try:
        P = prep(tr, t, ids, hs.shape[0])
        return run_blocks(tr, (ctx, tr.x_embedder(hs)), 0, 24, P)
    finally:
        LoRA.on = True


def sample_prompts(n: int, exclude: set[str], seed: int) -> list[str]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    f = hf_hub_download("poloclub/diffusiondb", "metadata.parquet", repo_type="dataset")
    cols = [c for c in ("prompt", "prompt_nsfw") if c in pq.read_schema(f).names]
    tab = pq.read_table(f, columns=cols).to_pandas()
    if "prompt_nsfw" in tab:
        tab = tab[tab["prompt_nsfw"] < 0.3]
    ps = sorted({p.strip() for p in tab["prompt"].dropna() if 20 <= len(p.strip()) <= 400} - exclude)
    random.Random(seed).shuffle(ps)
    return ps[:n]


def data(args) -> None:
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    prompts = sample_prompts(args.n_train + args.n_val, {c["prompt"] for c in probe}, 0)
    log(f"prompts {len(prompts)}")
    pipe = load_pipe(True)
    tr = pipe.transformer
    rec, meta = [], {}
    orig = tr.forward

    def spy(**kw):
        out = orig(**kw)
        if not rec[-1]["hs"]:
            rec[-1]["ctx"] = tr.context_embedder(kw["encoder_hidden_states"])[0].cpu()
            meta.setdefault("img", kw["img_ids"][0].cpu())
            meta.setdefault("txt", kw["txt_ids"][0].cpu())
        rec[-1]["hs"].append(kw["hidden_states"][0].cpu())
        rec[-1]["v"].append(out[0][0].cpu())
        ts = meta.setdefault("t", [])
        if len(ts) < 4:
            ts.append(float(kw["timestep"][0]))
        return out

    tr.forward = spy
    with torch.no_grad():
        for i, p in enumerate(prompts):
            rec.append({"hs": [], "v": []})
            pipe(prompt=p, height=512, width=512, num_inference_steps=4, guidance_scale=1.0,
                 generator=torch.Generator(DEV).manual_seed(1000 + i), output_type="latent")
            if i % 100 == 0:
                log(f"teacher {i}/{len(prompts)}")
        probe_emb = {c["id"]: pipe.encode_prompt(c["prompt"], device=DEV)[0][0].cpu() for c in probe}
    del tr.forward
    torch.save({"rec": rec, "meta": meta, "prompts": prompts, "probe_emb": probe_emb}, args.work / "data.pt")
    log(f"data saved, timesteps {meta['t']}")


def batches(rec, idx, bsz, rng):
    """Yield (step, ctx, hs, v) with one step per batch, so every sample shares one timestep."""
    while True:
        s = rng.randrange(4)
        pick = rng.sample(idx, bsz)
        yield (s, torch.stack([rec[i]["ctx"] for i in pick]).to(DEV), torch.stack([rec[i]["hs"][s] for i in pick]).to(DEV),
               torch.stack([rec[i]["v"][s] for i in pick]).to(DEV))


def rel(a, b) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm())


@torch.no_grad()
def validate(tr, rec, val, ids, ts, W) -> dict:
    ev, es = [], []
    for i in val:
        for s in range(4):
            ctx, hs, v = rec[i]["ctx"][None].to(DEV), rec[i]["hs"][s][None].to(DEV), rec[i]["v"][s][None].to(DEV)
            t = torch.tensor([ts[s]], device=DEV)
            x, vs = student(tr, ctx, hs, t, ids, W[s])
            ev.append(rel(vs, v))
            es.append(rel(x, teacher_state(tr, ctx, hs, t, ids)))
    return {"velocity_rel_err": round(float(np.mean(ev)), 4), "state24_rel_err": round(float(np.mean(es)), 4)}


def train(args) -> None:
    blob = torch.load(args.work / "data.pt", weights_only=False)
    rec, meta = blob["rec"], blob["meta"]
    ids = {"img": meta["img"].to(DEV), "txt": meta["txt"].to(DEV)}
    ts = meta["t"]
    n = len(rec)
    train_idx, val = list(range(n - args.n_val)), list(range(n - args.n_val, n))
    ridge = np.load(args.bridges)
    pipe = load_pipe(False)
    tr = pipe.transformer
    lora = add_lora(tr, args.rank)
    W = [nn.Parameter(torch.from_numpy(ridge[f"{GAP[0]}-{GAP[1]}_s{s}"]).float().to(DEV) * (args.arm != "global0")) for s in range(4)]
    if args.init:
        st = torch.load(args.init)
        sd = tr.state_dict(keep_vars=True)
        with torch.no_grad():
            for w, v in zip(W, st["W"]):
                w.copy_(v.to(DEV))
            for k, v in st["lora"].items():
                sd[k].copy_(v.to(DEV))
        log(f"init from {args.init}: val {validate(tr, rec, val, ids, ts, W)}")
    if args.arm == "ridge":
        r = validate(tr, rec, val, ids, ts, W)
        log(f"ridge val {r}")
        (args.work / "val-ridge.json").write_text(json.dumps(r))
        return
    opt = torch.optim.AdamW([{"params": W, "lr": args.lr_bridge}, {"params": lora, "lr": args.lr_lora}], weight_decay=0.0)
    rng = random.Random(args.seed)
    gen = batches(rec, train_idx, args.batch, rng)
    t0, it, hist = time.time(), 0, []
    budget = args.minutes * 60
    while time.time() - t0 < budget:
        frac = (time.time() - t0) / budget
        for g, base in zip(opt.param_groups, (args.lr_bridge, args.lr_lora)):
            g["lr"] = base * min(1.0, it / 50) * 0.5 * (1 + math.cos(math.pi * frac))
        s, ctx, hs, v = next(gen)
        t = torch.full((args.batch,), ts[s], device=DEV)
        if args.arm == "local":
            target = teacher_state(tr, ctx, hs, t, ids)
            x, _ = student(tr, ctx, hs, t, ids, W[s], upto_velocity=False)
            loss = (x.float() - target.float()).pow(2).mean() / target.float().pow(2).mean()
        else:
            _, vs = student(tr, ctx, hs, t, ids, W[s])
            loss = (vs.float() - v.float()).pow(2).mean() / v.float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hist.append(float(loss))
        it += 1
        if it % 100 == 0:
            log(f"{args.arm} it {it} loss {np.mean(hist[-100:]):.4f} ({time.time() - t0:.0f}s)")
    r = validate(tr, rec, val, ids, ts, W) | {"iterations": it, "minutes": args.minutes}
    log(f"{args.arm} val {r}")
    (args.work / f"val-{args.arm}.json").write_text(json.dumps(r))
    torch.save({"W": [w.detach().cpu() for w in W],
                "lora": {k: v.detach().cpu() for k, v in tr.state_dict(keep_vars=True).items() if k.endswith((".a", ".b"))},
                "rank": args.rank}, args.work / f"heal-{args.arm}.pt")
    np.savez(args.work / f"bridge-{args.arm}.npz",
             **{f"{GAP[0]}-{GAP[1]}_s{s}": W[s].detach().cpu().numpy() for s in range(4)},
             **{k: ridge[k] for k in ridge.files if k.startswith("gate_")})


def render(args) -> None:
    from PIL import Image
    blob = torch.load(args.work / "data.pt", weights_only=False)
    meta, probe_emb = blob["meta"], blob["probe_emb"]
    ids = {"img": meta["img"].to(DEV), "txt": meta["txt"].to(DEV)}
    ts = meta["t"]
    ridge = np.load(args.bridges)
    pipe = load_pipe(False)
    tr = pipe.transformer
    lora = add_lora(tr, args.rank)
    arms = {"ridge": None} | {a: torch.load(args.work / f"heal-{a}.pt") for a in ("local", "global", "global0", "long")
                              if (args.work / f"heal-{a}.pt").exists()}
    out = args.work / "png"
    out.mkdir(exist_ok=True)
    orig = tr.forward

    def student_fwd(W):
        def fwd(**kw):
            t = kw["timestep"]
            s = int(np.argmin([abs(float(t[0]) - x) for x in ts]))
            ctx = tr.context_embedder(kw["encoder_hidden_states"])
            return (student(tr, ctx, kw["hidden_states"], t, ids, W[s])[1],)
        return fwd

    def img(cid, seed):
        return pipe(prompt_embeds=probe_emb[cid][None].to(DEV), height=512, width=512, num_inference_steps=4, guidance_scale=1.0,
                    generator=torch.Generator(DEV).manual_seed(seed)).images[0]

    rows = []
    with torch.no_grad():
        for cid in probe_emb:
            for seed in (1, 2):
                img(cid, seed).save(out / f"{cid}-s{seed}-teacher.png")
        for arm, st in arms.items():
            if st is None:
                W = [torch.from_numpy(ridge[f"{GAP[0]}-{GAP[1]}_s{s}"]).float().to(DEV) for s in range(4)]
                LoRA.on = False
            else:
                W = [w.to(DEV) for w in st["W"]]
                sd = tr.state_dict(keep_vars=True)
                for k, v in st["lora"].items():
                    sd[k].data.copy_(v.to(DEV))
                LoRA.on = True
            tr.forward = student_fwd(W)
            for cid in probe_emb:
                for seed in (1, 2):
                    p = out / f"{cid}-s{seed}-{arm}.png"
                    img(cid, seed).save(p)
                    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
                    b = np.asarray(Image.open(out / f"{cid}-s{seed}-teacher.png").convert("RGB"), dtype=np.float32)
                    rows.append({"arm": arm, "case": cid, "seed": seed, "mad": float(np.abs(a - b).mean())})
            tr.forward = orig
            log(f"{arm} mean MAD {np.mean([r['mad'] for r in rows if r['arm'] == arm]):.1f}")
    (args.work / "render.json").write_text(json.dumps(rows, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=("data", "train", "render"))
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--probe", type=Path, help="json list of {id, prompt}")
    p.add_argument("--bridges", type=Path, help="ridge npz with 13-22_s* and gate_s*")
    p.add_argument("--arm", choices=("ridge", "local", "global", "global0", "long"))
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-val", type=int, default=40)
    p.add_argument("--minutes", type=float, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr-bridge", type=float, default=2e-5)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", type=Path, help="heal-*.pt to continue from")
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    {"data": data, "train": train, "render": render}[args.cmd](args)


if __name__ == "__main__":
    main()
