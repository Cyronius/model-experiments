# Turn a Qwen3.5 teacher into a smaller-shaped student: fold one shared PCA basis into the weights, shrink the MLPs, heal half-layer by half-layer against the teacher (2026-10-20).
import argparse
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import KLAccumulator, Runner, add_session_args, kl_per_token, load_model, save_json, session_from

ap = argparse.ArgumentParser()
add_session_args(ap, bs=8)
ap.add_argument("--shape-from", default="Qwen/Qwen3.5-0.8B", help="model whose config (and module layout) the student takes")
ap.add_argument("--self-chat", type=int, default=192, help="the teacher's own chat answers added to the calibration rows")
ap.add_argument("--steps", type=int, default=1000, help="healing steps per half-layer (0 = no healing)")
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--final-steps", type=int, default=300, help="steps fitting the final norm on KL to the teacher")
ap.add_argument("--ls-tokens", type=int, default=65536, help="tokens used for the MLP neuron choice and least-squares refit")
ap.add_argument("--identity-test", action="store_true", help="full-width basis, no shrink, no healing: must reproduce the teacher")
ap.add_argument("--force-untie", action="store_true", help="dry-run aid: give a tied teacher its own head copy")
ap.add_argument("--no-curve", action="store_true", help="skip the KL checks every 4 layers (they need spare GPU memory)")
ap.add_argument("--out", default="")
ap.add_argument("--tag", default="")
args = ap.parse_args()

S = session_from(args)
r, teacher, tok, dev = S.runner, S.model, S.tok, S.device
L, d = r.L, r.cfg.hidden_size
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


@torch.no_grad()
def self_chat_seqs(n):
    """The teacher's answers to UltraChat prompts through its real chat template, half with thinking on, packed into T-token rows."""
    import pandas as pd
    from pathlib import Path
    from common import _local_or_download
    path = _local_or_download("HuggingFaceH4/ultrachat_200k", repo_type="dataset", allow_patterns=["data/test_sft-*.parquet"])
    df = pd.read_parquet(next(Path(path).rglob("test_sft-*.parquet")))
    prompts = [conv[0]["content"][:1500] for conv in df.messages[:n]]
    tok.padding_side = "left"
    stream = []
    for i in range(0, n, 16):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=(j % 2 == 0)) for j, p in enumerate(prompts[i:i + 16])]
        enc = tok(texts, return_tensors="pt", padding=True).to(dev)
        g = teacher.generate(**enc, max_new_tokens=320, do_sample=False)
        for row, mask in zip(g, torch.cat([enc.attention_mask, torch.ones_like(g[:, enc.input_ids.shape[1]:])], 1)):
            stream += row[mask.bool()].tolist()
    return [("selfchat", stream[k:k + S.T], [1] * S.T) for k in range(0, len(stream) - S.T, S.T)]


calib = S.data["fit"] + (self_chat_seqs(args.self_chat) if args.self_chat and not args.identity_test else [])
IDS = torch.stack([torch.tensor(x[1]) for x in calib])
n = len(IDS)
S.eval_batches()
log(f"{n} calibration rows of {S.T} tokens")


def sub_update(run, s, h, ctx):
    i, half = divmod(s, 2)
    return run.mixer(i, h, ctx) if half == 0 else run.mlp(i, h)


@torch.no_grad()
def step_all(run, s, h):
    out = []
    for i in range(0, n, 8):
        hb = h[i:i + 8].float()
        _, ctx = run.prepare(IDS[i:i + 8])
        out.append((hb + sub_update(run, s, hb, ctx)).half())
    return torch.cat(out)


def flat(h):
    return h.reshape(-1, h.shape[-1]).float()


# Pass 1: one basis for the whole stream, from the pooled (uncentered, so no bias terms are needed) second moment of every site.
with torch.no_grad():
    h = torch.cat([r.prepare(IDS[i:i + 8])[0].half() for i in range(0, n, 8)])
    M = torch.zeros(d, d, device=dev)
    for s in range(2 * L + 1):
        X = flat(h)
        C = X.T @ X / len(X)
        M += C / C.trace()
        if s < 2 * L:
            h = step_all(r, s, h)
    ev, V = torch.linalg.eigh(M)
    V = V[:, ev.argsort(descending=True)]
student = copy.deepcopy(teacher) if args.identity_test else load_model(args.shape_from)[0]
ds = student.config.get_text_config().hidden_size if hasattr(student.config, "get_text_config") else student.config.hidden_size
if args.force_untie:
    teacher.lm_head.weight = nn.Parameter(teacher.lm_head.weight.detach().clone(), requires_grad=False)
untied = teacher.lm_head.weight.data_ptr() != r.lm.embed_tokens.weight.data_ptr()
if untied and not args.identity_test:
    student.lm_head = nn.Linear(ds, teacher.lm_head.weight.shape[0], bias=False, device=dev)
    student.lm_head.weight.requires_grad_(False)
    student.config.tie_word_embeddings = False
    student.config.get_text_config().tie_word_embeddings = False
# The identity test uses the plain identity so even the final norm folds exactly.
P = torch.eye(d, device=dev) if args.identity_test else V[:, :ds].contiguous()
log(f"basis: {ds} of {d} dims keep {(ev.sort(descending=True).values[:ds].sum() / ev.sum()).item():.3f} of the pooled energy")
sr = Runner(student)
assert untied or student.lm_head.weight.data_ptr() == sr.lm.embed_tokens.weight.data_ptr(), "student head must stay tied to its embeddings"


def norm_scale(X):
    """Mean of rms(projected)/rms(original): folding a norm into the next layer must undo the change in RMS width."""
    Z = X @ P
    return ((Z.pow(2).mean(-1) + 1e-6).sqrt() / (X.pow(2).mean(-1) + 1e-6).sqrt()).mean().item()


# Fold: readers get diag(1+w) P (times the RMS fix), writers get P^T. MLPs keep the full width until the shrink step.
T_layers, S_layers = r.lm.layers, sr.lm.layers
full_mlp = {}
with torch.no_grad():
    sr.lm.embed_tokens.weight.copy_(r.lm.embed_tokens.weight @ P)
    if untied and not args.identity_test:
        student.lm_head.weight.copy_(teacher.lm_head.weight @ P)
    h = torch.cat([r.prepare(IDS[i:i + 8])[0].half() for i in range(0, min(n, 64), 8)])
    sub = IDS[:min(n, 64)]
    scales = []
    for li in range(L):
        tl, sl = T_layers[li], S_layers[li]
        for half in (0, 1):
            s = 2 * li + half
            c = norm_scale(flat(h))
            scales.append(c)
            norm_t, norm_s = (tl.input_layernorm, sl.input_layernorm) if half == 0 else (tl.post_attention_layernorm, sl.post_attention_layernorm)
            read = c * (1 + norm_t.weight)[:, None] * P
            norm_s.weight.zero_()
            if half == 0:
                mt, ms = (tl.linear_attn, sl.linear_attn) if r.types[li] == "linear_attention" else (tl.self_attn, sl.self_attn)
                for name, p in mt.named_parameters():
                    tgt = dict(ms.named_parameters())[name]
                    if name in ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight",
                                "q_proj.weight", "k_proj.weight", "v_proj.weight"):
                        tgt.copy_(p @ read)
                    elif name in ("out_proj.weight", "o_proj.weight"):
                        tgt.copy_(P.T @ p)
                    else:
                        tgt.copy_(p)
            else:
                full_mlp[li] = tuple(w.cpu() for w in (tl.mlp.gate_proj.weight @ read, tl.mlp.up_proj.weight @ read, P.T @ tl.mlp.down_proj.weight))
                if args.identity_test:
                    for w, full in zip((sl.mlp.gate_proj.weight, sl.mlp.up_proj.weight, sl.mlp.down_proj.weight), full_mlp[li]):
                        w.copy_(full)
            h = torch.cat([(hb.float() + sub_update(r, s, hb.float(), r.prepare(sub[j:j + 8])[1])).half()
                           for j, hb in zip(range(0, len(sub), 8), h.split(8))])
    cf = norm_scale(flat(h))
    g = torch.diagonal(P.T @ ((1 + r.lm.norm.weight)[:, None] * P)) / cf
    sr.lm.norm.weight.copy_(g - 1)
log(f"folded; RMS fixes range {min(scales):.3f}-{max(scales):.3f}, final {cf:.3f}")


@torch.no_grad()
def student_kl(upto=None):
    """KL of the student against the teacher on the eval batches; with `upto`, run the student to that site, lift by P and let the teacher finish."""
    acc = KLAccumulator()
    for b in S.eval_batches():
        emb, ctx = sr.prepare(b["ids"])
        if upto is None:
            logits, _ = sr.run(emb, 0, ctx, positions=b["pos"], keep_first=False)
        else:
            _, rec = sr.run(emb, 0, ctx, record={upto}, positions=torch.tensor([0]), keep_first=False)
            logits, _ = r.run(rec[upto] @ P.T, upto, b["ctx"], positions=b["pos"], keep_first=False)
        acc.add(kl_per_token(b["ref_lp"].to(dev), logits), b["doms"], b["focus"])
    return acc.report()


out = {"args": vars(args), "basis_energy": (ev.sort(descending=True).values[:ds].sum() / ev.sum()).item(), "curve": []}
if args.identity_test:
    kl = student_kl()
    log(f"IDENTITY TEST: KL {kl['all']['mean']:.6f} (should be ~0)")
    save_json({"identity_kl": kl}, S.results / "projected_identity.json")
    raise SystemExit

out["folded_lift"] = {}
cfg_s = student.config.get_text_config() if hasattr(student.config, "get_text_config") else student.config
m_new = cfg_s.intermediate_size

# Pass 2: walk the student's own stream; shrink each MLP by least squares, then heal each half-layer to the teacher's projected state.
with torch.no_grad():
    x = torch.cat([r.prepare(IDS[i:i + 8])[0].half() for i in range(0, n, 8)])
    z = torch.cat([sr.prepare(IDS[i:i + 8])[0].half() for i in range(0, n, 8)])
for s in range(2 * L):
    li, half = divmod(s, 2)
    x = step_all(r, s, x)
    target = torch.cat([(xb.float() @ P).half() for xb in x.split(8)])
    if half == 1:
        with torch.no_grad():
            Gf, Uf, Df = (w.to(dev) for w in full_mlp.pop(li))
            norm = S_layers[li].post_attention_layernorm
            pick = torch.randperm(n * S.T, device="cpu")[:args.ls_tokens].to(z.device)
            zp = z.reshape(-1, z.shape[-1])[pick].float()
            Xn = norm(zp)
            Phi = F.silu(Xn @ Gf.T) * (Xn @ Uf.T)
            Y = target.reshape(-1, target.shape[-1])[pick].float() - zp
            keep = (Phi.pow(2).mean(0) * Df.pow(2).sum(0)).topk(m_new).indices.sort().values
            Pk = Phi[:, keep]
            A = Pk.T @ Pk
            A += 1e-3 * A.diagonal().mean() * torch.eye(m_new, device=dev)
            W = torch.linalg.solve(A, Pk.T @ Y)
            mlp = S_layers[li].mlp
            mlp.gate_proj.weight.copy_(Gf[keep])
            mlp.up_proj.weight.copy_(Uf[keep])
            mlp.down_proj.weight.copy_(W.T)
            del Phi, Pk, Xn, Y, zp, A, W, Gf, Uf, Df
            torch.cuda.empty_cache()
    first = last = None
    if args.steps:
        mods = ([S_layers[li].input_layernorm, S_layers[li].linear_attn if r.types[li] == "linear_attention" else S_layers[li].self_attn]
                if half == 0 else [S_layers[li].post_attention_layernorm, S_layers[li].mlp])
        params = [p for m in mods for p in m.parameters()]
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
        var = flat(target[:64]).var(0).mean()
        for step in range(args.steps):
            idx = torch.randperm(n)[:args.batch]
            zb, tb = z[idx].float(), target[idx].float()
            _, ctx = sr.prepare(IDS[idx])
            loss = (zb + sub_update(sr, s, zb, ctx) - tb).pow(2).mean() / var
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            first = loss.item() if step == 0 else first
            last = loss.item()
        for p in params:
            p.requires_grad_(False)
    z = step_all(sr, s, z)
    msg = f"half-layer {s}: heal loss {first} -> {last}"
    if half == 1 and li % 4 == 3 and not args.no_curve:
        kl = student_kl(upto=s + 1)
        out["curve"].append({"layer": li, "kl": kl["all"]["mean"]})
        msg += f"  KL with the teacher finishing from here: {kl['all']['mean']:.4f}"
    log(msg)
    save_json(out, S.results / f"projected{args.tag}.json")

# Final norm: the only weight between the last stream and the tied output head, fitted on KL to the teacher.
if args.final_steps:
    w = sr.lm.norm.weight
    w.requires_grad_(True)
    opt = torch.optim.Adam([w], lr=1e-3)
    for step in range(args.final_steps):
        idx = torch.randperm(n)[:4]
        pos = torch.arange(1, S.T, 4)
        with torch.no_grad():
            emb, ctx = r.prepare(IDS[idx])
            ref = F.log_softmax(r.run(emb, 0, ctx, positions=pos, keep_first=False)[0].float(), -1)
            hs = z[idx].float()
        lp = F.log_softmax(student.lm_head(sr.lm.norm(hs[:, pos])).float(), -1)
        loss = (ref.exp() * (ref - lp)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    w.requires_grad_(False)
    log(f"final norm fitted, train KL {loss.item():.4f}")

del z, x, target
torch.cuda.empty_cache()
kl = student_kl()
out["after"] = kl
out["params"] = sum(p.numel() for p in student.parameters())
log("STUDENT vs teacher: KL " + f"{kl['all']['mean']:.4f}  " + "  ".join(f"{dom} {kl[dom]['mean']:.4f}" for dom in kl if dom != "all")
    + f"  params {out['params'] / 1e9:.3f}B")
save_json(out, S.results / f"projected{args.tag}.json")
if args.out:
    student.to(torch.bfloat16).save_pretrained(args.out)
    tok.save_pretrained(args.out)
    save_json(out, f"{args.out}/build_report.json")
    log(f"saved {args.out}")
