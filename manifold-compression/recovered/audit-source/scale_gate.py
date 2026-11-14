# Damage isolation: cut the 2B to the 0.8B shape one piece at a time (width fold, MLP shrink) and vary the data behind each fit; marks are thousands of tokens (2026-10-21).
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import KLAccumulator, build_data, kl_per_token, load_model, logprobs, save_json, set_threads, _wikitext

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/damage-isolation"
T = 512
ROW_TOK = T - 1
MARKS_ACT = {"a32": 64, "a128": 256, "a512": 1024}
MARKS_FIT = {"f8": 16, "f32": 64, "f128": 256}
# Conditions: (name, activation mark that picks the basis and the kept neurons, fit mark for the MLP refit)
CONDS = {"a32_f32": ("a32", "f32"), "a128_f8": ("a128", "f8"), "a128_f32": ("a128", "f32"),
         "a128_f128": ("a128", "f128"), "a512_f32": ("a512", "f32")}
ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["stats", "fit", "build"])
ap.add_argument("--model", default="../models/dry/q2b")
ap.add_argument("--k", type=int, default=1024, help="kept width of the shared basis")
ap.add_argument("--keep", type=int, default=3584, help="kept MLP neurons")
ap.add_argument("--rows", type=int, default=1024, help="rows of the fit stream (smoke runs use fewer)")
ap.add_argument("--eval-rows", type=int, default=48)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--threads", type=int, default=20)
ap.add_argument("--arms", default="", help="build stage: comma list of arms; default all")
ap.add_argument("--smoke", action="store_true", help="tiny marks for a debug run")
ap.add_argument("--identity", action="store_true", help="build stage check: width folds use the identity basis, so KL must be ~0")
ap.add_argument("--no-gguf", action="store_true")
args = ap.parse_args()
if args.smoke:
    OUT = HERE / "results/damage-isolation-smoke"
    MARKS_ACT = {"a32": 8, "a128": 16, "a512": 32}
    MARKS_FIT = {"f8": 4, "f32": 8, "f128": 16}
set_threads(args.threads)
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


model, tok = load_model(args.model)
L = model.config.get_text_config().num_hidden_layers
d = model.config.get_text_config().hidden_size
M_FULL = model.config.get_text_config().intermediate_size
layers = model.model.layers
log(f"loaded {args.model}: {L} layers, d={d}, mlp={M_FULL}")


def streams():
    """Fit rows (disjoint from the eval rows) and eval rows, as [n, T] id tensors."""
    split = build_data(tok, T=T, n_fit=1500, n_eval=8, n_eval_copy=16, rich=True)
    fit = [r[1] for r in split["fit"] if r[0] != "copy"]
    wiki = tok(_wikitext("train"), add_special_tokens=False).input_ids
    fit += [wiki[i * T:(i + 1) * T] for i in range(len(wiki) // T)]
    g = torch.Generator().manual_seed(0)
    order = torch.randperm(len(fit), generator=g).tolist()
    fit = torch.tensor([fit[i] for i in order[:args.rows]])
    ev = torch.tensor([r[1] for r in split["eval"]][:args.eval_rows])
    return fit, ev


def run_rows(ids, on_row_done):
    """Run the transformer body (no vocab head) on rows in batches, with hooks collecting what the caller wants."""
    with torch.no_grad():
        for i in range(0, len(ids), args.bs):
            model.model(input_ids=ids[i:i + args.bs])
            on_row_done(min(i + args.bs, len(ids)))


class Stats:
    """Pass A: per-site second moments (-> shared basis) and MLP neuron importance, snapshotted at row marks."""

    def __init__(self, marks):
        self.marks = marks
        self.G = {s: torch.zeros(d, d, dtype=torch.float64) for s in range(2 * L + 1)}
        self.imp = [torch.zeros(M_FULL, dtype=torch.float64) for _ in range(L)]
        self.snap = {}
        self.rows = 0
        self.hooks = []
        for l in range(L):
            self.hooks.append(layers[l].input_layernorm.register_forward_pre_hook(lambda m, a, s=2 * l: self.site(s, a[0])))
            self.hooks.append(layers[l].post_attention_layernorm.register_forward_pre_hook(lambda m, a, s=2 * l + 1: self.site(s, a[0])))
            self.hooks.append(layers[l].mlp.register_forward_pre_hook(lambda m, a, l=l: self.mlp(l, a[0])))
        self.hooks.append(model.model.norm.register_forward_pre_hook(lambda m, a: self.site(2 * L, a[0])))

    def site(self, s, x):
        x = x[:, 1:].reshape(-1, d)
        self.G[s] += (x.T @ x).double()

    def mlp(self, l, xn):
        x = xn[:, 1:].reshape(-1, d)
        mlp = layers[l].mlp
        phi = F.silu(x @ mlp.gate_proj.weight.T) * (x @ mlp.up_proj.weight.T)
        self.imp[l] += phi.pow(2).sum(0).double()

    def done(self, rows):
        self.rows = rows
        for name, n in self.marks.items():
            if name not in self.snap and rows >= n:
                tokens = n * ROW_TOK
                M = sum(self.G[s] / self.G[s].trace() for s in self.G)
                ev, V = torch.linalg.eigh(M)
                P = V[:, ev.argsort(descending=True)][:, :args.k].float().contiguous()
                energy = (ev.sort(descending=True).values[:args.k].sum() / ev.sum()).item()
                self.snap[name] = {"P": P, "energy": energy, "imp": [i.clone() / tokens for i in self.imp]}
                log(f"stats snapshot {name}: {tokens} tokens, basis keeps {energy:.3f}")


def pick_keep(imp_l, l):
    dn = layers[l].mlp.down_proj.weight
    score = imp_l.float() * dn.pow(2).sum(0)
    return score.topk(args.keep).indices.sort().values


if args.stage == "stats":
    fit, ev = streams()
    log(f"fit stream {tuple(fit.shape)}, eval {tuple(ev.shape)}")
    st = Stats({k: v for k, v in MARKS_ACT.items() if v <= len(fit)})
    run_rows(fit, st.done)
    for h in st.hooks:
        h.remove()
    torch.save({n: {"P": s["P"], "energy": s["energy"], "imp": s["imp"]} for n, s in st.snap.items()}, OUT / "stats.pt")
    torch.save({"fit": fit, "eval": ev}, OUT / "rows.pt")
    # The oracle fits on the eval rows themselves.
    st2 = Stats({"or": len(ev)})
    run_rows(ev, st2.done)
    for h in st2.hooks:
        h.remove()
    torch.save(st2.snap["or"], OUT / "stats_oracle.pt")
    log("stats done")
    raise SystemExit


rows = torch.load(OUT / "rows.pt")
fit_ids, ev_ids = rows["fit"], rows["eval"]
stats = torch.load(OUT / "stats.pt")
stats["or"] = torch.load(OUT / "stats_oracle.pt")


class Fit:
    """Pass B and C on one row stream: RMS fix per site for each basis, and the MLP least-squares refit per condition."""

    def __init__(self, conds, bases, fit_marks):
        self.conds, self.bases, self.fit_marks = conds, bases, fit_marks
        self.c = {b: {s: [0.0, 0] for s in range(2 * L + 1)} for b in bases}
        self.keep = {name: [pick_keep(stats[a]["imp"][l], l) for l in range(L)] for name, (a, _) in conds.items()}
        self.A = {name: [torch.zeros(args.keep, args.keep, dtype=torch.float64) for _ in range(L)] for name in conds}
        self.B = {name: [torch.zeros(args.keep, d, dtype=torch.float64) for _ in range(L)] for name in conds}
        self.out = {}
        self.rows = 0
        self.hooks = []
        for l in range(L):
            self.hooks.append(layers[l].input_layernorm.register_forward_pre_hook(lambda m, a, s=2 * l: self.site(s, a[0])))
            self.hooks.append(layers[l].post_attention_layernorm.register_forward_pre_hook(lambda m, a, s=2 * l + 1: self.site(s, a[0])))
            self.hooks.append(layers[l].mlp.register_forward_pre_hook(lambda m, a, l=l: self.mlp(l, a[0])))
        self.hooks.append(model.model.norm.register_forward_pre_hook(lambda m, a: self.site(2 * L, a[0])))

    def active(self, name):
        return self.rows < self.fit_marks[self.conds[name][1]]

    def site(self, s, x):
        x = x[:, 1:].reshape(-1, d)
        r0 = (x.pow(2).mean(-1) + 1e-6).sqrt()
        for b in self.bases:
            P = stats[b]["P"]
            r1 = ((x @ P).pow(2).mean(-1) + 1e-6).sqrt()
            ratio = (r1 / r0)
            self.c[b][s][0] += ratio.sum().item()
            self.c[b][s][1] += ratio.numel()

    def mlp(self, l, xn):
        todo = [n for n in self.conds if self.active(n)]
        if not todo:
            return
        x = xn[:, 1:].reshape(-1, d)
        mlp = layers[l].mlp
        phi = F.silu(x @ mlp.gate_proj.weight.T) * (x @ mlp.up_proj.weight.T)
        y = phi @ mlp.down_proj.weight.T
        for n in todo:
            pk = phi[:, self.keep[n][l]]
            self.A[n][l] += (pk.T @ pk).double()
            self.B[n][l] += (pk.T @ y).double()

    def done(self, rows):
        self.rows = rows
        for n, (_, fm) in self.conds.items():
            if n not in self.out and rows >= self.fit_marks[fm]:
                downs = []
                for l in range(L):
                    A = self.A[n][l].clone()
                    A += 1e-3 * A.diagonal().mean() * torch.eye(args.keep, dtype=A.dtype)
                    W = torch.linalg.solve(A, self.B[n][l]).float()
                    downs.append(W.T.contiguous())
                self.out[n] = {"keep": self.keep[n], "down": downs}
                log(f"refit {n}: {self.fit_marks[fm] * ROW_TOK} tokens")

    def c_table(self):
        return {b: [v[0] / v[1] for v in (self.c[b][s] for s in range(2 * L + 1))] for b in self.bases}


if args.stage == "fit":
    fits = Fit(CONDS, ["a32", "a128", "a512"], MARKS_FIT)
    run_rows(fit_ids[:max(MARKS_FIT.values())], fits.done)
    for h in fits.hooks:
        h.remove()
    # Basis scale factors need all of the fit tokens used for the largest fit mark, which is enough for a mean.
    torch.save({"c": fits.c_table(), "refits": fits.out}, OUT / "fits.pt")
    n_or = len(ev_ids)
    orc = Fit({"or": ("or", "or")}, ["or"], {"or": n_or})
    run_rows(ev_ids, orc.done)
    for h in orc.hooks:
        h.remove()
    torch.save({"c": orc.c_table(), "refits": orc.out}, OUT / "fits_oracle.pt")
    log("fit done")
    raise SystemExit


# ---------- build stage ----------
fits = torch.load(OUT / "fits.pt")
orc = torch.load(OUT / "fits_oracle.pt")
cs = fits["c"] | orc["c"]
refits = fits["refits"] | orc["refits"]
import copy
import shutil
import subprocess
import sys

CONVERT = Path("C:/code/llama.cpp/convert_hf_to_gguf.py")
GGUF_DIR = HERE.parent / "models/damage-isolation"
GGUF_DIR.mkdir(parents=True, exist_ok=True)
READERS = {"linear_attn": ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"], "self_attn": ["q_proj", "k_proj", "v_proj"]}
WRITER = {"linear_attn": "out_proj", "self_attn": "o_proj"}


def read_fold(W, norm_w, Q, c):
    return (c * W * (1 + norm_w)[None, :]) @ Q


@torch.no_grad()
def build_arm(width, mlp):
    """width: None or an activation mark name ('a64', 'a256', 'a1M', 'or'); mlp: None or a condition name."""
    m = copy.deepcopy(model)
    if width is not None:
        P = stats[width]["P"]
        Q = torch.eye(d) if args.identity else P @ P.T
        c = [1.0] * (2 * L + 1) if args.identity else cs[width]
    for l in range(L):
        lay = m.model.layers[l]
        mlp_mod = lay.mlp
        down = mlp_mod.down_proj.weight
        gate, up = mlp_mod.gate_proj.weight, mlp_mod.up_proj.weight
        if mlp is not None:
            keep = refits[mlp]["keep"][l]
            mask = torch.zeros(M_FULL)
            mask[keep] = 1
            new_down = torch.zeros(d, M_FULL)
            new_down[:, keep] = refits[mlp]["down"][l]
            gate = gate * mask[:, None]
            up = up * mask[:, None]
            down = new_down
        if width is not None:
            name = "linear_attn" if hasattr(lay, "linear_attn") else "self_attn"
            mix = getattr(lay, name)
            for r in READERS[name]:
                getattr(mix, r).weight.copy_(read_fold(getattr(mix, r).weight, lay.input_layernorm.weight, Q, c[2 * l]))
            getattr(mix, WRITER[name]).weight.copy_(Q @ getattr(mix, WRITER[name]).weight)
            lay.input_layernorm.weight.zero_()
            gate = read_fold(gate, lay.post_attention_layernorm.weight, Q, c[2 * l + 1])
            up = read_fold(up, lay.post_attention_layernorm.weight, Q, c[2 * l + 1])
            down = Q @ down
            lay.post_attention_layernorm.weight.zero_()
        mlp_mod.gate_proj.weight.copy_(gate)
        mlp_mod.up_proj.weight.copy_(up)
        mlp_mod.down_proj.weight.copy_(down)
    if width is not None:
        emb = model.model.embed_tokens.weight @ Q
        head = read_fold(model.lm_head.weight, model.model.norm.weight, Q, c[2 * L])
        m.model.embed_tokens.weight = torch.nn.Parameter(emb.contiguous(), requires_grad=False)
        m.lm_head.weight = torch.nn.Parameter(head.contiguous(), requires_grad=False)
        m.model.norm.weight.zero_()
        m.config.tie_word_embeddings = False
        m.config.get_text_config().tie_word_embeddings = False
    return m


ARMS = {"F_a32": ("a32", None), "F_a128": ("a128", None), "F_a512": ("a512", None), "F_or": ("or", None),
        "M_a32_f32": (None, "a32_f32"), "M_a128_f8": (None, "a128_f8"), "M_a128_f32": (None, "a128_f32"),
        "M_a128_f128": (None, "a128_f128"), "M_a512_f32": (None, "a512_f32"), "M_or": (None, "or"),
        "FM_a128_f32": ("a128", "a128_f32"), "FM_or": ("or", "or")}
want = args.arms.split(",") if args.arms else list(ARMS)

# Reference log-probs of the stock model on the eval rows (every 8th position).
pos = torch.arange(1, T, 8)
with torch.no_grad():
    ref = [logprobs(model(input_ids=ev_ids[i:i + 4]).logits[:, pos]) for i in range(0, len(ev_ids), 4)]
results = {}
for arm in want:
    w, ml = ARMS[arm]
    m = build_arm(w, ml).eval()
    kls = []
    with torch.no_grad():
        for j, i in enumerate(range(0, len(ev_ids), 4)):
            kls.append(kl_per_token(ref[j], m(input_ids=ev_ids[i:i + 4]).logits[:, pos]).flatten())
    kl = torch.cat(kls)
    results[arm] = {"kl_mean": kl.mean().item(), "kl_p95": kl.quantile(0.95).item()}
    log(f"{arm}: KL {results[arm]['kl_mean']:.4f}  p95 {results[arm]['kl_p95']:.4f}")
    save_json(results, OUT / ("kl_identity.json" if args.identity else "kl.json"))
    if args.no_gguf:
        continue
    hf = OUT / "hf" / arm
    m.to(torch.bfloat16).save_pretrained(hf)
    tok.save_pretrained(hf)
    g = GGUF_DIR / f"di-{arm}.gguf"
    with open(OUT / f"convert_{arm}.log", "w") as lf:
        subprocess.run([sys.executable, str(CONVERT), str(hf), "--outfile", str(g), "--outtype", "bf16", "--no-mtp"],
                       check=True, stdout=lf, stderr=subprocess.STDOUT)
    shutil.rmtree(hf)
    del m
    log(f"{arm}: {g.name} {g.stat().st_size / 1e9:.2f} GB")
log("build done")
