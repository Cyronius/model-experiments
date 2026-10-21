# Part-swap test: put one stock 0.8B part into our projected 0.8B at a time and check chat coherence (2026-10-19).
import argparse
import copy
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import gguf
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

HUB = os.path.expanduser("~/.cache/huggingface/hub")
HERE = Path(__file__).resolve().parent
OURS_GGUF = HERE.parent / "models/manifold-compression/mc-proj-0.8b-tail.gguf"
GGUF_DIR = HERE.parent / "models/manifold-compression/part-swap"
CONVERT = Path("C:/code/llama.cpp/convert_hf_to_gguf.py")
SMOKE = HERE / "smoke"
OUT = HERE / "results/part-swap-20261019"
VARIANTS = ["ours", "stock", "mlp", "attn", "io", "ctrl-mlp", "ctrl-attn", "ctrl-io", "ctrl-all"]
ap = argparse.ArgumentParser()
ap.add_argument("variants", nargs="*", default=VARIANTS)
ap.add_argument("--mode", choices=["gguf", "hf"], default="gguf",
                help="gguf: fold each variant into a standard checkpoint and convert it for llama.cpp; hf: run it on the CPU with explicit maps")
ap.add_argument("--max-new", type=int, default=400, help="hf mode only")
ap.add_argument("--refit", action="store_true", help="recompute the basis maps even if a saved copy exists")
args = ap.parse_args()
torch.set_num_threads(12)
torch.manual_seed(0)
t0 = time.time()
OUT.mkdir(parents=True, exist_ok=True)
GGUF_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.time() - t0:.0f}s] {msg}", flush=True)


snap = glob.glob(f"{HUB}/models--Qwen--Qwen3.5-0.8B/snapshots/*/")[0]
tok = AutoTokenizer.from_pretrained(snap)
stock = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32).eval()
for p in stock.parameters():
    p.requires_grad_(False)

# GGUF -> HF transforms, checked against the stock 0.8B's own GGUF: norms with (1 + w) are stored as 1 + w, A_log as -exp(A_log).
minus1, ident = (lambda x: x - 1), (lambda x: x)
BLOCK = {"attn_norm.weight": ("input_layernorm.weight", minus1),
         "post_attention_norm.weight": ("post_attention_layernorm.weight", minus1),
         "ssm_a": ("linear_attn.A_log", lambda x: torch.log(-x)),
         "ssm_conv1d.weight": ("linear_attn.conv1d.weight", lambda x: x.unsqueeze(1)),
         "ssm_dt.bias": ("linear_attn.dt_bias", ident), "ssm_alpha.weight": ("linear_attn.in_proj_a.weight", ident),
         "ssm_beta.weight": ("linear_attn.in_proj_b.weight", ident), "attn_qkv.weight": ("linear_attn.in_proj_qkv.weight", ident),
         "attn_gate.weight": ("linear_attn.in_proj_z.weight", ident), "ssm_norm.weight": ("linear_attn.norm.weight", ident),
         "ssm_out.weight": ("linear_attn.out_proj.weight", ident), "ffn_gate.weight": ("mlp.gate_proj.weight", ident),
         "ffn_up.weight": ("mlp.up_proj.weight", ident), "ffn_down.weight": ("mlp.down_proj.weight", ident),
         "attn_q.weight": ("self_attn.q_proj.weight", ident), "attn_k.weight": ("self_attn.k_proj.weight", ident),
         "attn_v.weight": ("self_attn.v_proj.weight", ident), "attn_output.weight": ("self_attn.o_proj.weight", ident),
         "attn_q_norm.weight": ("self_attn.q_norm.weight", minus1), "attn_k_norm.weight": ("self_attn.k_norm.weight", minus1)}
READERS = {"linear_attn": ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"], "self_attn": ["q_proj", "k_proj", "v_proj"]}
WRITER = {"linear_attn": "out_proj", "self_attn": "o_proj"}


def gguf_tensor(t):
    shape = tuple(int(x) for x in reversed(t.shape))
    if t.tensor_type.name == "BF16":
        raw = np.ascontiguousarray(t.data).view(np.uint16).reshape(shape)
        return torch.from_numpy((raw.astype(np.uint32) << 16).view(np.float32))
    assert t.tensor_type.name == "F32", t.tensor_type.name
    return torch.from_numpy(np.array(t.data, dtype=np.float32).reshape(shape))


@torch.no_grad()
def load_ours():
    m = copy.deepcopy(stock)
    sd = m.state_dict()
    done = set()
    for t in gguf.GGUFReader(str(OURS_GGUF)).tensors:
        if t.name == "token_embd.weight":
            name, fn = "model.embed_tokens.weight", ident
        elif t.name == "output_norm.weight":
            name, fn = "model.norm.weight", minus1
        else:
            _, l, rest = t.name.split(".", 2)
            sub, fn = BLOCK[rest]
            name = f"model.layers.{l}.{sub}"
        v = fn(gguf_tensor(t))
        assert sd[name].shape == v.shape, (name, sd[name].shape, v.shape)
        sd[name].copy_(v)
        done.add(name)
    missing = [k for k in sd if k not in done and k != "lm_head.weight"]
    assert not missing, missing[:5]
    assert m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()
    return m


ours = load_ours()
L = len(ours.model.layers)
log("our GGUF loaded into the 0.8B architecture")


def after(l):
    return f"in{l + 1}" if l + 1 < L else "out"


def mixer_name(layer):
    return "linear_attn" if hasattr(layer, "linear_attn") else "self_attn"


# Basis maps between our stream and the stock stream, fit on wikitext and MT-Bench chats (not the smoke prompts).
MAPS = OUT / "maps.pt"
if MAPS.exists() and not args.refit:
    maps = torch.load(MAPS)
    log("basis maps loaded")
else:
    T = 512
    wiki = (HERE.parent / "bench/data/wikitext-2-raw/wiki.test.raw").read_text(encoding="utf-8")
    ids = tok(wiki).input_ids
    rows = [ids[i * T:(i + 1) * T] for i in range(12)]
    qs = {json.loads(l)["question_id"]: json.loads(l) for l in (HERE.parent / "bench/data/mtbench.jsonl").open(encoding="utf-8")}
    stream = []
    for f in sorted((HERE / "results/pod-proj-20261019/results/proj-20261019_work/stock-2b/mtbench").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        msgs = []
        for u, a in zip(qs[d["question_id"]]["prompt"], d["answers"]):
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a["content"]}]
        stream += tok(tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)).input_ids
    rows += [stream[i * T:(i + 1) * T] for i in range(12)]
    IDS = torch.tensor(rows)
    TE = (torch.arange(len(IDS)) % 4 == 3).repeat_interleave(T - 1)
    TR = ~TE

    @torch.no_grad()
    def streams(model):
        mid, fin, ins = {l: [] for l in range(L)}, [], {l: [] for l in range(L)}
        flat = (lambda x: x[:, 1:].reshape(-1, x.shape[-1]).half())
        hooks = [model.model.layers[l].post_attention_layernorm.register_forward_pre_hook(
            lambda mod, a, l=l: mid[l].append(flat(a[0]))) for l in range(L)]
        hooks.append(model.model.norm.register_forward_pre_hook(lambda mod, a: fin.append(flat(a[0]))))
        for i in range(0, len(IDS), 4):
            o = model(input_ids=IDS[i:i + 4], output_hidden_states=True)
            for l in range(L):
                ins[l].append(flat(o.hidden_states[l]))
        for h in hooks:
            h.remove()
        S = {f"in{l}": torch.cat(ins[l]) for l in range(L)}
        S.update({f"mid{l}": torch.cat(mid[l]) for l in range(L)})
        S["out"] = torch.cat(fin)
        return S

    def ridge(X, Y, lam, center):
        mx, my = (X[TR].mean(0), Y[TR].mean(0)) if center else (torch.zeros(X.shape[1]), torch.zeros(Y.shape[1]))
        Xc, Yc = X[TR] - mx, Y[TR] - my
        A = Xc.T.double() @ Xc.double()
        A += lam * A.diagonal().mean() * torch.eye(A.shape[0], dtype=A.dtype)
        W = torch.linalg.solve(A, Xc.T.double() @ Yc.double()).float()
        b = my - mx @ W
        R = Y[TE] - (X[TE] @ W + b)
        return W, b, (1 - R.pow(2).sum() / (Y[TE] - Y[TE].mean(0)).pow(2).sum()).item()

    def rms_ratio(X, W):
        return (X.pow(2).mean(-1).sqrt() / (X @ W).pow(2).mean(-1).sqrt().clamp(min=1e-6)).mean().item()

    So, Ss = streams(ours), streams(stock)
    log("streams recorded")
    maps = {}
    for site in So:
        Xo, Xs = So[site].float(), Ss[site].float()
        Wa, ba, ra = ridge(Xo, Xs, 1e-3, True)
        Wb, bb, rb = ridge(Xs, Xo, 1e-3, True)
        Wo2s, _, rl = ridge(Xo, Xs, 1e-3, False)
        Ws2o, _, rl2 = ridge(Xs, Xo, 1e-3, False)
        maps[site] = {"o2s_aff": (Wa, ba), "s2o_aff": (Wb, bb), "o2s": Wo2s, "s2o": Ws2o,
                      "c_o2s": rms_ratio(Xo, Wo2s), "c_rt": rms_ratio(Xs, Ws2o @ Wo2s),
                      "r2": {"affine_o2s": round(ra, 3), "affine_s2o": round(rb, 3), "linear_o2s": round(rl, 3), "linear_s2o": round(rl2, 3)}}
    del So, Ss
    torch.save(maps, MAPS)
    (OUT / "maps_r2.json").write_text(json.dumps({s: v["r2"] | {"c_o2s": round(v["c_o2s"], 3)} for s, v in maps.items()}, indent=1))
    log("maps fit: " + ", ".join(f"{s} {maps[s]['r2']['linear_o2s']}" for s in ("in0", "mid6", "in12", "mid18", "out")))


def read_fold(W, norm_w, A, c):
    """Reader weights for input c * norm(x) @ A seen through a (1 + w) norm: exact for the map, approximate for the RMS."""
    return (c * W * (1 + norm_w)[None, :]) @ A.T


@torch.no_grad()
def build_gguf(v):
    if v == "ours":
        return ours
    if v == "stock":
        return stock
    if v == "mlp":
        m = copy.deepcopy(ours)
        for l in range(L):
            lay, src, mp = m.model.layers[l], stock.model.layers[l], maps[f"mid{l}"]
            w = src.post_attention_layernorm.weight
            for name in ("gate_proj", "up_proj"):
                getattr(lay.mlp, name).weight.copy_(read_fold(getattr(src.mlp, name).weight, w, mp["o2s"], mp["c_o2s"]))
            lay.mlp.down_proj.weight.copy_(maps[after(l)]["s2o"].T @ src.mlp.down_proj.weight)
            lay.post_attention_layernorm.weight.zero_()
        return m
    if v == "attn":
        m = copy.deepcopy(ours)
        for l in range(L):
            lay, src, mp = m.model.layers[l], stock.model.layers[l], maps[f"in{l}"]
            name = mixer_name(lay)
            mix, smix = getattr(lay, name), getattr(src, name)
            mix.load_state_dict(smix.state_dict())
            for r in READERS[name]:
                getattr(mix, r).weight.copy_(read_fold(getattr(smix, r).weight, src.input_layernorm.weight, mp["o2s"], mp["c_o2s"]))
            getattr(mix, WRITER[name]).weight.copy_(maps[f"mid{l}"]["s2o"].T @ getattr(smix, WRITER[name]).weight)
            lay.input_layernorm.weight.zero_()
        return m
    if v == "io":
        m = copy.deepcopy(ours)
        W, b = maps["in0"]["s2o_aff"]
        mo = maps["out"]
        emb = stock.model.embed_tokens.weight @ W + b
        head = read_fold(stock.lm_head.weight, stock.model.norm.weight, mo["o2s"], mo["c_o2s"])
        m.model.embed_tokens.weight = nn.Parameter(emb.contiguous(), requires_grad=False)
        m.lm_head.weight = nn.Parameter(head.contiguous(), requires_grad=False)
        m.model.norm.weight.zero_()
        m.config.tie_word_embeddings = False
        m.config.get_text_config().tie_word_embeddings = False
        return m
    if v.startswith("ctrl-"):
        # Controls: the stock model with one swap's sites (or all of them) sent through our coordinates and back.
        sites = {"ctrl-attn": {"attn"}, "ctrl-mlp": {"mlp"}, "ctrl-io": {"io"}, "ctrl-all": {"attn", "mlp", "io"}}[v]
        m = copy.deepcopy(stock)
        for l in range(L):
            lay = m.model.layers[l]
            name = mixer_name(lay)
            mix, mi, mm, mn = getattr(lay, name), maps[f"in{l}"], maps[f"mid{l}"], maps[after(l)]
            if "attn" in sites:
                for r in READERS[name]:
                    getattr(mix, r).weight.copy_(read_fold(getattr(mix, r).weight, lay.input_layernorm.weight,
                                                           mi["s2o"] @ mi["o2s"], mi["c_rt"]))
                getattr(mix, WRITER[name]).weight.copy_((mm["s2o"] @ mm["o2s"]).T @ getattr(mix, WRITER[name]).weight)
                lay.input_layernorm.weight.zero_()
            if "mlp" in sites:
                for r in ("gate_proj", "up_proj"):
                    getattr(lay.mlp, r).weight.copy_(read_fold(getattr(lay.mlp, r).weight, lay.post_attention_layernorm.weight,
                                                               mm["s2o"] @ mm["o2s"], mm["c_rt"]))
                lay.mlp.down_proj.weight.copy_((mn["s2o"] @ mn["o2s"]).T @ lay.mlp.down_proj.weight)
                lay.post_attention_layernorm.weight.zero_()
        if "io" in sites:
            (Wb, bb), (Wa, ba), mo = maps["in0"]["s2o_aff"], maps["in0"]["o2s_aff"], maps["out"]
            emb = (stock.model.embed_tokens.weight @ Wb + bb) @ Wa + ba
            head = read_fold(stock.lm_head.weight, stock.model.norm.weight, mo["s2o"] @ mo["o2s"], mo["c_rt"])
            m.model.embed_tokens.weight = nn.Parameter(emb.contiguous(), requires_grad=False)
            m.lm_head.weight = nn.Parameter(head.contiguous(), requires_grad=False)
            m.model.norm.weight.zero_()
            m.config.tie_word_embeddings = False
            m.config.get_text_config().tie_word_embeddings = False
        return m
    raise ValueError(v)


if args.mode == "gguf":
    for v in args.variants:
        m = build_gguf(v)
        d = OUT / "hf" / v
        m.save_pretrained(d)
        tok.save_pretrained(d)
        g = GGUF_DIR / f"ps-{v}.gguf"
        with open(OUT / f"convert_{v}.log", "w") as lf:
            subprocess.run([sys.executable, str(CONVERT), str(d), "--outfile", str(g), "--outtype", "bf16", "--no-mtp"],
                           check=True, stdout=lf, stderr=subprocess.STDOUT)
        shutil.rmtree(d)
        log(f"{v}: {g.name} {g.stat().st_size / 1e9:.2f} GB")
        if m is not ours and m is not stock:
            del m
    log("done")
    raise SystemExit


# hf mode (CPU fallback): the same swaps with explicit affine maps, run through Hugging Face generate.
class MapIn(nn.Module):
    """A norm that first moves the stream into another model's coordinates (affine maps)."""

    def __init__(self, norm, maps_):
        super().__init__()
        self.norm = norm
        for i, (W, b) in enumerate(maps_):
            self.register_buffer(f"W{i}", W)
            self.register_buffer(f"b{i}", b)
        self.n = len(maps_)

    def forward(self, x):
        for i in range(self.n):
            x = x @ getattr(self, f"W{i}") + getattr(self, f"b{i}")
        return self.norm(x)


class MapOut(nn.Module):
    """A block whose output change is moved into another model's coordinates (linear part only)."""

    def __init__(self, inner, Ws):
        super().__init__()
        self.inner = inner
        for i, W in enumerate(Ws):
            self.register_buffer(f"W{i}", W)
        self.n = len(Ws)

    def _map(self, y):
        for i in range(self.n):
            y = y @ getattr(self, f"W{i}")
        return y

    def forward(self, *a, **k):
        out = self.inner(*a, **k)
        return (self._map(out[0]),) + tuple(out[1:]) if isinstance(out, tuple) else self._map(out)


def build_hf(v):
    if v in ("ours", "stock"):
        return ours if v == "ours" else stock
    if v == "mlp":
        m = copy.deepcopy(ours)
        for l in range(L):
            lay, src = m.model.layers[l], stock.model.layers[l]
            lay.post_attention_layernorm = MapIn(copy.deepcopy(src.post_attention_layernorm), [maps[f"mid{l}"]["o2s_aff"]])
            lay.mlp = MapOut(copy.deepcopy(src.mlp), [maps[after(l)]["s2o_aff"][0]])
        return m
    if v == "attn":
        m = copy.deepcopy(ours)
        for l in range(L):
            lay, src = m.model.layers[l], stock.model.layers[l]
            lay.input_layernorm = MapIn(copy.deepcopy(src.input_layernorm), [maps[f"in{l}"]["o2s_aff"]])
            name = mixer_name(lay)
            setattr(lay, name, MapOut(copy.deepcopy(getattr(src, name)), [maps[f"mid{l}"]["s2o_aff"][0]]))
        return m
    if v == "io":
        m = copy.deepcopy(ours)
        W, b = maps["in0"]["s2o_aff"]
        m.model.embed_tokens = nn.Embedding.from_pretrained(stock.model.embed_tokens.weight.detach() @ W + b, freeze=True)
        m.model.norm = MapIn(copy.deepcopy(stock.model.norm), [maps["out"]["o2s_aff"]])
        head = nn.Linear(stock.lm_head.in_features, stock.lm_head.out_features, bias=False)
        head.weight = nn.Parameter(stock.lm_head.weight.detach().clone(), requires_grad=False)
        m.lm_head = head
        return m
    if v.startswith("ctrl-"):
        sites = {"ctrl-attn": {"attn"}, "ctrl-mlp": {"mlp"}, "ctrl-io": {"io"}, "ctrl-all": {"attn", "mlp", "io"}}[v]
        m = copy.deepcopy(stock)
        for l in range(L):
            lay = m.model.layers[l]
            mi, mm, mn = maps[f"in{l}"], maps[f"mid{l}"], maps[after(l)]
            if "attn" in sites:
                lay.input_layernorm = MapIn(lay.input_layernorm, [mi["s2o_aff"], mi["o2s_aff"]])
                name = mixer_name(lay)
                setattr(lay, name, MapOut(getattr(lay, name), [mm["s2o_aff"][0], mm["o2s_aff"][0]]))
            if "mlp" in sites:
                lay.post_attention_layernorm = MapIn(lay.post_attention_layernorm, [mm["s2o_aff"], mm["o2s_aff"]])
                lay.mlp = MapOut(lay.mlp, [mn["s2o_aff"][0], mn["o2s_aff"][0]])
        if "io" in sites:
            (Wb, bb), (Wa, ba) = maps["in0"]["s2o_aff"], maps["in0"]["o2s_aff"]
            m.model.embed_tokens = nn.Embedding.from_pretrained((stock.model.embed_tokens.weight.detach() @ Wb + bb) @ Wa + ba,
                                                                freeze=True)
            head = nn.Linear(stock.lm_head.in_features, stock.lm_head.out_features, bias=False)
            head.weight = nn.Parameter(stock.lm_head.weight.detach().clone(), requires_grad=False)
            m.lm_head = head
            m.model.norm = MapIn(m.model.norm, [maps["out"]["s2o_aff"], maps["out"]["o2s_aff"]])
        return m
    raise ValueError(v)


prompts = [json.loads(l) for l in (SMOKE / "prompts.jsonl").open(encoding="utf-8")]
EOS = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}


@torch.no_grad()
def generate(model, items, max_new):
    tok.padding_side = "left"
    texts = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for p in items]
    enc = tok(texts, return_tensors="pt", padding=True)
    g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=list(EOS), pad_token_id=tok.pad_token_id)
    out = []
    for p, seq in zip(items, g[:, enc.input_ids.shape[1]:].tolist()):
        stop = next((i for i, x in enumerate(seq) if x in EOS), None)
        body = seq[:stop] if stop is not None else seq
        out.append({"id": p["id"], "answer": tok.decode(body, skip_special_tokens=True),
                    "finish": "stop" if stop is not None else "length", "tokens": len(body)})
    return out


for v in args.variants:
    model = build_hf(v).eval()
    rows = generate(model, prompts, args.max_new)
    with (SMOKE / f"answers_psh-{v}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"{v}: {sum(r['finish'] == 'length' for r in rows)}/{len(rows)} answers hit the cap")
log("done")
