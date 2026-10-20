# Shared pieces for the manifold-compression measurements (2026-10-17): model loading, domain data, a layer-by-layer runner with edit hooks, KL.
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
WORK = ROOT / "work"
RESULTS = ROOT / "results"
DOMAINS = ["prose", "code", "math", "chat", "copy"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _local_or_download(name, **kw):
    if Path(name).is_dir():
        return name
    try:
        return snapshot_download(name, local_files_only=True, **kw)
    except Exception:
        return snapshot_download(name, **kw)


def load_model(name, dtype=torch.float32, device=DEVICE):
    path = _local_or_download(name)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


# ---------- data ----------

def _chunks(ids, T, n):
    return [ids[i * T:(i + 1) * T] for i in range(min(n, len(ids) // T))]


def _wikitext(split):
    text = (REPO / f"bench/data/wikitext-2-raw/wiki.{split}.raw").read_text(encoding="utf-8")
    return re.sub(r"\n\s*\n", "\n\n", text)


def _code_text(extra=False):
    path = _local_or_download("openai/openai_humaneval", repo_type="dataset")
    df = pd.read_parquet(next(Path(path).rglob("*.parquet")))
    text = "\n\n".join(df.prompt + df.canonical_solution)
    if extra:
        path = _local_or_download("google-research-datasets/mbpp", repo_type="dataset", allow_patterns=["full/*"])
        mb = pd.concat([pd.read_parquet(f) for f in sorted(Path(path).rglob("full/*.parquet"))])
        text += "\n\n" + "\n\n".join(f"# {t}\n{c}" for t, c in zip(mb.text, mb.code))
    return text


def _math_text():
    rows = [json.loads(l) for l in (REPO / "bench/data/gsm8k_raw.jsonl").open(encoding="utf-8")]
    return "\n\n".join(f"Question: {r['question']}\nAnswer: {r['answer']}" for r in rows)


def _ultrachat_text(n_chars=3_000_000):
    path = _local_or_download("HuggingFaceH4/ultrachat_200k", repo_type="dataset",
                              allow_patterns=["data/test_sft-*.parquet"])
    df = pd.read_parquet(next(Path(path).rglob("test_sft-*.parquet")))
    parts, size = [], 0
    for conv in df.messages:
        for msg in conv:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
            size += len(parts[-1])
        if size > n_chars:
            break
    return "\n".join(parts)


def _chat_text(tok):
    cases = [json.loads(l) for l in (REPO / "bench/cases/tools.jsonl").open(encoding="utf-8")]
    parts = []
    for c in cases:
        for msg in c["messages"]:
            body = msg.get("content") or json.dumps(msg.get("tool_calls", ""))
            parts.append(f"<|im_start|>{msg['role']}\n{body}<|im_end|>")
    return "\n".join(parts)


def _copy_seqs(tok, T, n, rng):
    """Plant a code early in the sequence, ask for it near the end; return (ids, target mask)."""
    filler = tok(_wikitext("train"), add_special_tokens=False).input_ids
    names = ["Alvarez", "Okonkwo", "Lindqvist", "Tanaka", "Moreau", "Kowalski", "Haddad", "Ferreira"]
    out = []
    for _ in range(n):
        name, code = rng.choice(names), str(rng.randint(100000, 999999))
        plant = tok(f" The access code for {name} is {code}.", add_special_tokens=False).input_ids
        query = tok(f" Recall: the access code for {name} is", add_special_tokens=False).input_ids
        answer = tok(f" {code}.", add_special_tokens=False).input_ids
        body = T - len(plant) - len(query) - len(answer)
        start = rng.randint(0, len(filler) - body - 1)
        pre = rng.randint(8, 64)
        ids = filler[start:start + pre] + plant + filler[start + pre:start + body] + query + answer
        mask = [0] * (T - len(answer)) + [1] * len(answer)
        out.append((ids[:T], mask[:T]))
    return out


def build_data(tok, T=512, n_fit=16, n_eval=4, seed=0, n_eval_copy=None, rich=False):
    """Return {'fit': [...], 'eval': [...]}, each a list of (domain, ids[T], focus_mask[T]).

    rich: also use MBPP for code and UltraChat for chat (needs a download); the tools cases alone give only 2 chat chunks.
    n_eval_copy: copy-task eval sequences (default n_eval); its tail stats need many more than the other domains.
    """
    rng = random.Random(seed)
    n = n_fit + n_eval
    n_eval_copy = n_eval if n_eval_copy is None else n_eval_copy
    seqs = {}
    texts = {"prose": _wikitext("test"), "code": _code_text(extra=rich), "math": _math_text(),
             "chat": _ultrachat_text() if rich else _chat_text(tok)}
    for d, text in texts.items():
        ids = tok(text, add_special_tokens=False).input_ids
        ch = _chunks(ids, T, n)
        seqs[d] = [(d, c, [1] * T) for c in ch]
    seqs["copy"] = [("copy", ids, m) for ids, m in _copy_seqs(tok, T, n_fit + n_eval_copy, rng)]
    split = {"fit": [], "eval": []}
    for d in DOMAINS:
        s = seqs[d]
        k = n_fit if d == "copy" else max(1, round(len(s) * n_fit / n))
        split["fit"] += s[:k]
        split["eval"] += s[k:]
    return split


def batches(seqs, bs):
    for i in range(0, len(seqs), bs):
        b = seqs[i:i + bs]
        yield ([s[0] for s in b], torch.tensor([s[1] for s in b]), torch.tensor([s[2] for s in b]))


# ---------- runner ----------
# Sites: 2*i = residual entering layer i, 2*i+1 = residual after layer i's mixer (before its MLP), 2*L = final residual.

class Runner:
    def __init__(self, model):
        self.model = model
        self.lm = model.model
        self.cfg = self.lm.config
        self.L = self.cfg.num_hidden_layers
        self.types = self.cfg.layer_types
        self.n_sites = 2 * self.L + 1

    def site_name(self, s):
        if s == self.n_sites - 1:
            return "final"
        i, half = divmod(s, 2)
        kind = "gdn" if self.types[i] == "linear_attention" else "attn"
        return f"L{i}.{'in' if half == 0 else 'mid'}({kind})"

    @torch.no_grad()
    def prepare(self, ids):
        ids = ids.to(self.model.device)
        emb = self.lm.embed_tokens(ids)
        pos = torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1).expand(4, ids.shape[0], -1)
        text_pos, pos3 = pos[0], pos[1:]
        mk = dict(config=self.cfg, inputs_embeds=emb, attention_mask=None, past_key_values=None, position_ids=text_pos)
        ctx = {
            "masks": {"full_attention": create_causal_mask(**mk), "linear_attention": create_recurrent_attention_mask(**mk)},
            "pos_emb": self.lm.rotary_emb(emb, pos3),
            "text_pos": text_pos,
        }
        return emb, ctx

    def mixer(self, i, h, ctx):
        layer = self.lm.layers[i]
        x = layer.input_layernorm(h)
        if self.types[i] == "linear_attention":
            return layer.linear_attn(hidden_states=x, cache_params=None, attention_mask=ctx["masks"]["linear_attention"])
        out, _ = layer.self_attn(hidden_states=x, attention_mask=ctx["masks"]["full_attention"],
                                 position_ids=ctx["text_pos"], past_key_values=None, position_embeddings=ctx["pos_emb"])
        return out

    def mlp(self, i, h):
        layer = self.lm.layers[i]
        return layer.mlp(layer.post_attention_layernorm(h))

    @torch.no_grad()
    def run(self, h, start, ctx, edit=None, mixer_edit=None, mlp_edit=None, record=None, positions=None, keep_first=True):
        """Run from site `start` (h is the state there) to the logits.

        edit: {site: fn(h) -> h}, applied when the run reaches that site (including `start`).
        mixer_edit: {layer: fn(h_in, update) -> update}.
        mlp_edit: {layer: fn(h_mid, update) -> update}, for the MLP half of the layer.
        record: set of sites whose (post-edit) states are returned.
        positions: sequence positions to compute logits for (the vocab projection dominates CPU cost).
        keep_first: never edit position 0. It is the attention sink, fits were made without it, and a student would keep its own.
        """
        k0 = 1 if keep_first else 0
        edit, mixer_edit, mlp_edit, rec = edit or {}, mixer_edit or {}, mlp_edit or {}, {}
        for s in range(start, self.n_sites):
            if s in edit:
                h = torch.cat([h[:, :k0], edit[s](h)[:, k0:]], 1)
            if record is not None and s in record:
                rec[s] = h
            if s == self.n_sites - 1:
                break
            i, half = divmod(s, 2)
            if half == 0:
                u = self.mixer(i, h, ctx)
                if i in mixer_edit:
                    u = torch.cat([u[:, :k0], mixer_edit[i](h, u)[:, k0:]], 1)
            else:
                u = self.mlp(i, h)
                if i in mlp_edit:
                    u = torch.cat([u[:, :k0], mlp_edit[i](h, u)[:, k0:]], 1)
            h = h + u
        if positions is not None:
            h = h[:, positions.to(h.device)]
        logits = self.model.lm_head(self.lm.norm(h))
        return logits, rec


def logprobs(logits):
    return F.log_softmax(logits.float(), dim=-1)


def kl_per_token(ref_lp, logits):
    """KL(ref || edited) per token, in nats."""
    lp = logprobs(logits)
    return (ref_lp.exp() * (ref_lp - lp)).sum(-1)


def stats(v):
    v = torch.as_tensor(v).float()
    if v.numel() == 0:
        return {"mean": float("nan"), "p95": float("nan"), "p99": float("nan"), "n": 0}
    return {"mean": v.mean().item(), "p95": v.quantile(0.95).item(), "p99": v.quantile(0.99).item(), "n": v.numel()}


class KLAccumulator:
    """Per-domain KL across eval batches; the copy domain counts only its answer tokens."""

    def __init__(self):
        self.parts = {}

    def add(self, kl, domains, focus):
        kl = kl.cpu()
        for j, d in enumerate(domains):
            v = kl[j][focus[j].bool()] if d == "copy" else kl[j]
            self.parts.setdefault(d, []).append(v)

    def report(self):
        out = {d: stats(torch.cat(v)) for d, v in self.parts.items()}
        out["all"] = stats(torch.cat([torch.cat(v) for v in self.parts.values()]))
        return out


class Session:
    """Model + data + cached clean eval batches (states at every site, reference log-probs at sampled positions)."""

    def __init__(self, model_name, T=512, n_fit=16, n_eval=4, bs=2, stride=4, seed=0, dtype=torch.float32,
                 n_eval_copy=None, rich=False):
        self.name = model_name
        self.tag = model_name.split("/")[-1]
        self.device = DEVICE
        self.model, self.tok = load_model(model_name, dtype=dtype)
        self.runner = Runner(self.model)
        self.data = build_data(self.tok, T=T, n_fit=n_fit, n_eval=n_eval, seed=seed, n_eval_copy=n_eval_copy, rich=rich)
        self.bs, self.stride, self.T = bs, stride, T
        self.work = WORK / self.tag
        self.results = RESULTS / self.tag
        self._eval = None

    def eval_batches(self):
        """Clean eval batches, built once and kept in CPU memory. Position 0 (attention sink) is never scored."""
        if self._eval is None:
            self._eval = []
            for doms, ids, focus in batches(self.data["eval"], self.bs):
                grid = set(range(1, self.T, self.stride)) | set(focus.nonzero()[:, 1].tolist())
                pos = torch.tensor(sorted(grid - {0}))
                emb, ctx = self.runner.prepare(ids)
                logits, states = self.runner.run(emb, 0, ctx, record=set(range(self.runner.n_sites)), positions=pos)
                self._eval.append(dict(doms=doms, ids=ids, focus=focus[:, pos], pos=pos, ctx=ctx,
                                       states={k: v.half().cpu() for k, v in states.items()}, ref_lp=logprobs(logits).cpu()))
        return self._eval

    def state(self, b, site):
        return b["states"][site].to(self.device).float()

    def kl(self, start, edit=None, mixer_edit=None, mlp_edit=None, keep_first=True):
        """Replace-and-continue: run every eval batch from `start` with the given edits; per-domain KL stats."""
        acc = KLAccumulator()
        for b in self.eval_batches():
            logits, _ = self.runner.run(self.state(b, start), start, b["ctx"], edit=edit, mixer_edit=mixer_edit,
                                        mlp_edit=mlp_edit, positions=b["pos"], keep_first=keep_first)
            acc.add(kl_per_token(b["ref_lp"].to(self.device), logits), b["doms"], b["focus"])
        return acc.report()

    def fit_states(self, site):
        """Clean fit-set states at one site, [N, d] float32 on the session device (first token of each sequence dropped)."""
        return torch.from_numpy(np.load(self.work / f"site{site:03d}.npy").astype(np.float32)).to(self.device)

    def fit_meta(self):
        return json.loads((self.work / "meta.json").read_text())


def add_session_args(ap, n_fit=16, n_eval=4, bs=4):
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--n-fit", type=int, default=n_fit)
    ap.add_argument("--n-eval", type=int, default=n_eval)
    ap.add_argument("--n-eval-copy", type=int, default=None)
    ap.add_argument("--rich", action="store_true", help="MBPP + UltraChat in the data mix")
    ap.add_argument("--stride", type=int, default=4, help="score every Nth position (plus all copy answers)")
    ap.add_argument("--bs", type=int, default=bs)
    ap.add_argument("--threads", type=int, default=12)


def session_from(args):
    set_threads(args.threads)
    return Session(args.model, n_fit=args.n_fit, n_eval=args.n_eval, bs=args.bs, stride=args.stride,
                   n_eval_copy=args.n_eval_copy, rich=args.rich)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))


def set_threads(n=12):
    torch.set_num_threads(n)


def yield_if_asked():
    """Under `dibs run --preemptible --requeue`, step aside when another session queues; dibs reruns us after."""
    f = os.environ.get("DIBS_YIELD_FILE")
    if f and Path(f).exists():
        print("dibs asked us to yield; exiting 75 to be requeued", flush=True)
        sys.exit(75)
