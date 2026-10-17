import json
import platform
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models" / "ternary"
WORK = MODELS / "work"
RESULTS = Path(__file__).resolve().parent / "results"

PARENT_REPO = "Qwen/Qwen3-1.7B"
PRISM_REPO = "prism-ml/Ternary-Bonsai-1.7B-unpacked"
PARENT_DIR = MODELS / "Qwen3-1.7B"
PRISM_DIR = MODELS / "Ternary-Bonsai-1.7B-unpacked"
WIKITEXT = ROOT / "bench" / "data" / "wikitext-2-raw"

GROUP = 128
LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CANDIDATES = torch.tensor([-1.0, 0.0, 1.0])


class Weights:
    def __init__(self, model_dir):
        self.where = {}
        for f in sorted(Path(model_dir).glob("*.safetensors")):
            with safe_open(str(f), framework="pt") as sf:
                for k in sf.keys():
                    self.where[k] = f

    def names(self):
        return list(self.where)

    def get(self, name):
        with safe_open(str(self.where[name]), framework="pt") as sf:
            # clone: mmap-backed tensors pin whole shards on Windows (commit exhaustion)
            return sf.get_tensor(name).clone()


def tensor_kind(name):
    parts = name.split(".")
    layer = int(parts[2]) if parts[:2] == ["model", "layers"] else -1
    for lin in LINEARS:
        if f".{lin}." in name:
            return lin, layer
    if "embed_tokens" in name:
        return "embed", layer
    if name.startswith("lm_head"):
        return "lm_head", layer
    return "other", layer


def setup(threads):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    RESULTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)


def run_info(threads, **extra):
    import numpy
    import transformers

    revs = MODELS / "revisions.json"
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "threads": threads,
        "revisions": json.loads(revs.read_text()) if revs.exists() else None,
        **extra,
    }


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1))


def groups(w, g=GROUP):
    rows, cols = w.shape
    return w.reshape(rows, cols // g, g)


def decode_ternary(w16, g=GROUP):
    a = groups(w16.float().abs(), g)
    s = a.amax(-1)
    nz = a > 0
    smallest = torch.where(nz, a, torch.full_like(a, float("inf"))).amin(-1)
    exact = (~nz.any(-1)) | (smallest == s)
    t = (groups(w16.float(), g) / s.clamp_min(1e-30)[..., None]).round().clamp(-1, 1)
    return t.to(torch.int8).reshape(w16.shape), s, exact.float().mean().item()


def dequant(t, s, g=GROUP):
    return (groups(t.float(), g) * s[..., None]).reshape(t.shape)


def rtn(w, s, g=GROUP):
    x = groups(w, g) / s.clamp_min(1e-30)[..., None]
    return x.round().clamp(-1, 1).to(torch.int8).reshape(w.shape)


def absmean_scale(w, g=GROUP):
    return groups(w, g).abs().mean(-1)


def lsq_scale(w, t, g=GROUP):
    tf = groups(t.float(), g)
    return (tf * groups(w, g)).sum(-1) / (tf * tf).sum(-1).clamp_min(1)


def mse_opt(w, g=GROUP, chunk=8192):
    ts, ss = [], []
    for r0 in range(0, w.shape[0], chunk):
        gw = groups(w[r0:r0 + chunk], g)
        a = gw.abs().sort(-1, descending=True).values
        cs = a.cumsum(-1)
        k = torch.arange(1, g + 1, dtype=cs.dtype)
        # for a fixed support the LS scale is the mean kept magnitude; best support is top-k
        best = (cs * cs / k).argmax(-1, keepdim=True)
        ss.append(cs.gather(-1, best).squeeze(-1) / (best.squeeze(-1) + 1).to(cs.dtype))
        kth = a.gather(-1, best)
        ts.append(torch.where(gw.abs() >= kth, gw.sign(), torch.zeros_like(gw)).to(torch.int8).flatten(1))
    return torch.cat(ts), torch.cat(ss)


def hinv_upper(H, damp=0.01):
    H = H.double().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    H += damp * torch.diag(H).mean() * torch.eye(H.shape[0], dtype=H.dtype)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    return torch.linalg.cholesky(Hinv, upper=True).float(), dead


def _block_search(Wb, s, Ub, k):
    R, B = Wb.shape
    w = Wb[:, None, :].clone()
    cost = torch.zeros(R, 1)
    back, pick, errs = [], [], []
    for i in range(B):
        e = (w[:, :, i, None] - s[:, None, None] * CANDIDATES) / Ub[i, i]
        c = (cost[:, :, None] + e * e).reshape(R, -1)
        c, sel = c.topk(min(k, c.shape[1]), dim=1, largest=False)
        parent = sel // 3
        e_sel = e.reshape(R, -1).gather(1, sel)
        w = w.gather(1, parent[:, :, None].expand(-1, -1, B))
        w[:, :, i + 1:] -= e_sel[:, :, None] * Ub[i, i + 1:]
        back.append(parent)
        pick.append(sel % 3 - 1)
        errs.append(e_sel)
        cost = c
    b = cost.argmin(1, keepdim=True)
    t = torch.empty(R, B, dtype=torch.int8)
    err = torch.empty(R, B)
    for i in range(B - 1, -1, -1):
        t[:, i] = pick[i].gather(1, b).squeeze(1).to(torch.int8)
        err[:, i] = errs[i].gather(1, b).squeeze(1)
        b = back[i].gather(1, b)
    return t, err


# k=1 is plain GPTQ (Babai nearest plane); k>1 keeps the k cheapest paths per row inside each group
def gptq_ternary(W, H, scales, k=1, g=GROUP):
    W = W.float().clone()
    U, dead = hinv_upper(H)
    W[:, dead] = 0
    T = torch.zeros(W.shape, dtype=torch.int8)
    for i1 in range(0, W.shape[1], g):
        i2 = i1 + g
        t, err = _block_search(W[:, i1:i2], scales[:, i1 // g], U[i1:i2, i1:i2], k)
        T[:, i1:i2] = t
        W[:, i2:] -= err @ U[i1:i2, i2:]
    return T


def out_error(W, What, H):
    D = W - What
    return ((D @ H) * D).sum().item() / ((W @ H) * W).sum().item()


def wikitext_ids(tokenizer, split):
    text = (WIKITEXT / f"wiki.{split}.raw").read_text(encoding="utf-8")
    return torch.tensor(tokenizer(text, add_special_tokens=False)["input_ids"])
