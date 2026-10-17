import argparse
import math

import torch

from common import (
    GROUP, PARENT_DIR, PRISM_DIR, RESULTS, Weights, absmean_scale, decode_ternary, dequant,
    groups, lsq_scale, mse_opt, run_info, rtn, setup, tensor_kind, write_json,
)

BINS = torch.linspace(-2, 2, 81)
ABS_BINS = torch.linspace(0, 2, 21)
CHUNK = 8192


def hist(x, edges):
    idx = torch.bucketize(x.clamp(edges[0], edges[-1] - 1e-6), edges, right=True) - 1
    return torch.bincount(idx.flatten(), minlength=len(edges) - 1)[: len(edges) - 1]


def quantiles(v):
    v = torch.cat(v).float()
    v = v[torch.isfinite(v)]
    if v.numel() == 0:
        return None
    q = torch.quantile(v[torch.randperm(v.numel())[:1_000_000]], torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return [round(x, 4) for x in q.tolist()]


def analyze(W, P16):
    t_all, s_all, exact = decode_ternary(P16)
    flat = W.abs().flatten()
    sample = flat[torch.randint(0, flat.numel(), (min(flat.numel(), 2_000_000),))]
    top20 = torch.quantile(sample.float(), 0.8).item()
    st = {
        "shape": list(W.shape), "n": W.numel(), "groups": s_all.numel(), "exact_groups": exact,
        "zero": {"prism": 0, "mse": 0, "rtn_absmean": 0, "rtn_prism_s": 0},
        "agree": {"rtn_prism_s": 0, "rtn_absmean": 0, "mse": 0},
        "top20_sign": [0, 0], "confusion_rtn_prism_s_vs_prism": torch.zeros(3, 3, dtype=torch.long),
        "x_hist_by_prism_code": torch.zeros(3, len(BINS) - 1, dtype=torch.long),
        "abs_x_total": torch.zeros(len(ABS_BINS) - 1, dtype=torch.long),
        "abs_x_agree": torch.zeros(len(ABS_BINS) - 1, dtype=torch.long),
        "sq": {"w": 0.0, "prism": 0.0, "mse": 0.0, "rtn_prism_s": 0.0, "rtn_absmean": 0.0, "prism_codes_lsq_s": 0.0},
    }
    ratios = {"prism_over_absmean": [], "prism_over_lsq": [], "prism_over_mse": []}
    for r0 in range(0, W.shape[0], CHUNK):
        w = W[r0:r0 + CHUNK].float()
        t, s = t_all[r0:r0 + CHUNK], s_all[r0:r0 + CHUNK]
        s_am = absmean_scale(w)
        t_ps, t_am = rtn(w, s), rtn(w, s_am)
        t_mse, s_mse = mse_opt(w)
        s_lsq = lsq_scale(w, t)
        for key, tt in (("prism", t), ("mse", t_mse), ("rtn_absmean", t_am), ("rtn_prism_s", t_ps)):
            st["zero"][key] += (tt == 0).sum().item()
        for key, tt in (("rtn_prism_s", t_ps), ("rtn_absmean", t_am), ("mse", t_mse)):
            st["agree"][key] += (tt == t).sum().item()
        big = (w.abs() >= top20) & (t != 0)
        st["top20_sign"][0] += ((torch.sign(w) == t) & big).sum().item()
        st["top20_sign"][1] += big.sum().item()
        st["confusion_rtn_prism_s_vs_prism"] += torch.bincount(
            ((t_ps.long() + 1) * 3 + (t.long() + 1)).flatten(), minlength=9).reshape(3, 3)
        x = (groups(w) / s.clamp_min(1e-30)[..., None]).reshape(w.shape)
        for c in (-1, 0, 1):
            st["x_hist_by_prism_code"][c + 1] += hist(x[t == c], BINS)
        st["abs_x_total"] += hist(x.abs(), ABS_BINS)
        st["abs_x_agree"] += hist(x.abs()[t_ps == t], ABS_BINS)
        st["sq"]["w"] += (w * w).sum().item()
        for key, Wq in (("prism", dequant(t, s)), ("mse", dequant(t_mse, s_mse)), ("rtn_prism_s", dequant(t_ps, s)),
                        ("rtn_absmean", dequant(t_am, s_am)), ("prism_codes_lsq_s", dequant(t, s_lsq))):
            st["sq"][key] += ((w - Wq) ** 2).sum().item()
        step = max(1, s.numel() // 4096)
        for key, den in (("prism_over_absmean", s_am), ("prism_over_lsq", s_lsq), ("prism_over_mse", s_mse)):
            ratios[key].append((s / den).flatten()[::step])
    st["rel_weight_err"] = {k: math.sqrt(v / st["sq"]["w"]) for k, v in st["sq"].items() if k != "w"}
    st["scale_ratio_quantiles"] = {k: quantiles(v) for k, v in ratios.items()}
    for k, v in list(st.items()):
        if torch.is_tensor(v):
            st[k] = v.tolist()
    return st


def norm_check(name, parent, prism):
    same = parent.to(prism.dtype)
    d = (prism.float() - parent.float()).abs()
    return {
        "bit_identical_after_cast": bool(torch.equal(same, prism)),
        "max_abs_diff": d.max().item(),
        "rel_l2_diff": (d.norm() / parent.float().norm().clamp_min(1e-30)).item(),
        "frac_elements_changed": (same != prism).float().mean().item(),
    }


def equivalence_check(gamma_parent, gamma_prism, W_list, P_list):
    r = (gamma_prism.float() / gamma_parent.float()).clamp_min(1e-12)
    W = torch.cat([w.float() for w in W_list])
    P = torch.cat([p.float() for p in P_list])
    c = (W * P).sum(0) / (W * W).sum(0).clamp_min(1e-30)
    ok = (c > 0) & torch.isfinite(c)
    a, b = r[ok].log(), c[ok].log()
    a, b = a - a.mean(), b - b.mean()
    return {"corr_log_ratio_vs_log_colscale": (a @ b / (a.norm() * b.norm()).clamp_min(1e-30)).item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    setup(args.threads)
    parent, prism = Weights(PARENT_DIR), Weights(PRISM_DIR)
    out = {"run": run_info(args.threads), "tensors": {}, "norms": {}, "equivalence": {}, "missing_in_parent": []}
    for name in prism.names():
        kind, layer = tensor_kind(name)
        pname = name if name in parent.where else "model.embed_tokens.weight" if kind == "lm_head" else None
        if pname is None:
            out["missing_in_parent"].append(name)
            continue
        P, W = prism.get(name), parent.get(pname)
        if P.shape != W.shape:
            out["tensors"][name] = {"error": f"shape {list(P.shape)} vs parent {list(W.shape)}"}
            continue
        if P.dim() == 2 and P.shape[1] % GROUP == 0:
            _, _, exact = decode_ternary(P)
            if exact > 0.99:
                st = analyze(W, P)
                st.update(kind=kind, layer=layer, parent_name=pname)
                out["tensors"][name] = st
                print(f"{name}: agree_rtn_prism_s={st['agree']['rtn_prism_s'] / st['n']:.4f} "
                      f"zero={st['zero']['prism'] / st['n']:.4f}", flush=True)
                continue
            out["tensors"][name] = {"kind": kind, "layer": layer, "ternary": False, "exact_groups": exact}
        out["norms"][name] = norm_check(name, W, P)
        print(f"{name}: non-ternary identical={out['norms'][name]['bit_identical_after_cast']}", flush=True)
    for i in range(1 + max(t.get("layer", -1) for t in out["tensors"].values())):
        pre = f"model.layers.{i}."
        for norm, lins in (("input_layernorm", ("q_proj", "k_proj", "v_proj")),
                           ("post_attention_layernorm", ("gate_proj", "up_proj"))):
            key = f"{pre}{norm}.weight"
            if key in out["norms"] and not out["norms"][key]["bit_identical_after_cast"]:
                names = [f"{pre}{'self_attn' if l[0] in 'qkv' else 'mlp'}.{l}.weight" for l in lins]
                out["equivalence"][key] = equivalence_check(
                    parent.get(key), prism.get(key), [parent.get(n) for n in names],
                    [prism.get(n) for n in names])
    write_json(RESULTS / "fingerprint.json", out)


if __name__ == "__main__":
    main()
