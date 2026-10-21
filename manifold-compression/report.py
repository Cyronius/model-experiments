# Turn results/<model>/*.json into one markdown summary (2026-10-17).
import argparse
import json
import statistics

from common import RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
args = ap.parse_args()
tag = args.model.split("/")[-1]
R = RESULTS / tag
out = [f"# {tag}: phase results\n"]


def load(name):
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def m(rep, dom="all", key="mean"):
    return f"{rep[dom][key]:.3f}" if rep and dom in rep else "–"


w = load("width")
if w:
    names = w["site_names"]
    out.append("## Width (M1 + M2)\n")
    out.append("KL in nats, mean over tokens. Copy = the planted-code answer tokens only.\n")
    out.append("| k | variance kept (median site) | all sites, PCA | copy | all sites, shared basis | all sites, PCA + MLP | copy | median MLP R² of cut part |")
    out.append("|---|---|---|---|---|---|---|---|")
    for k in map(str, w["ks"]):
        energy = statistics.median(v[k] for v in w["spectrum"].values())
        r2 = statistics.median(w["mlp_val_r2"][k].values()) if k in w["mlp_val_r2"] else float("nan")
        out.append(f"| {k} | {energy:.2f} | {m(w['all_sites'].get(k))} | {m(w['all_sites'].get(k), 'copy')} | "
                   f"{m(w['all_shared'].get(k))} | {m(w['mlp_all'].get(k))} | {m(w['mlp_all'].get(k), 'copy')} | {r2:.3f} |")
    out.append("\nSingle-site projection (KL, PCA / PCA + MLP):\n")
    out.append("| site | " + " | ".join(f"k={k}" for k in w["ks"]) + " |")
    out.append("|---|" + "---|" * len(w["ks"]))
    for s, v in w["single"].items():
        cells = []
        for k in map(str, w["ks"]):
            lin = v[k]["kl"]["all"]["mean"]
            mlp = w["mlp_single"].get(s, {}).get(k)
            cells.append(f"{lin:.3f}" + (f" / {mlp['kl']['all']['mean']:.3f}" if mlp else ""))
        out.append(f"| {names[int(s)]} | " + " | ".join(cells) + " |")
    out.append("")

c = load("decoder_curve")
if c:
    out.append("## Decoder learning curve (M2, large collection)\n")
    out.append("Held-out R² of the part PCA cuts, recovered by an MLP from the kept k coordinates, vs fit tokens.\n")
    for site, by_k in c["curve"].items():
        for k, pts in by_k.items():
            out.append(f"- {site}, k={k}: " + ", ".join(f"{p['n'] // 1000}K: {p['val_r2']:.3f}" for p in pts))
    out.append("")

k_ = load("decoder_kl")
if k_:
    out.append("## Decoder payoff in KL (M2, large collection)\n")
    out.append("| site | k | KL, PCA | KL, PCA + decoder | reduction |")
    out.append("|---|---|---|---|---|")
    for site, by_k in k_["sites"].items():
        for k, v in by_k.items():
            a, b2 = v["pca"]["all"]["mean"], v["decoder"]["all"]["mean"]
            out.append(f"| {site} | {k} | {a:.4f} | {b2:.4f} | {100 * (a - b2) / a:.0f}% |")
    out.append("")

b = load("boundary")
if b:
    out.append("## Own-state test (M3)\n")
    out.append("Replace a mixing layer's update with a prediction from the token's own state (no context), vs dropping the update.\n")
    out.append("| layer | kind | R² linear | R² MLP | KL predict | copy p99 | KL drop | copy p99 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for i, x in b["layers"].items():
        kind = "attn" if x["kind"] == "full_attention" else "gdn"
        out.append(f"| {i} | {kind} | {x['r2_linear']:.2f} | {x['r2_mlp']:.2f} | {m(x['kl_predict'])} | "
                   f"{m(x['kl_predict'], 'copy', 'p99')} | {m(x['kl_drop'])} | {m(x['kl_drop'], 'copy', 'p99')} |")
    for key, title in [("greedy", "ranked by mean KL"), ("greedy_tail", "ranked by mean KL + copy-task worst 1%")]:
        if not b.get(key):
            continue
        out.append(f"\nGreedy substitution, {title} (cumulative):\n")
        out.append("| n | added | kind | KL all | copy mean | copy p99 |")
        out.append("|---|---|---|---|---|---|")
        for g in b[key]:
            kind = "attn" if g["kind"] == "full_attention" else "gdn"
            out.append(f"| {g['n_substituted']} | L{g['layer']} | {kind} | {m(g['kl'])} | {m(g['kl'], 'copy')} | {m(g['kl'], 'copy', 'p99')} |")
    out.append("")

p = load("perturb")
if p:
    out.append("## Perturbation (M4)\n")
    out.append("Relative deviation at the final site divided by relative deviation at injection. Below 1 = the stream outgrows the error.\n")
    keys = sorted({k for v in p["runs"].values() for k in v})
    out.append("| site | " + " | ".join(f"{k} ratio | {k} KL" for k in keys) + " |")
    out.append("|---|" + "---|---|" * len(keys))
    for s, v in p["runs"].items():
        out.append(f"| {p['site_names'][int(s)]} | " + " | ".join(
            f"{v[k]['rel_ratio']:.2f} | {v[k]['kl']['all']['mean']:.4f}" if k in v else "– | –" for k in keys) + " |")
    out.append("")

d = load("idim")
if d:
    out.append("## Intrinsic dimension (M5)\n")
    out.append("Levina-Bickel estimate on 3000 points; biased low at this sample size, so a loose lower bound only.\n")
    ks = list(map(str, d["ks"]))
    out.append("| site | " + " | ".join(f"k={k}" for k in ks) + " |")
    out.append("|---|" + "---|" * len(ks))
    for s, v in list(d["id"].items())[::4]:
        out.append(f"| {d['site_names'][int(s)]} | " + " | ".join(f"{v[k]:.1f}" for k in ks) + " |")

(R / "summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"wrote {R / 'summary.md'}")
