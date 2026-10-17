"""The five canonical figures, rendered from runs/*/final.json.

Color follows the job it does. Series identity (condition) uses a fixed
categorical order, never cycled; lambda is a magnitude, so lambda sweeps use a
single-hue sequential ramp; the utilization heatmap is sequential too. Three of
the categorical slots sit under 3:1 contrast on a white surface, so every series
is direct-labeled and a table view is written alongside the figures -- that is
the relief the palette validation requires, not an optional extra.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .data import CELLS

RUNS = Path(__file__).resolve().parent / "runs"
FIGS = Path(__file__).resolve().parent.parent / "figures"

# validated categorical order (light mode); see palette validation in README
C = {"mow_routed": "#2a78d6", "fixed": "#eb6834", "moe_adaptive": "#1baf7a",
     "moe_topk": "#eda100", "fine_bal": "#e87ba4", "fine_nobal": "#008300",
     "dense": "#4a3aa7"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
CELL_C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.dpi": 140, "savefig.bbox": "tight",
})


def _ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, loc="left", color=INK, pad=8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


# ---------------------------------------------------------------- loading

def load_runs(runs_dir=RUNS) -> list:
    out = []
    for d in sorted(Path(runs_dir).glob("*/final.json")):
        with open(d) as f:
            r = json.load(f)
        r["dir"] = d.parent.name
        out.append(r)
    return out


def agg(vals):
    """mean, min, max -- the plan asks for mean with range, not a std dev."""
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if not len(a):
        return None
    return {"mean": float(a.mean()), "lo": float(a.min()), "hi": float(a.max()),
            "n": len(a)}


def group(runs, key):
    g = defaultdict(list)
    for r in runs:
        g[key(r)].append(r)
    return g


def cond_key(r) -> str:
    """Run name with the seed suffix removed -- one key per condition."""
    return re.sub(r"_s\d+$", "", r.get("dir") or r["name"])


def series(runs, tag, xf, yf):
    """-> sorted [(x_agg, y_agg)] per condition, averaged across seeds.

    Grouping is by condition, not by x. Each seed lands at a slightly different
    FLOPs fraction, so grouping on x would split one condition's seeds into
    separate points and draw a zigzag instead of a frontier.
    """
    by_cond = defaultdict(list)
    for r in runs:
        if r["tag"] == tag:
            by_cond[cond_key(r)].append(r)
    pts = []
    for rs in by_cond.values():
        x, y = agg([xf(r) for r in rs]), agg([yf(r) for r in rs])
        if x is not None and y is not None:
            pts.append((x, y))
    return sorted(pts, key=lambda p: p[0]["mean"])


def _plot_series(ax, pts, color, label, marker="o", ls="-", label_pos="right",
                 direct=True):
    if not pts:
        return
    ax_ = lambda p: p["mean"] if isinstance(p, dict) else p
    x = [ax_(p[0]) for p in pts]
    y = [p[1]["mean"] for p in pts]
    lo = [max(0.0, p[1]["mean"] - p[1]["lo"]) for p in pts]   # clamp fp noise
    hi = [max(0.0, p[1]["hi"] - p[1]["mean"]) for p in pts]
    xerr = None
    if pts and isinstance(pts[0][0], dict):
        xerr = [[max(0.0, p[0]["mean"] - p[0]["lo"]) for p in pts],
                [max(0.0, p[0]["hi"] - p[0]["mean"]) for p in pts]]
    ax.errorbar(x, y, yerr=[lo, hi], xerr=xerr, color=color, lw=2, marker=marker,
                ms=7, mec="white", mew=1.2, capsize=0, elinewidth=1.2, alpha=0.95,
                label=label, ls=ls, zorder=3)
    if not direct:
        return
    right = label_pos == "right"
    i = -1 if right else 0
    ax.annotate(label, (x[i], y[i]), textcoords="offset points",
                xytext=(9 if right else -9, 5), ha="left" if right else "right",
                color=color, fontsize=8.5, fontweight="bold", zorder=4)


# ---------------------------------------------------------------- figure 1

def _d_ff(runs, default=512):
    for r in runs:
        if r.get("d_ff"):
            return int(r["d_ff"])
    return default


def fig1_nesting_tax(runs):
    """Loss at each fixed tier of the nested model vs separately-trained dense
    models of the same width. The gap is the price of sharing one weight matrix."""
    dff = _d_ff(runs)
    dense = {}
    for r in [r for r in runs if r["tag"] == "dense"]:
        h = int(round(r["mlp_frac"] * dff))
        dense.setdefault(h, []).append(r["loss"])
    nested = defaultdict(list)
    nested_acc, dense_acc = defaultdict(list), defaultdict(list)
    # per-seed, so the tax can be a paired difference. dense and nested co-vary
    # strongly by seed, so the paired sd is roughly half the individual sds --
    # independent error bars would describe a comparison nobody is making.
    dense_by_seed, nested_by_seed = defaultdict(dict), defaultdict(dict)
    for r in [r for r in runs if r["tag"] == "dense"]:
        dense_by_seed[int(round(r["mlp_frac"] * dff))][r["seed"]] = r["loss"]
    # prefer the full-budget nesting pretrain; it is step-matched to the dense
    # baselines, so the gap is the cost of nesting rather than of fewer steps
    ntag = "mow_pretrain_full" if any(r["tag"] == "mow_pretrain_full" for r in runs)         else "mow_pretrain"
    for r in [r for r in runs if r["tag"] == ntag]:
        for h, v in r.get("fixed_tiers", {}).items():
            nested[int(h)].append(v["loss"])
            nested_acc[int(h)].append(v.get("acc"))
            nested_by_seed[int(h)][r["seed"]] = v["loss"]
    for r in [r for r in runs if r["tag"] == "dense"]:
        dense_acc[int(round(r["mlp_frac"] * dff))].append(r.get("acc"))
    if not dense or not nested:
        return None

    widths = sorted(set(dense) & set(nested))
    d = [agg(dense[w]) for w in widths]
    n = [agg(nested[w]) for w in widths]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.9),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    _ax(a1, "Shared nested weights vs separate dense models",
        "MLP hidden width", "answer-token loss (nats)")
    a1.fill_between(widths, [x["mean"] for x in d], [x["mean"] for x in n],
                    color=C["fixed"], alpha=0.10, zorder=1)
    _plot_series(a1, list(zip(widths, d)), C["dense"], "dense baseline", "s",
                 label_pos="left")
    _plot_series(a1, list(zip(widths, n)), C["fixed"], "nested tier", "o")
    a1.set_xscale("log", base=2)          # widths double each step
    a1.set_xticks(widths)
    a1.set_xticklabels([str(w) for w in widths])
    a1.minorticks_off()
    a1.set_xlim(widths[0] / 1.7, widths[-1] * 2.1)

    def paired(w):
        seeds = sorted(set(dense_by_seed[w]) & set(nested_by_seed[w]))
        diffs = [nested_by_seed[w][k] - dense_by_seed[w][k] for k in seeds]
        m = sum(diffs) / len(diffs)
        if len(diffs) < 2:
            return m, 0.0, len(diffs)
        sd = (sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1)) ** 0.5
        return m, 1.96 * sd / len(diffs) ** 0.5, len(diffs)

    paired_stats = [paired(w) for w in widths]
    tax = [t[0] for t in paired_stats]
    tax_ci = [t[1] for t in paired_stats]
    n_seeds = paired_stats[0][2] if paired_stats else 0
    _ax(a2, f"Cost per tier (nested − dense), paired by seed (n={n_seeds})",
        "MLP hidden width", "Δ loss (nats)")
    bars = a2.bar([str(w) for w in widths], tax, color=C["fixed"], width=0.55,
                  zorder=3, alpha=0.9,
                  yerr=tax_ci, capsize=4,
                  error_kw=dict(ecolor=INK, lw=1.2, zorder=4))
    a2.axhline(0, color=INK2, lw=1)
    for b, t in zip(bars, tax):
        a2.annotate(f"{t:+.3f}", (b.get_x() + b.get_width() / 2, t),
                    ha="center", va="bottom" if t >= 0 else "top",
                    xytext=(0, 3 if t >= 0 else -3), textcoords="offset points",
                    fontsize=8.5, color=INK, fontweight="bold")
    m = max(abs(t) + c for t, c in zip(tax, tax_ci)) * 1.45 + 1e-6
    a2.set_ylim(min(-m * 0.3, min(tax) * 1.5), m)
    fig.suptitle("Figure 1 — Nesting tax", x=0.005, ha="left", fontsize=12,
                 fontweight="bold", y=1.04)
    fig.savefig(FIGS / "fig1_nesting_tax.png")
    plt.close(fig)
    return {"widths": widths, "dense": d, "nested": n, "tax": tax,
            "tax_ci95": tax_ci, "tax_n_seeds": n_seeds,
            "dense_acc": {w: agg(dense_acc[w]) for w in widths},
            "nested_acc": {w: agg(nested_acc[w]) for w in widths}}


# ---------------------------------------------------------------- figure 2

def fig2_frontier(runs, f1):
    """The headline: answer-token loss vs mean per-token MLP FLOPs fraction.

    The dashed line interpolates the fixed nested tiers. Routed MoW has to sit
    below it to have bought anything -- otherwise the router is just picking
    widths that a fixed width already reaches at the same cost.
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    _ax(ax, "", f"mean per-token MLP FLOPs (fraction of the dense {_d_ff(runs)} MLP)",
        "answer-token loss (nats)")

    # The fixed-tier curve IS the interpolation reference. Only it and routed MoW
    # are direct-labeled -- those two are what the falsification test compares --
    # and the legend plus figures/results_table.md cover the rest.
    fixed = None
    if f1:
        dff = _d_ff(runs)
        fixed = [(w / dff, a) for w, a in zip(f1["widths"], f1["nested"])]
        _plot_series(ax, fixed, C["fixed"], "fixed tier (no router)", "o")

    for tag, color, marker, label, direct in [
        ("moe_adaptive", C["moe_adaptive"], "^", "adaptive-k MoE (λ sweep)", False),
        ("moe_fine_bal", C["fine_bal"], "v", "fine MoE, balance on", False),
        ("moe_fine_nobal", C["fine_nobal"], "P", "fine MoE, balance off", False),
        ("moe_topk", C["moe_topk"], "s", "MoE top-k", False),
        ("mow_router", C["mow_routed"], "D", "routed MoW (λ sweep)", True),
    ]:
        pts = series(runs, tag, lambda r: r["mlp_frac"], lambda r: r["loss"])
        _plot_series(ax, pts, color, label, marker, direct=direct)

    dense_pts = series(runs, "dense", lambda r: r["mlp_frac"], lambda r: r["loss"])
    if dense_pts:
        ax.plot([p[0]["mean"] for p in dense_pts], [p[1]["mean"] for p in dense_pts],
                color=C["dense"], lw=1.2, ls=":", marker="s", ms=5, alpha=0.6,
                zorder=2, label="dense baselines")

    ax.set_xlim(0, 1.30)
    ax.legend(loc="upper right", fontsize=8.2, ncol=1, labelspacing=0.45)
    # shade the win region last, once the data has settled the y range
    if fixed:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi)
        ax.fill_between([p[0] for p in fixed], [p[1]["mean"] for p in fixed], lo,
                        color=C["fixed"], alpha=0.06, zorder=0, lw=0)
        ax.text(0.985, 0.02, "shaded = beats fixed widths at equal compute",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color=C["fixed"], zorder=5)
    fig.suptitle("Figure 2 — Compute-quality frontier", x=0.005, ha="left",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.savefig(FIGS / "fig2_frontier.png")
    plt.close(fig)
    return fixed


# ---------------------------------------------------------------- figure 3

def fig3_interpretability(runs):
    """Does the MoW router spend more compute on harder inputs? lambda is a
    magnitude, so the sweep gets a sequential ramp rather than categorical hues."""
    sel = [r for r in runs if r["tag"] == "mow_router"]
    if not sel:
        return None
    lams = sorted({r["lambda_flops"] for r in sel})
    ramp = plt.cm.Blues(np.linspace(0.42, 0.95, len(lams)))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), sharey=True)
    panels = [("arithmetic", ("arith_easy", "arith_hard"), "operand digits"),
              ("brackets", ("dyck_easy", "dyck_hard"), "nesting depth"),
              ("recall", ("recall_easy", "recall_hard"), "log2(table size)")]
    slopes = {}
    for ax, (title, cells, xlabel) in zip(axes, panels):
        _ax(ax, f"{title}: compute spent vs difficulty", xlabel,
            "mean MLP FLOPs fraction")
        seen = set()
        for li, lam in enumerate(lams):
            byd = defaultdict(list)
            for r in [r for r in sel if r["lambda_flops"] == lam]:
                for c in cells:
                    for d, v in r["per_cell"][c].get("by_difficulty", {}).items():
                        byd[int(d)].append(v["mlp_frac"])
            if not byd:
                continue
            xs = sorted(byd)
            seen.update(xs)
            ys = [float(np.mean(byd[d])) for d in xs]
            ax.plot(xs, ys, color=ramp[li], lw=2, marker="o", ms=6.5,
                    mec="white", mew=1.1, zorder=3, label=f"λ={lam:g}")
            if len(xs) > 1:
                slopes.setdefault(title, {})[lam] = float(
                    np.polyfit(xs, ys, 1)[0])
        if seen:
            ax.set_xticks(sorted(seen))
            ax.set_xlim(min(seen) - 0.4, max(seen) + 1.3)
    axes[0].set_ylabel("mean MLP FLOPs fraction")
    axes[-1].legend(title="FLOP penalty", fontsize=7.5, title_fontsize=8,
                    loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Figure 3 — Router interpretability: tier choice vs difficulty",
                 x=0.005, ha="left", fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(FIGS / "fig3_router_interpretability.png")
    plt.close(fig)
    return slopes


# ---------------------------------------------------------------- figure 4

def fig4_specialization(runs):
    """MoE should split on task type; MoW should split on difficulty. Utilization
    is a magnitude -> sequential single hue; the tier histogram is identity ->
    categorical."""
    moe = [r for r in runs if r["tag"] == "moe_topk"]
    mow = [r for r in runs if r["tag"] == "mow_router"]
    if not moe or not mow:
        return None
    moe = [r for r in moe if r["mlp_frac"] < 0.4] or moe          # prefer top-1

    # MoE: expert utilization per cell, summed over layers
    E = len(moe[0]["per_cell"][CELLS[0]]["choice_hist"][0])
    util = np.zeros((len(CELLS), E))
    for r in moe:
        for i, c in enumerate(CELLS):
            h = np.array(r["per_cell"][c]["choice_hist"], dtype=float).sum(0)
            util[i] += h / max(h.sum(), 1)
    util /= len(moe)

    # MoW: tier histogram per cell, at the most adaptive lambda
    best = max(mow, key=lambda r: r.get("entropy") or 0)
    lam = best["lambda_flops"]
    sel = [r for r in mow if r["lambda_flops"] == lam]
    T = len(sel[0]["per_cell"][CELLS[0]]["choice_hist"][0])
    tier = np.zeros((len(CELLS), T))
    for r in sel:
        for i, c in enumerate(CELLS):
            h = np.array(r["per_cell"][c]["choice_hist"], dtype=float).sum(0)
            tier[i] += h / max(h.sum(), 1)
    tier /= len(sel)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 4.2),
                                 gridspec_kw={"width_ratios": [1.05, 1]})
    im = a1.imshow(util, cmap="Blues", vmin=0, vmax=max(util.max(), 1e-6),
                   aspect="auto")
    _ax(a1, "MoE top-1: expert utilization by data cell", "expert", "")
    a1.grid(False)
    a1.set_xticks(range(E)); a1.set_yticks(range(len(CELLS)))
    a1.set_yticklabels(CELLS)
    for i in range(len(CELLS)):
        for j in range(E):
            a1.text(j, i, f"{util[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if util[i, j] > util.max() * 0.55 else INK)
    fig.colorbar(im, ax=a1, fraction=0.035, pad=0.02, label="share of answer tokens")

    _ax(a2, f"Routed MoW (λ={lam:g}): tier choice by data cell", "tier (hidden width)",
        "share of answer tokens")
    n = len(CELLS)
    w = 0.8 / n
    xs = np.arange(T)
    for i, c in enumerate(CELLS):
        a2.bar(xs + (i - (n - 1) / 2) * w, tier[i], width=w * 0.9, label=c,
               color=CELL_C[i % len(CELL_C)], zorder=3, alpha=0.92)
    a2.set_xticks(xs)
    tiers = next((r["tiers"] for r in sel if r.get("tiers")), list(range(T)))
    a2.set_xticklabels([str(h) for h in tiers][:T])
    a2.legend(fontsize=7.5, ncol=3, loc="upper center")
    a2.set_ylim(0, min(1.0, tier.max() * 1.45))
    fig.suptitle("Figure 4 — Specialization (MoE, by type) vs prioritization (MoW, by difficulty)",
                 x=0.005, ha="left", fontsize=12, fontweight="bold", y=1.04)
    fig.savefig(FIGS / "fig4_specialization_vs_prioritization.png")
    plt.close(fig)
    return {"moe_util": util.tolist(), "mow_tier": tier.tolist(), "mow_lambda": lam}


# ---------------------------------------------------------------- figure 5

def fig5_emergent(runs):
    """Fine-grained adaptive-k MoE: does an always-on expert core appear when the
    balance loss is removed? That core would be nesting discovered for free."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    _ax(a1, "Per-expert activation frequency (sorted)", "expert rank",
        "share of answer tokens")
    _ax(a2, "Chosen k distribution", "k (experts activated)", "share of answer tokens")

    found = False
    for tag, color, label in [("moe_fine_bal", C["fine_bal"], "balance loss on"),
                              ("moe_fine_nobal", C["fine_nobal"], "balance loss off")]:
        sel = [r for r in runs if r["tag"] == tag]
        if not sel:
            continue
        found = True
        curves, khists = [], []
        for r in sel:
            h = np.zeros(len(r["per_cell"][CELLS[0]]["choice_hist"][0]))
            k = np.zeros(len(r["per_cell"][CELLS[0]]["k_hist"][0]))
            for c in CELLS:
                h += np.array(r["per_cell"][c]["choice_hist"], dtype=float).sum(0)
                k += np.array(r["per_cell"][c]["k_hist"], dtype=float).sum(0)
            curves.append(np.sort(h / max(h.sum(), 1))[::-1])
            khists.append(k / max(k.sum(), 1))
        m = np.mean(curves, 0)
        a1.plot(range(1, len(m) + 1), m, color=color, lw=2, marker="o", ms=5,
                mec="white", mew=1, zorder=3)
        a1.fill_between(range(1, len(m) + 1), np.min(curves, 0), np.max(curves, 0),
                        color=color, alpha=0.13, zorder=2)
        a1.annotate(label, (1, m[0]), textcoords="offset points", xytext=(8, 2),
                    color=color, fontsize=8.5, fontweight="bold")
        km = np.mean(khists, 0)
        ks = [1, 2, 4, 8, 16][:len(km)]
        off = -0.18 if tag == "moe_fine_bal" else 0.18
        a2.bar(np.arange(len(km)) + off, km, width=0.34, color=color, zorder=3,
               alpha=0.92, label=label)
        a2.set_xticks(range(len(km))); a2.set_xticklabels([str(k) for k in ks])
    if not found:
        plt.close(fig)
        return None
    a1.axhline(1 / 16, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    a1.annotate("uniform (1/16)", (11, 1 / 16), textcoords="offset points",
                xytext=(0, 5), color=INK2, fontsize=8)
    a2.legend(fontsize=8)
    fig.suptitle("Figure 5 — Emergent prioritization in fine-grained adaptive-k MoE",
                 x=0.005, ha="left", fontsize=12, fontweight="bold", y=1.04)
    fig.savefig(FIGS / "fig5_emergent_prioritization.png")
    plt.close(fig)
    return True


# ---------------------------------------------------------------- table view

def results_table(runs) -> str:
    """The table view. Required relief for the low-contrast palette slots, and
    the numbers behind every figure."""
    rows = defaultdict(list)
    for r in runs:
        rows[(r["tag"], r["lambda_flops"])].append(r)
    lines = ["| condition | λ | seeds | MLP FLOPs frac | loss | exact-match | router entropy |",
             "|---|---|---|---|---|---|---|"]
    for (tag, lam), rs in sorted(rows.items()):
        f = agg([r["mlp_frac"] for r in rs])
        l = agg([r["loss"] for r in rs])
        a = agg([r.get("acc") for r in rs])
        e = agg([r.get("entropy") for r in rs])
        fmt = lambda x, p=3: "—" if x is None else \
            (f"{x['mean']:.{p}f}" if x["n"] < 2 else
             f"{x['mean']:.{p}f} ({x['lo']:.{p}f}–{x['hi']:.{p}f})")
        lines.append(f"| {tag} | {lam:g} | {len(rs)} | {fmt(f)} | {fmt(l, 4)} | "
                     f"{fmt(a)} | {fmt(e)} |")
    return "\n".join(lines)


def main():
    try:                       # the table carries a literal lambda; cp1252 chokes
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    FIGS.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        print("no completed runs found in", RUNS)
        return
    print(f"loaded {len(runs)} runs")
    f1 = fig1_nesting_tax(runs)
    fig2_frontier(runs, f1)
    slopes = fig3_interpretability(runs)
    f4 = fig4_specialization(runs)
    fig5_emergent(runs)
    table = results_table(runs)
    (FIGS / "results_table.md").write_text(table + "\n", encoding="utf-8")
    summary = {"n_runs": len(runs), "fig3_slopes": slopes,
               "fig4": f4, "fig1_tax": f1["tax"] if f1 else None,
               "fig1_widths": f1["widths"] if f1 else None,
               "fig1_tax_ci95": f1["tax_ci95"] if f1 else None,
               "fig1_tax_n_seeds": f1["tax_n_seeds"] if f1 else None}
    (FIGS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(table)
    print(f"\nfigures written to {FIGS}")


if __name__ == "__main__":
    main()
