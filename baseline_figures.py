#!/usr/bin/env python3
"""
baseline_figures.py — turn the experiment1 baseline CSVs into paper assets:
  1. Table VI as F1 (LaTeX)                  -> <out>_table_f1.tex
  2. Delta table (Ours - best baseline)      -> <out>_table_delta.tex
  3. Diverging Delta-F1 heatmap              -> <out>_heatmap.png / .pdf

Consumes the CSVs written by experiment1_baselines.py (columns:
Method, grad-p, grad-p_f1, spf, spf_f1, lfp, lfp_f1, ooa, ooa_f1, No Attack).

Usage:
  python3 baseline_figures.py --csv NYC-Taxi=results/nyc-taxi/nyc-taxi_table_baselines.csv \
      Foursquare=results/foursquare/foursquare_table_baselines.csv \
      Yelp=results/yelp/yelp_table_baselines.csv \
      Geolife=results/geolife/geolife_table_baselines.csv --out fig/baselines

Deps: numpy, pandas, matplotlib.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# attack f1 column -> display name (order = table/heatmap column order)
ATTACKS = [("grad-p_f1", "Grad-poison", "Grad-\npoison"),
           ("spf_f1",    "GPS-spoof",   "GPS-\nspoof"),
           ("lfp_f1",    "Label-flip",  "Label-\nflip"),
           ("ooa_f1",    "On-off",      "On-off")]
SEMANTIC = {"spf_f1", "lfp_f1"}                       # the kappa layer's target
BASELINES = ["FedAvg", "Multi-Krum", "TrimmedMean", "Bulyan", "RFVIR", "FLAME"]
METHODS   = BASELINES + ["Ours"]


def load(pairs):
    data = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--csv expects label=path, got '{p}'")
        label, path = p.split("=", 1)
        data[label] = pd.read_csv(path).set_index("Method")
    return data


def table_f1(data, out):
    def fmt(v, best):
        return f"\\textbf{{{v:.1f}}}" if best else f"{v:.1f}"
    L = [r"\begin{table*}[t]\centering\small",
         r"\caption{Byzantine robustness by \textbf{F1-score (\%)} under 20\% Byzantine "
         r"clients on real data (best per attack in \textbf{bold}). Accuracy is omitted "
         r"because majority-class inflation makes it uninformative on the imbalanced "
         r"datasets.}", r"\label{tab:baselines_f1}",
         r"\begin{tabular}{ll" + "c" * len(METHODS) + "}", r"\toprule",
         "Dataset & Attack & " + " & ".join(METHODS) + r" \\", r"\midrule"]
    for di, (dname, df) in enumerate(data.items()):
        for i, (col, aname, _) in enumerate(ATTACKS):
            vals = {m: float(df.loc[m, col]) for m in METHODS}
            best = max(vals.values())
            cells = [fmt(vals[m], best > 0 and abs(vals[m] - best) < 1e-9) for m in METHODS]
            L.append(f"{dname if i == 0 else ''} & {aname} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table*}"]
    open(f"{out}_table_f1.tex", "w").write("\n".join(L) + "\n")
    print(f"wrote {out}_table_f1.tex")


def deltas(data):
    D = np.zeros((len(data), len(ATTACKS)))
    for i, (dname, df) in enumerate(data.items()):
        for j, (col, _, _) in enumerate(ATTACKS):
            D[i, j] = float(df.loc["Ours", col]) - max(float(df.loc[b, col]) for b in BASELINES)
    print(f"delta F1 (Ours - best baseline) =\n{D}")
    return D


def table_delta(data, D, out):
    def cell(v):
        s = f"{v:+.1f}"
        return f"\\textbf{{{s}}}" if v > 0 else s
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{F1 gap of Ours vs.\ the \emph{best} baseline per cell "
         r"($\Delta$F1 $=$ Ours $-\max_{\text{baseline}}$), 20\% Byzantine, real data. "
         r"Positive (\textbf{bold}) $=$ Ours best.}", r"\label{tab:ours_delta}",
         r"\begin{tabular}{l" + "c" * len(ATTACKS) + "}", r"\toprule",
         "Dataset & " + " & ".join(a[1] for a in ATTACKS) + r" \\", r"\midrule"]
    for i, dname in enumerate(data):
        L.append(f"{dname} & " + " & ".join(cell(D[i, j]) for j in range(len(ATTACKS))) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{out}_table_delta.tex", "w").write("\n".join(L) + "\n")
    print(f"wrote {out}_table_delta.tex")


def heatmap(data, D, out, clip=20.0):
    names = list(data.keys())
    cmap = LinearSegmentedColormap.from_list("oib", ["#E69F00", "#f2f2f2", "#0072B2"])
    norm = TwoSlopeNorm(vmin=-clip, vcenter=0, vmax=clip)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    im = ax.imshow(D, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(ATTACKS))); ax.set_xticklabels([a[2] for a in ATTACKS], fontsize=11)
    ax.set_yticks(range(len(names)));   ax.set_yticklabels(names, fontsize=11)
    ax.set_xticks(np.arange(-.5, len(ATTACKS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    for i in range(len(names)):
        for j in range(len(ATTACKS)):
            v = D[i, j]; txt = "0.0" if abs(v) < 1e-9 else f"{v:+.1f}"
            col = "white" if abs(norm(v) - 0.5) > 0.32 else "#222222"
            ax.text(j, i, txt, ha="center", va="center", fontsize=12.5,
                    fontweight="bold" if v > 0 else "normal", color=col)
    # bracket over the contiguous semantic-attack columns
    sem = [j for j, a in enumerate(ATTACKS) if a[0] in SEMANTIC]
    if sem:
        lo, hi = min(sem) - 0.4, max(sem) + 0.4
        ax.annotate("", xy=(hi, -0.62), xytext=(lo, -0.62),
                    arrowprops=dict(arrowstyle="-", lw=1.5, color="#0072B2"), annotation_clip=False)
        ax.annotate("semantic attacks  ($\\kappa$ layer's target)", xy=((lo + hi) / 2, -0.80),
                    ha="center", va="center", fontsize=10, color="#0072B2", annotation_clip=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, extend="both")
    cb.set_label("$\\Delta$F1  =  Ours $-$ best baseline  (%)", fontsize=10)
    #ax.set_title("Where the $\\kappa$ semantic layer helps: F1 gain over the best baseline",
    #             fontsize=12.5, pad=58, fontweight="bold")
    ax.text(0.5, 1.135, "blue = Ours wins,  orange = Ours loses   ·   20% Byzantine, real data",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="#555555")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{out}_heatmap.{e}", dpi=220, bbox_inches="tight")
    print(f"wrote {out}_heatmap.png and .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="label=path per dataset")
    ap.add_argument("--out", default="baselines")
    ap.add_argument("--clip", type=float, default=20.0, help="heatmap color clip (|dF1|)")
    a = ap.parse_args()
    import os
    d = os.path.dirname(a.out)
    if d:
        os.makedirs(d, exist_ok=True)
    data = load(a.csv)
    D = deltas(data)
    #table_f1(data, a.out)
    #table_delta(data, D, a.out)
    heatmap(data, D, a.out, clip=a.clip)


if __name__ == "__main__":
    main()
