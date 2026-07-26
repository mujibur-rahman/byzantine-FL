#!/usr/bin/env python3
"""
layer_figures.py — Table X + per-layer F1 figure from experiment3 layer CSVs.

Consumes the *_layer_confusion.csv files (columns: Attack, Outlier P/R/F1,
Trust P/R/F1, Kappa P/R/F1, Combined P/R/F1, Kappa unique TP, Trust unique TP,
Outlier unique TP) and emits:
  1. Table X (LaTeX): per dataset x attack, P/R/F1 for each layer + Kappa
     unique-TP. Combined F1 bolded only when it beats every single layer.
  2. 2x2 faceted grouped-bar figure: layer F1 per attack, one panel per
     dataset, with the Byzantine base-rate reference line.

Usage:
  python3 layer_figures.py --csv NYC-Taxi=results/nyc-taxi/nyc-taxi_layer_confusion.csv \
      Foursquare=results/foursquare/foursquare_layer_confusion.csv \
      Yelp=results/yelp/yelp_layer_confusion.csv \
      Geolife=results/geolife/geolife_layer_confusion.csv --out fig/layers --byz 20

Deps: numpy, pandas, matplotlib.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ATTACK_ORDER = ["grad-p", "spf", "lfp", "ooa"]
ATTACK_LABEL = {"grad-p": "Grad-poison", "spf": "GPS-spoof",
                "lfp": "Label-flip", "ooa": "On-off"}
LAYERS = ["Outlier", "Trust", "Kappa", "Combined"]
# Okabe-Ito, CVD-safe, fixed order
LCOLOR = {"Outlier": "#999999", "Trust": "#E69F00",
          "Kappa": "#0072B2", "Combined": "#009E73"}


def load(pairs):
    data = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--csv expects label=path, got '{p}'")
        label, path = p.split("=", 1)
        data[label] = pd.read_csv(path).set_index("Attack")
    return data


def table_x(data, out):
    def prf(df, atk, layer):
        return (df.loc[atk, f"{layer} P"], df.loc[atk, f"{layer} R"], df.loc[atk, f"{layer} F1"])
    L = [r"\begin{table*}[t]\centering\scriptsize",
         r"\caption{Per-layer detection (Precision / Recall / F1, \%) under 20\% Byzantine, "
         r"real data. \textbf{Kappa-only TP} = Byzantine clients caught by the Kappa layer but "
         r"missed by the outlier and trust layers (Corollary~1). Combined F1 is \textbf{bold} only "
         r"where it exceeds every single layer. Note: on semantic attacks the Kappa layer's "
         r"precision approaches the 20\% base rate (broad flagging), so read P/R/F1 jointly.}",
         r"\label{tab:layer_analysis}",
         r"\begin{tabular}{ll" + "c" * 4 + "c}", r"\toprule",
         r"Dataset & Attack & Outlier (P/R/F1) & Trust (P/R/F1) & Kappa (P/R/F1) & "
         r"Combined (P/R/F1) & \shortstack{Kappa-only\\TP} \\", r"\midrule"]
    for dname, df in data.items():
        atks = [a for a in ATTACK_ORDER if a in df.index]
        for i, atk in enumerate(atks):
            cells = []
            comb_f1 = df.loc[atk, "Combined F1"]
            single_best = max(df.loc[atk, f"{lyr} F1"] for lyr in ["Outlier", "Trust", "Kappa"])
            for lyr in LAYERS:
                p, r, f = prf(df, atk, lyr)
                cell = f"{p:.0f}/{r:.0f}/{f:.0f}"
                if lyr == "Combined" and comb_f1 > single_best:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
            ktp = int(df.loc[atk, "Kappa unique TP"])
            pre = dname if i == 0 else ""
            L.append(f"{pre} & {ATTACK_LABEL[atk]} & " + " & ".join(cells) + f" & {ktp} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table*}"]
    open(f"{out}_table_x.tex", "w").write("\n".join(L) + "\n")
    print(f"wrote {out}_table_x.tex")


def figure(data, out, byz=20.0):
    names = list(data.keys())
    n = len(names); ncol = 2; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow), squeeze=False)
    x = np.arange(len(ATTACK_ORDER)); w = 0.8 / len(LAYERS)
    for idx, dname in enumerate(names):
        ax = axes[idx // ncol][idx % ncol]
        df = data[dname]
        for k, lyr in enumerate(LAYERS):
            vals = [float(df.loc[a, f"{lyr} F1"]) if a in df.index else np.nan for a in ATTACK_ORDER]
            ax.bar(x + (k - (len(LAYERS) - 1) / 2) * w, vals, w,
                   color=LCOLOR[lyr], label=lyr, edgecolor="white", linewidth=0.5)
        ax.axhline(byz, ls="--", lw=1, color="#c0392b", zorder=0)
        ax.text(len(ATTACK_ORDER) - 0.5, byz + 1.5, f"{byz:.0f}% Byz. base rate",
                fontsize=7.5, color="#c0392b", ha="right")
        ax.set_title(dname, fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels([ATTACK_LABEL[a] for a in ATTACK_ORDER], fontsize=8.5)
        ax.set_ylim(0, 100); ax.set_ylabel("F1 (%)", fontsize=9)
        ax.grid(axis="y", lw=0.4, color="#dddddd"); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(LAYERS), frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Per-layer detection F1 by attack (dashed = 20% Byzantine base rate)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    for e in ("png", "pdf"):
        fig.savefig(f"{out}_layerF1.{e}", dpi=220, bbox_inches="tight")
    print(f"wrote {out}_layerF1.png and .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="label=path per dataset")
    ap.add_argument("--out", default="layers")
    ap.add_argument("--byz", type=float, default=20.0, help="Byzantine base rate (%)")
    a = ap.parse_args()
    d = os.path.dirname(a.out)
    if d:
        os.makedirs(d, exist_ok=True)
    data = load(a.csv)
    table_x(data, a.out)
    figure(data, a.out, byz=a.byz)


if __name__ == "__main__":
    main()
