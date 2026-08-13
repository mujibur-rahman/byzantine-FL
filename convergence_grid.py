#!/usr/bin/env python3
"""
convergence_grid.py — one figure combining the convergence curves of all
datasets (a 2x2 grid of accuracy-vs-round, one panel per dataset).

Consumes the <out>_curves.csv files written by convergence_experiment.py
(columns: method, round, acc, f1).

Usage:
  python3 convergence_grid.py \
      --csv NYC-Taxi=conv_nyc_curves.csv Foursquare=4sq_conv_curves.csv \
            Yelp=yelp_conv_curves.csv Geolife=geolife_conv_curves.csv \
      --out convergence_all --thresh 85

Deps: numpy, pandas, matplotlib.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "FedAvg":      ("#999999", "-"),
    "TrimmedMean": ("#56B4E9", "-"),
    "RFVIR":       ("#009E73", "-"),
    "Multi-Krum":  ("#E69F00", "-"),
    "Bulyan":      ("#D55E00", "-"),
    "FLAME":       ("#CC79A7", "-"),
    "Ours":        ("#0072B2", "-"),
}


def load(pairs):
    data = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--csv expects label=path, got '{p}'")
        label, path = p.split("=", 1)
        data[label] = pd.read_csv(path)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="label=path per dataset")
    ap.add_argument("--out", default="convergence_all")
    ap.add_argument("--thresh", type=float, default=85.0)
    ap.add_argument("--ymin", type=float, default=None, help="fixed y-axis lower bound")
    a = ap.parse_args()
    d = os.path.dirname(a.out)
    if d:
        os.makedirs(d, exist_ok=True)

    data = load(a.csv)
    names = list(data.keys())
    ncol = 2
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.7 * nrow),
                             squeeze=False, sharex=False)

    for idx, name in enumerate(names):
        ax = axes[idx // ncol][idx % ncol]
        df = data[name]
        multi = "seed" in df.columns and df["seed"].nunique() > 1
        for m in STYLE:
            sub = df[df.method == m]
            if sub.empty:
                continue
            g = sub.groupby("round")["acc"]
            mean = g.mean()
            color, ls = STYLE[m]
            lw = 2.6 if m == "Ours" else 1.4
            z = 6 if m == "Ours" else 3
            ax.plot(mean.index, mean.values, ls, color=color, lw=lw,
                    label=m, zorder=z)
            if multi:
                std = g.std(ddof=1).fillna(0.0)
                ax.fill_between(mean.index, mean.values - std.values,
                                mean.values + std.values, color=color,
                                alpha=0.10, lw=0, zorder=z - 1)
        ax.axhline(a.thresh, ls=":", lw=1, color="#aaaaaa")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Federated round", fontsize=9)
        ax.set_ylabel("Accuracy (%)", fontsize=9)
        if a.ymin is not None:
            ax.set_ylim(a.ymin, 100)
        ax.grid(True, lw=0.4, color="#ececec"); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for j in range(len(names), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    # single shared legend (fixed method order), below the grid
    handles = [plt.Line2D([0], [0], color=STYLE[m][0],
                          lw=2.6 if m == "Ours" else 1.4) for m in STYLE]
    fig.legend(handles, list(STYLE), loc="lower center", ncol=len(STYLE),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    #fig.suptitle("Convergence speed by aggregation algorithm across datasets "
    #             f"(dotted = {a.thresh:.0f}% threshold)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    for e in ("png", "pdf"):
        fig.savefig(f"{a.out}.{e}", dpi=220, bbox_inches="tight")
    print(f"wrote {a.out}.png and .pdf")


if __name__ == "__main__":
    main()
