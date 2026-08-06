#!/usr/bin/env python3
"""
noniid_multi.py -- combine several datasets' non-IID sweeps into ONE figure:
rows = Byzantine attacks, columns = datasets. One panel per (attack, dataset),
one line per non-IID level. Shared legend. Built for tight page budgets.

Consumes the <out>_curves.csv files from noniid_sweep.py
(columns: attack, alpha, round, acc, f1).

Usage:
  python3 noniid_multi.py \
      --csv NYC-Taxi=noniid_nyc_curves.csv Yelp=noniid_yelp_curves.csv \
            Foursquare=noniid_4sq_curves.csv \
      --metric f1 --min-alpha 0.1 --out noniid_f1_all

Outputs: <out>.png/.pdf
Deps: numpy, pandas, matplotlib.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ATTACKS = [("grad-p", "grad-p"),
           ("spf",    "spf"),
           ("lfp",    "lfp"),
           ("ooa",    "ooa")]

ALPHA_STYLE = [
    ("#0072B2", "-"),
    ("#E69F00", "--"),
    ("#009E73", "-."),
    ("#D55E00", ":"),
    ("#CC79A7", (0, (3, 1, 1, 1))),
    ("#555555", (0, (1, 1))),
]


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
    ap.add_argument("--metric", default="f1", choices=["acc", "f1"])
    ap.add_argument("--min-alpha", type=float, default=0.0, dest="min_alpha")
    ap.add_argument("--out", default="noniid_multi")
    a = ap.parse_args()
    d = os.path.dirname(a.out)
    if d: os.makedirs(d, exist_ok=True)

    data = load(a.csv)
    names = list(data.keys())
    ylab = {"acc": "Accuracy (%)", "f1": "F1"}[a.metric]
    metric_name = {"acc": "Accuracy", "f1": "F1"}[a.metric]

    # union of alpha levels (>= floor) across datasets, for a stable legend
    alphas = sorted({v for df in data.values()
                     for v in df["alpha"].unique() if v >= a.min_alpha})

    nrow, ncol = len(ATTACKS), len(names)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.0 * nrow),
                             squeeze=False, sharex=True)

    for r, (code, atitle) in enumerate(ATTACKS):
        for c, name in enumerate(names):
            ax = axes[r][c]
            df = data[name]
            df = df[(df.attack == code) & (df["alpha"] >= a.min_alpha)]
            for i, alpha in enumerate(alphas):
                sub = df[df.alpha == alpha].sort_values("round")
                if sub.empty: continue
                color, ls = ALPHA_STYLE[i % len(ALPHA_STYLE)]
                ax.plot(sub["round"], sub[a.metric], ls=ls, color=color, lw=2.6)
            ax.grid(True, ls="--", alpha=0.5); ax.set_axisbelow(True)
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
            ax.tick_params(axis="both", labelsize=13)
            if r == 0:
                ax.set_title(name, fontsize=18, pad=8)
            if r == nrow - 1:
                ax.set_xlabel("Round", fontsize=16)
            if c == 0:
                ax.set_ylabel(f"{atitle}\n{ylab}", fontsize=13)

    handles = [plt.Line2D([0], [0], color=ALPHA_STYLE[i][0], ls=ALPHA_STYLE[i][1], lw=2.6)
               for i in range(len(alphas))]
    fig.legend(handles, [str(x) for x in alphas],
               title="Non-IID Level (Dirichlet $\\alpha$)",
               loc="lower center", ncol=len(alphas), frameon=False,
               bbox_to_anchor=(0.5, -0.03), fontsize=14, title_fontsize=15)
    #fig.suptitle(f"{metric_name} under Byzantine attacks across non-IID levels (Ours)",
    #             fontsize=18)
    fig.tight_layout(rect=[0, 0.045, 1, 0.965])
    for e in ("png", "pdf"):
        fig.savefig(f"{a.out}.{e}", dpi=220, bbox_inches="tight")
    print(f"wrote {a.out}.png and .pdf")


if __name__ == "__main__":
    main()
