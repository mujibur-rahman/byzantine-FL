#!/usr/bin/env python3
"""
divergence_figure.py — master table + directional-consistency figure for the
Kappa vs KL vs Renyi ablation, with +/- std error bars.

Consumes the per-dataset CSVs written by divergence_ablation.py
(columns: attack, measure, AUC_mean, AUC_std). Re-run the ablation once per
dataset with a distinct --out so each CSV is kept, e.g.:

  python3 divergence_ablation.py --data nyc_real100000.csv        --seeds 10 --out nyc
  python3 divergence_ablation.py --data foursquare_real100000.csv --seeds 10 --out foursquare
  python3 divergence_ablation.py --data yelp_real100000.csv       --seeds 10 --out yelp
  python3 divergence_ablation.py --data geolife_real100000.csv    --seeds 10 --out geolife

Then:
  python3 divergence_figure.py --csv NYC=nyc.csv Foursquare=foursquare.csv \
                               Yelp=yelp.csv Geolife=geolife.csv --out divergence

Outputs:
  <out>_master.tex   full AUC table (all datasets x attacks x measures, mean+-std)
  <out>_master.csv   same, tidy
  <out>_directional.png / .pdf   grouped-bar figure: label-flip vs grad-poison,
                                 kappa/KL/Renyi per dataset with +/- std, chance line

Dependencies: numpy, pandas, matplotlib.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ATTACKS   = ["lfp", "spf", "ooa", "grad-p"]
MEASURES  = ["kappa", "KL", "KL_sym", "Renyi_0.5", "Renyi_2", "Renyi_inf"]
# Okabe-Ito, CVD-safe, fixed order (identity, never cycled)
COLOR = {"kappa": "#0072B2", "KL": "#D55E00", "Renyi_2": "#009E73"}
LABEL = {"kappa": r"Cohen's $\kappa$", "KL": "KL", "Renyi_2": r"Renyi ($\alpha$=2)"}


def load(pairs):
    frames = []
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--csv expects label=path, got '{p}'")
        label, path = p.split("=", 1)
        df = pd.read_csv(path)
        if "AUC_std" not in df.columns:
            df["AUC_std"] = 0.0
        df["dataset"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def write_master(df, out):
    order = list(dict.fromkeys(df["dataset"]))          # preserve CLI order
    df.to_csv(f"{out}_master.csv", index=False)
    piv_m = df.pivot_table(index=["dataset", "attack"], columns="measure", values="AUC_mean")
    piv_s = df.pivot_table(index=["dataset", "attack"], columns="measure", values="AUC_std")

    def cell(ds, atk, meas):
        try:
            m = piv_m.loc[(ds, atk), meas]; s = piv_s.loc[(ds, atk), meas]
        except KeyError:
            return "--"
        return f"{m:.3f}\\,$\\pm$\\,{s:.3f}" if s > 0 else f"{m:.3f}"

    cols = [m for m in MEASURES if m in df["measure"].unique()]
    with open(f"{out}_master.tex", "w") as f:
        f.write("\\begin{table*}[t]\\centering\\small\n")
        f.write("\\caption{Byzantine-detection AUC per divergence measure, "
                "attack, and dataset (higher is better; 0.5 = chance).}\n")
        f.write("\\label{tab:divergence_master}\n")
        f.write("\\begin{tabular}{ll" + "c" * len(cols) + "}\n\\toprule\n")
        f.write("Dataset & Attack & " + " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\\n\\midrule\n")
        for ds in order:
            for i, atk in enumerate(ATTACKS):
                name = ds if i == 0 else ""
                f.write(f"{name} & {atk} & " + " & ".join(cell(ds, atk, m) for m in cols) + " \\\\\n")
            f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    print(f"wrote {out}_master.tex and {out}_master.csv")


def directional_figure(df, out, series=("kappa", "KL", "Renyi_2")):
    order = list(dict.fromkeys(df["dataset"]))
    panels = [("lfp", "Label-flip (semantic attack)"),
              ("grad-p", "Gradient-poisoning")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    x = np.arange(len(order)); w = 0.8 / len(series)

    for ax, (atk, title) in zip(axes, panels):
        for k, meas in enumerate(series):
            m, s = [], []
            for ds in order:
                row = df[(df.dataset == ds) & (df.attack == atk) & (df.measure == meas)]
                m.append(float(row["AUC_mean"].iloc[0]) if len(row) else np.nan)
                s.append(float(row["AUC_std"].iloc[0]) if len(row) else 0.0)
            ax.bar(x + (k - (len(series) - 1) / 2) * w, m, w, yerr=s, capsize=3,
                   color=COLOR.get(meas, "#888888"), label=LABEL.get(meas, meas),
                   edgecolor="white", linewidth=0.6)
        ax.axhline(0.5, ls="--", lw=1, color="#666666", zorder=0)
        ax.text(len(order) - 0.5, 0.51, "chance", fontsize=8, color="#666666", ha="right")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(order, fontsize=9)
        ax.set_ylim(0, 1.05); ax.grid(axis="y", lw=0.4, color="#dddddd", zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Detection AUC")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    #fig.suptitle("Directional (in)consistency: KL flips between attacks; "
    #             r"$\kappa$ stays above chance for both", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}_directional.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {out}_directional.png and .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="label=path per dataset")
    ap.add_argument("--out", default="divergence")
    a = ap.parse_args()
    df = load(a.csv)
    write_master(df, a.out)
    directional_figure(df, a.out)


if __name__ == "__main__":
    main()
