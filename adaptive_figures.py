#!/usr/bin/env python3
"""
adaptive_figures.py — Table XII + latency figure from experiment5 adaptive CSVs.

Consumes the *_adaptive_detection_latency.csv files (columns: Variant, Attack,
Outlier latency, Trust latency, Kappa latency, Combined latency,
Burn-in FPR (%), Final Acc (%)) and emits:
  1. Table XII (LaTeX): detection latency per layer + burn-in FPR + final acc,
     per dataset x variant. Rows are marked invalid when the result is not
     interpretable (burn-in FPR above --fpr-max => detector saturated, or final
     accuracy below --acc-min => model did not converge).
  2. Latency figure: Kappa vs Trust detection latency (rounds after activation)
     for the sleeper and slow-drift variants, per dataset — visualising the
     stateless-vs-stateful gap. 'Never' is drawn as a capped bar.

Usage:
  python3 adaptive_figures.py --csv NYC-Taxi=results/nyc-taxi/nyc-taxi_adaptive_detection_latency.csv \
      Foursquare=results/foursquare/foursquare_adaptive_detection_latency.csv \
      Yelp=results/yelp/yelp_adaptive_detection_latency.csv \
      Geolife=results/geolife/geolife_adaptive_detection_latency.csv --out fig/adaptive

Deps: numpy, pandas, matplotlib.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANT_KEY = {"A": "Sleeper", "B": "Mimicry", "C": "Slow-drift"}
LAYERS = ["Outlier latency", "Trust latency", "Kappa latency", "Combined latency"]
NEVER_CAP = None   # set in main from --never-cap


def load(pairs):
    data = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--csv expects label=path, got '{p}'")
        label, path = p.split("=", 1)
        df = pd.read_csv(path)
        df["V"] = df["Variant"].astype(str).str[0].map(VARIANT_KEY).fillna(df["Variant"])
        data[label] = df
    return data


def _lat(v):
    """latency cell -> (numeric or np.inf for Never, display string)."""
    s = str(v).strip()
    if s.lower().startswith("never"):
        return np.inf, "Never"
    try:
        return float(s.split()[0]), s.split()[0]
    except ValueError:
        return np.inf, s


def table_xii(data, out, fpr_max, acc_min):
    L = [r"\begin{table*}[t]\centering\small",
         r"\caption{Adaptive-adversary detection latency (rounds after activation; "
         r"lower is better, `--' = never) under a stateless Kappa layer vs.\ the "
         r"stateful trust layer, with burn-in false-positive rate and final accuracy. "
         r"$\dagger$ marks results that are not interpretable: burn-in FPR $>" +
         f"{fpr_max:.0f}" + r"\%$ (detector saturated) or final accuracy $<" +
         f"{acc_min:.0f}" + r"\%$ (model did not converge).}",
         r"\label{tab:adaptive}",
         r"\begin{tabular}{ll" + "cccc" + "cc}", r"\toprule",
         r"Dataset & Variant & Outlier & Trust & \textbf{Kappa} & Combined & "
         r"Burn-in FPR & Final Acc \\", r"\midrule"]
    for dname, df in data.items():
        for i, (_, r) in enumerate(df.iterrows()):
            fpr = float(r["Burn-in FPR (%)"]); acc = float(r["Final Acc (%)"])
            invalid = (fpr > fpr_max) or (acc < acc_min)
            mark = r"$^\dagger$" if invalid else ""
            cells = []
            for lyr in LAYERS:
                _, disp = _lat(r[lyr])
                cells.append(r"\textbf{" + disp + "}" if lyr == "Kappa latency"
                             and disp != "Never" and not invalid else disp)
            pre = dname if i == 0 else ""
            L.append(f"{pre} & {r['V']}{mark} & " + " & ".join(cells)
                     + f" & {fpr:.1f}\\% & {acc:.1f}\\% \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table*}"]
    open(f"{out}_table_xii.tex", "w").write("\n".join(L) + "\n")
    print(f"wrote {out}_table_xii.tex")


def latency_figure(data, out, cap, fpr_max, acc_min):
    names = list(data.keys())
    variants = ["Sleeper", "Slow-drift"]      # detectable ones; Mimicry evades
    fig, axes = plt.subplots(1, len(variants), figsize=(5.6 * len(variants), 4.4),
                             sharey=True)
    x = np.arange(len(names)); w = 0.38
    for ax, var in zip(axes, variants):
        kap, tru, invalid = [], [], []
        for dn in names:
            df = data[dn]; row = df[df["V"] == var]
            if row.empty:
                kap.append(np.nan); tru.append(np.nan); invalid.append(False); continue
            k, _ = _lat(row["Kappa latency"].iloc[0])
            t, _ = _lat(row["Trust latency"].iloc[0])
            fpr = float(row["Burn-in FPR (%)"].iloc[0]); acc = float(row["Final Acc (%)"].iloc[0])
            inv = (fpr > fpr_max) or (acc < acc_min)
            kap.append(cap if np.isinf(k) else k)
            tru.append(cap if np.isinf(t) else t)
            invalid.append(inv)
        bk = ax.bar(x - w/2, kap, w, color="#0072B2", label="Kappa (stateless)",
                    edgecolor="white")
        bt = ax.bar(x + w/2, tru, w, color="#E69F00", label="Trust (stateful)",
                    edgecolor="white")
        # annotate Never (== cap) and invalid rows
        for xi, (k, t, inv) in enumerate(zip(kap, tru, invalid)):
            if k >= cap: ax.text(xi - w/2, cap + 0.5, "Never", ha="center", fontsize=9, color="#0072B2", rotation=90, va="bottom")
            if t >= cap: ax.text(xi + w/2, cap + 0.5, "Never", ha="center", fontsize=9, color="#E69F00", rotation=90, va="bottom")
            if inv: ax.text(xi, cap + 4.5, "†", ha="center", fontsize=13, color="#c0392b")
        ax.set_title(var, fontsize=12)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11.5)
        ax.set_ylim(0, cap + 6)
        ax.set_ylabel("Detection latency (rounds after activation)", fontsize=11.5)
        ax.grid(axis="y", lw=0.4, color="#dddddd"); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=11.5, loc="upper right")
    #fig.suptitle("Stateless Kappa detects adaptive attacks in ~1 round; "
    #             "stateful trust lags by tens of rounds\n"
    #             "(bars capped at 'Never'; † = non-converged / saturated, not interpretable)",
    #             fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for e in ("png", "pdf"):
        fig.savefig(f"{out}_latency.{e}", dpi=220, bbox_inches="tight")
    print(f"wrote {out}_latency.png and .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="label=path per dataset")
    ap.add_argument("--out", default="adaptive")
    ap.add_argument("--fpr-max", type=float, default=20.0, help="max valid burn-in FPR %")
    ap.add_argument("--acc-min", type=float, default=75.0, help="min converged final acc %")
    ap.add_argument("--never-cap", type=float, default=40.0, help="bar height for 'Never' (=N_ROUNDS-T_burn)")
    a = ap.parse_args()
    d = os.path.dirname(a.out)
    if d:
        os.makedirs(d, exist_ok=True)
    data = load(a.csv)
    table_xii(data, a.out, a.fpr_max, a.acc_min)
    latency_figure(data, a.out, a.never_cap, a.fpr_max, a.acc_min)


if __name__ == "__main__":
    main()
