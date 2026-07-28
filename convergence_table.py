#!/usr/bin/env python3
"""
convergence_table.py -- combine the per-dataset convergence curves into one
paste-ready LaTeX table (one row per method x dataset).

Recomputes the same four metrics as convergence_experiment.summarise() straight
from the <out>_curves.csv files, so the table always matches the figure:
  final_acc  : mean accuracy over the last 5 rounds
  round@90%  : first round reaching 90% of that method's OWN final_acc (speed)
  round@thr  : first round reaching an absolute accuracy threshold (--thresh)
  auc        : mean accuracy across all rounds (normalised area under curve)

Usage:
  python3 convergence_table.py \
      --csv NYC-Taxi=conv_nyc_curves.csv Foursquare=4sq_conv_curves.csv \
            Yelp=yelp_conv_curves.csv Geolife=geolife_conv_curves.csv \
      --thresh 85 --out convergence_table

Outputs: <out>.tex (LaTeX table), <out>.csv (flat long-form for the record).
Deps: numpy, pandas.
"""
import argparse
import numpy as np
import pandas as pd

METHOD_ORDER = ["FedAvg", "TrimmedMean", "RFVIR", "Multi-Krum",
                "Bulyan", "FLAME", "Ours"]


def metrics(sub, thresh):
    sub = sub.sort_values("round")
    acc = sub["acc"].values
    rnd = sub["round"].values
    final = float(np.mean(acc[-5:]))
    tgt = 0.90 * final
    r90 = next((int(rnd[i]) for i in range(len(acc)) if acc[i] >= tgt), None)
    rth = next((int(rnd[i]) for i in range(len(acc)) if acc[i] >= thresh), None)
    auc = float(np.mean(acc))
    return final, r90, rth, auc


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
    ap.add_argument("--thresh", type=float, default=85.0)
    ap.add_argument("--out", default="convergence_table")
    a = ap.parse_args()

    data = load(a.csv)
    datasets = list(data.keys())
    thr = a.thresh

    # long-form record + per-(dataset,method) lookup
    rec = []
    table = {}
    for ds, df in data.items():
        methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
        for m in methods:
            f, r90, rth, auc = metrics(df[df.method == m], thr)
            table[(ds, m)] = (f, r90, rth, auc)
            rec.append({"dataset": ds, "method": m, "final_acc": round(f, 1),
                        "round@90%": r90, f"round@{thr:.0f}%": rth,
                        "auc": round(auc, 1)})
    pd.DataFrame(rec).to_csv(f"{a.out}.csv", index=False)

    # best-per-column-per-dataset for bolding (fastest / highest)
    best = {}
    for ds in datasets:
        rows = {m: table[(ds, m)] for m in METHOD_ORDER if (ds, m) in table}
        best[(ds, "final")] = max(v[0] for v in rows.values())
        best[(ds, "r90")] = min(v[1] for v in rows.values() if v[1] is not None)
        best[(ds, "rth")] = min(v[2] for v in rows.values() if v[2] is not None)
        best[(ds, "auc")] = max(v[3] for v in rows.values())

    thr_s = f"{thr:.0f}"
    L = []
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Convergence speed of Byzantine-robust aggregators across "
             r"the four spatial-crowdsourcing datasets (clean setting, 100 clients, "
             r"20 sampled per round, 50 rounds). \textbf{Acc.}: mean accuracy over "
             r"the last five rounds. \textbf{R@90}: first round reaching 90\% of the "
             r"method's own final accuracy (pure saturation speed). \textbf{R@" + thr_s +
             r"}: first round reaching " + thr_s + r"\% absolute accuracy. "
             r"\textbf{AUC}: mean accuracy across all rounds. Best per column per "
             r"dataset in \textbf{bold}; lower is better for R@$\cdot$, higher for "
             r"Acc./AUC. Geolife accuracy equals the majority-class baseline "
             r"($\approx$96.5\%; minority-class F1$\to$0 for every method), so its "
             r"accuracy figures reflect trivial-classifier saturation rather than "
             r"fraud learning (see text).}")
    L.append(r"\label{tab:convergence}")
    L.append(r"\small")
    ncol = len(datasets)
    L.append(r"\begin{tabular}{l" + "".join([r"cccc" for _ in range(ncol)]) + r"}")
    L.append(r"\toprule")
    head = [r"\multicolumn{4}{c}{\textbf{" + ds + r"}}" for ds in datasets]
    L.append(r"\textbf{Method} & " + " & ".join(head) + r" \\")
    sub = [r"Acc. & R@90 & R@" + thr_s + r" & AUC" for _ in datasets]
    L.append(r" & " + " & ".join(sub) + r" \\")
    L.append(r"\midrule")

    def fmt(val, is_best, kind):
        if val is None:
            s = "--"
        elif kind == "int":
            s = f"{int(val)}"
        else:
            s = f"{val:.1f}"
        return r"\textbf{" + s + r"}" if is_best and val is not None else s

    for m in METHOD_ORDER:
        cells = []
        for ds in datasets:
            if (ds, m) not in table:
                cells += ["--", "--", "--", "--"]
                continue
            f, r90, rth, auc = table[(ds, m)]
            cells.append(fmt(f, abs(f - best[(ds, "final")]) < 1e-9, "f1"))
            cells.append(fmt(r90, r90 == best[(ds, "r90")], "int"))
            cells.append(fmt(rth, rth == best[(ds, "rth")], "int"))
            cells.append(fmt(auc, abs(auc - best[(ds, "auc")]) < 1e-9, "f1"))
        row = (r"\textbf{" + m + r"}") if m == "Ours" else m
        if m == "Ours":
            L.append(r"\midrule")
        L.append(row + " & " + " & ".join(cells) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table*}")
    tex = "\n".join(L) + "\n"
    with open(f"{a.out}.tex", "w") as fh:
        fh.write(tex)
    print(tex)
    print(f"wrote {a.out}.tex and {a.out}.csv")


if __name__ == "__main__":
    main()
