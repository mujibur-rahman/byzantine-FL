#!/usr/bin/env python3
"""
churn_experiment.py — robustness of the proposed defence to CLIENT CHURN
(dropout) under Byzantine attack. In spatial crowdsourcing, sampled clients
routinely go offline mid-round; we drop each sampled client with probability
p_churn per round (run_federation's p_churn) and track accuracy over rounds,
for each of the four attacks, multi-seed (mean±s.d.).

Two outputs:
  1. Accuracy-vs-round figure at a chosen churn rate (one line per attack,
     ±s.d. band) — the "four attacks under dropout" plot.
  2. A final-accuracy-vs-churn table (per attack × churn rate, mean±s.d. and
     Δ from no churn) — quantifies how much churn costs.

RUN FROM THE REPO DIRECTORY.

Usage:
  python3 churn_experiment.py --data nyc_real100000.csv --out churn_nyc
  python3 churn_experiment.py --data nyc_real100000.csv --churns 0 0.1 0.2 0.3 \
      --attacks grad-p spf lfp ooa --seeds 42 43 44 --plot-churn 0.1 --out churn_nyc
  python3 churn_experiment.py --replot churn_nyc_curves.csv --out churn_nyc
Outputs: <out>_curves.csv, <out>_summary.csv, <out>_latex.txt, <out>.png/.pdf
Deps: numpy, pandas, matplotlib, scikit-learn (+ repo modules).
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

ATTACKS = [("grad-p", "Grad-p", "#D55E00", ":"),
           ("spf",    "SPF",       "#E69F00", "--"),
           ("lfp",    "LFP",     "#0072B2", "-"),
           ("ooa",    "OOA",             "#009E73", "-.")]
ATK_TITLE = {c: t for c, t, _, _ in ATTACKS}


def synthetic(n=50000, fraud=0.17, seed=42):
    rng = np.random.RandomState(seed); nf = int(n*fraud); nl = n-nf
    legit = np.column_stack([rng.uniform(-74.05,-73.75,nl),rng.uniform(40.6,40.9,nl),
        rng.uniform(-74.05,-73.75,nl),rng.uniform(40.6,40.9,nl),rng.uniform(300,3600,nl),
        rng.uniform(0.5,20,nl),rng.uniform(5,80,nl),rng.randint(0,24,nl).astype(float)])
    fr = np.column_stack([rng.uniform(-74.05,-73.75,nf),rng.uniform(40.6,40.9,nf),
        rng.uniform(-74.05,-73.75,nf),rng.uniform(40.6,40.9,nf),rng.uniform(60,300,nf),
        rng.uniform(0.1,1.0,nf),rng.uniform(30,120,nf),rng.randint(0,24,nf).astype(float)])
    X = np.vstack([legit,fr]); y = np.concatenate([np.zeros(nl),np.ones(nf)])
    d = np.maximum(X[:,5],0.1); X = np.column_stack([X, X[:,6]/d, np.maximum(X[:,4],1.0)/d])
    i = rng.permutation(n); return X[i], y[i]


def load_data(path):
    if path is None:
        return synthetic()
    if path.endswith(".npz"):
        d = np.load(path); return d["X"].astype(float), d["y"].astype(float)
    df = pd.read_csv(path)
    y = df["label"].values.astype(float) if "label" in df.columns else df.iloc[:,-1].values.astype(float)
    X = (df.drop(columns=["label"]) if "label" in df.columns else df.iloc[:,:-1]).values.astype(float)
    return X, y


def run(a):
    from fl_base import make_validation_set
    from fl_runner import build_federation, run_federation
    from aggregators import AGGREGATOR_REGISTRY
    X, y = load_data(a.data)
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    Xv, yv = make_validation_set(Xs, y, min(2000, len(Xs)), seed=42)

    rows = []
    for code, title, _, _ in ATTACKS:
        if code not in a.attacks:
            continue
        for churn in a.churns:
            for seed in a.seeds:
                np.random.seed(seed)
                server, clients, byz = build_federation(
                    Xs, y, scaler, Xv, yv, n_clients=a.clients,
                    clients_per_round=a.per_round, byzantine_ratio=a.byz,
                    attack_type=code, noniid_alpha=0.5, seed=seed)
                server.aggregator = AGGREGATOR_REGISTRY[a.method](n_clients=a.clients)
                hist = run_federation(server, clients, set(byz), n_rounds=a.rounds,
                                      p_churn=churn, verbose=False, seed=seed)
                for h in hist:
                    rows.append({"attack": code, "churn": churn, "seed": seed,
                                 "round": h["round"], "acc": h["accuracy"], "f1": h["f1"]})
            sub = [r for r in rows if r["attack"] == code and r["churn"] == churn
                   and r["round"] >= a.rounds - 5]
            fa = np.mean([r["acc"] for r in sub]) if sub else 0.0
            print(f"  {title:18s} churn={churn:<4} final_acc={fa:5.1f}")
    df = pd.DataFrame(rows); df.to_csv(f"{a.out}_curves.csv", index=False)
    print(f"wrote {a.out}_curves.csv")
    return df


def summarise(df, out):
    churns = sorted(df["churn"].unique())
    attacks = [c for c, *_ in ATTACKS if c in df["attack"].unique()]
    # final acc per (attack, churn), mean±std over seeds
    rmax = df["round"].max()
    rec = {}
    for c in attacks:
        for ch in churns:
            s = df[(df.attack == c) & (df.churn == ch) & (df["round"] >= rmax - 4)]
            per_seed = s.groupby("seed")["acc"].mean()
            rec[(c, ch)] = (per_seed.mean(), per_seed.std(ddof=0))
    # console + csv
    outrows = []
    for c in attacks:
        row = {"attack": ATK_TITLE[c]}
        base = rec[(c, churns[0])][0]
        for ch in churns:
            m, s = rec[(c, ch)]
            d = "" if ch == churns[0] else f" ({m-base:+.1f})"
            row[f"churn={ch}"] = f"{m:.1f}±{s:.1f}{d}"
        outrows.append(row)
    S = pd.DataFrame(outrows); S.to_csv(f"{out}_summary.csv", index=False)
    print("\nFinal accuracy vs churn (mean±s.d., Δ from no-churn):")
    print(S.to_string(index=False)); print(f"\nwrote {out}_summary.csv")

    # LaTeX
    NL = r" \\"
    L = [r"\begin{table}[h]", r"\centering", r"\small",
         r"\caption{Final accuracy (\%) under client churn per attack "
         r"(20\% Byzantine, mean$\pm$s.d.\ over " + str(df["seed"].nunique()) +
         r" seeds). $\Delta$ from no churn in parentheses.}",
         r"\label{table:churn}",
         r"\begin{tabular}{l" + "c" * len(churns) + r"}", r"\toprule",
         r"\textbf{Attack} & " + " & ".join(rf"$p{{=}}{ch}$" for ch in churns) + NL,
         r"\midrule"]
    for c in attacks:
        cells = []
        base = rec[(c, churns[0])][0]
        for ch in churns:
            m, s = rec[(c, ch)]
            d = "" if ch == churns[0] else f" ({m-base:+.1f})"
            cells.append(f"{m:.1f}$\\pm${s:.1f}{d}")
        L.append(ATK_TITLE[c] + " & " + " & ".join(cells) + NL)
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{out}_latex.txt", "w").write("\n".join(L) + "\n")
    print(f"wrote {out}_latex.txt")


def plot(df, out, plot_churn):
    churns = sorted(df["churn"].unique())
    if plot_churn not in churns:
        plot_churn = min(churns, key=lambda x: abs(x - plot_churn))
    d = df[df.churn == plot_churn]
    multi = d["seed"].nunique() > 1
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for code, title, color, ls in ATTACKS:
        s = d[d.attack == code]
        if s.empty:
            continue
        g = s.groupby("round")["acc"]
        mean = g.mean()
        ax.plot(mean.index, mean.values, ls, color=color, lw=2.2, label=title)
        if multi:
            std = g.std(ddof=1).fillna(0.0)
            ax.fill_between(mean.index, mean.values - std.values,
                            mean.values + std.values, color=color, alpha=0.2, lw=0)
    ax.set_xlabel("Federated round", fontsize=14)
    ax.set_ylabel("Global model accuracy (%)", fontsize=14)
    #ax.set_title(f"Accuracy under client churn ($p_{{\\mathrm{{churn}}}}={plot_churn}$), "
    #             f"20\\% Byzantine", fontsize=14)
    ax.grid(True, lw=0.6, color="#e8e8e8"); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=11, title="Attack", loc="lower right")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=220, bbox_inches="tight")
    print(f"wrote {out}.png and .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--method", default="Ours")
    ap.add_argument("--attacks", nargs="+", default=["grad-p", "spf", "lfp", "ooa"])
    ap.add_argument("--churns", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--byz", type=float, default=0.20)
    ap.add_argument("--clients", type=int, default=100)
    ap.add_argument("--per-round", type=int, default=20, dest="per_round")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--plot-churn", type=float, default=0.1, dest="plot_churn",
                    help="churn rate shown in the accuracy-vs-round figure")
    ap.add_argument("--out", default="churn")
    ap.add_argument("--replot", default=None)
    a = ap.parse_args()
    d = os.path.dirname(a.out)
    if d:
        os.makedirs(d, exist_ok=True)
    df = pd.read_csv(a.replot) if a.replot else run(a)
    summarise(df, a.out)
    plot(df, a.out, a.plot_churn)


if __name__ == "__main__":
    main()
