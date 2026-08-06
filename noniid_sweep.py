#!/usr/bin/env python3
"""
noniid_sweep.py -- robustness of the proposed aggregator across data
heterogeneity. For each Byzantine attack, runs the REAL FL pipeline at several
non-IID (Dirichlet alpha) levels and records the global model's accuracy at
every round, then draws the 2x2 accuracy-vs-round figure (one panel per attack,
one line per non-IID level).

Unlike a hand-drawn decay model, every curve here is measured from
build_federation / run_federation -- so it is publishable evidence.

RUN FROM THE REPO DIRECTORY.

Usage:
  python3 noniid_sweep.py --data nyc_real100000.csv --out noniid_nyc
  python3 noniid_sweep.py --data nyc_real100000.csv --method Ours --byz 0.20 --out noniid_nyc
  python3 noniid_sweep.py --replot noniid_nyc_curves.csv --out noniid_nyc   # redraw only

Outputs: <out>_curves.csv (attack,alpha,round,acc,f1), <out>.png/.pdf
Deps: numpy, pandas, matplotlib, scikit-learn (+ repo modules).
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# real repo attack codes -> panel titles
ATTACKS = [("grad-p", "Gradient Poisoning"),
           ("spf",    "GPS Spoofing"),
           ("lfp",    "Label Flipping"),
           ("ooa",    "On-Off Attack")]

# CVD-safe Okabe-Ito, distinct dashes per non-IID level
ALPHA_STYLE = [
    ("#0072B2", "-"),   # 0.01
    ("#E69F00", "--"),  # 0.05
    ("#009E73", "-."),  # 0.1
    ("#D55E00", ":"),   # 0.2
    ("#CC79A7", (0, (3, 1, 1, 1))),  # 0.5
    ("#555555", (0, (1, 1))),        # 1.0
]


def synthetic(n=100_000, fraud=0.15, seed=42):
    rng = np.random.RandomState(seed); nf = int(n*fraud); nl = n-nf
    legit = np.column_stack([rng.uniform(-74.05,-73.75,nl),rng.uniform(40.6,40.9,nl),
        rng.uniform(-74.05,-73.75,nl),rng.uniform(40.6,40.9,nl),rng.uniform(300,3600,nl),
        rng.uniform(0.5,20,nl),rng.uniform(5,80,nl),rng.randint(0,24,nl).astype(float)])
    fr = np.column_stack([rng.uniform(-74.05,-73.75,nf),rng.uniform(40.6,40.9,nf),
        rng.uniform(-74.05,-73.75,nf),rng.uniform(40.6,40.9,nf),rng.uniform(60,300,nf),
        rng.uniform(0.1,1.0,nf),rng.uniform(30,120,nf),rng.randint(0,24,nf).astype(float)])
    X = np.vstack([legit, fr]); y = np.concatenate([np.zeros(nl), np.ones(nf)])
    d = np.maximum(X[:,5],0.1); X = np.column_stack([X, X[:,6]/d, np.maximum(X[:,4],1.0)/d])
    i = rng.permutation(n); return X[i], y[i]


def load_data(path):
    if path is None: return synthetic()
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
    for code, title in ATTACKS:
        for alpha in a.alphas:
            # reseed so every (attack, alpha) cell starts from the SAME model
            # init and only the Dirichlet partition/attack differ
            np.random.seed(42)
            server, clients, byz = build_federation(
                Xs, y, scaler, Xv, yv, n_clients=a.clients,
                clients_per_round=a.per_round, byzantine_ratio=a.byz,
                attack_type=code, noniid_alpha=alpha, seed=42)
            server.aggregator = AGGREGATOR_REGISTRY[a.method](n_clients=a.clients)
            hist = run_federation(server, clients, set(byz), n_rounds=a.rounds,
                                  verbose=False, seed=42)
            for h in hist:
                rows.append({"attack": code, "alpha": alpha, "round": h["round"],
                             "acc": h["accuracy"], "f1": h["f1"]})
            fa = np.mean([h["accuracy"] for h in hist[-5:]]) if hist else 0.0
            print(f"  {title:18s} a={alpha:<5}  final_acc={fa:5.1f}%")
    df = pd.DataFrame(rows); df.to_csv(f"{a.out}_curves.csv", index=False)
    print(f"wrote {a.out}_curves.csv")
    return df


def plot(df, out, method, metric="acc", min_alpha=0.0):
    ylab = {"acc": "Accuracy (%)", "f1": "Minority-class F1"}[metric]
    df = df[df["alpha"] >= min_alpha]
    alphas = sorted(df["alpha"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5), sharey=True)
    for ax, (code, title) in zip(axes.flatten(), ATTACKS):
        d = df[df.attack == code]
        for i, alpha in enumerate(alphas):
            sub = d[d.alpha == alpha].sort_values("round")
            if sub.empty: continue
            color, ls = ALPHA_STYLE[i % len(ALPHA_STYLE)]
            ax.plot(sub["round"], sub[metric], ls=ls, color=color, lw=3.0, label=str(alpha))
        ax.set_title(title, fontsize=19)
        ax.set_xlabel("Round", fontsize=17)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, ls="--", alpha=0.5); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0, 0].set_ylabel(ylab, fontsize=17)
    axes[1, 0].set_ylabel(ylab, fontsize=17)
    handles = [plt.Line2D([0], [0], color=ALPHA_STYLE[i][0], ls=ALPHA_STYLE[i][1], lw=3.0)
               for i in range(len(alphas))]
    fig.legend(handles, [str(x) for x in alphas], title="Non-IID Level (Dirichlet $\\alpha$)",
               loc="lower center", ncol=len(alphas), frameon=False, bbox_to_anchor=(0.5, -0.02),
               fontsize=15, title_fontsize=16)
    metric_name = {"acc": "Accuracy", "f1": "Minority-class F1"}[metric]
    fig.suptitle(f"{metric_name} under Byzantine attacks across non-IID levels ({method})",
                 fontsize=18)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=220, bbox_inches="tight")
    print(f"wrote {out}.png and .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--method", default="Ours")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    ap.add_argument("--byz", type=float, default=0.20)
    ap.add_argument("--clients", type=int, default=100)
    ap.add_argument("--per-round", type=int, default=20, dest="per_round")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--out", default="noniid_sweep")
    ap.add_argument("--replot", default=None)
    ap.add_argument("--metric", default="acc", choices=["acc", "f1"],
                    help="which curve to plot")
    ap.add_argument("--min-alpha", type=float, default=0.0, dest="min_alpha",
                    help="drop non-IID levels below this alpha from the figure")
    a = ap.parse_args()
    d = os.path.dirname(a.out)
    if d: os.makedirs(d, exist_ok=True)
    df = pd.read_csv(a.replot) if a.replot else run(a)
    plot(df, a.out, a.method, metric=a.metric, min_alpha=a.min_alpha)


if __name__ == "__main__":
    main()
