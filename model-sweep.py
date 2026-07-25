
"""

robustness_sweep.py — accuracy vs malicious-% grouped-bar grid (one subplot per

attack), comparing Ours (multi-layer kappa defence) against FedAvg (and any

other aggregators you pass). Mirrors the SF-CABD "impact of malicious

percentage" figure.



Runs the repo's real FL pipeline (build_federation / run_federation /

AGGREGATOR_REGISTRY), so RUN FROM THE REPO DIRECTORY.



Usage:

  python3 robustness_sweep.py --data nyc_real100000.csv --out robust_nyc

  python3 robustness_sweep.py --data nyc_real.npz --methods Ours FedAvg Multi-Krum --attacks lfp spf ooa grad-p --byz 0.1 0.2 0.3 0.4 --rounds 50

  python3 robustness_sweep.py --replot robust_nyc.csv --out robust_nyc   # just redraw



Outputs:  <out>.csv  and  <out>.png / .pdf

Deps: numpy, pandas, matplotlib, scikit-learn (+ the repo modules).

"""

import argparse

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler



ATTACK_TITLE = {"lfp": "Label-flip", "spf": "GPS-spoof",

                "ooa": "On-off", "grad-p": "Gradient-poison"}

# two-tone, CVD-safe: Ours dark, baselines lighter

BAR_COLORS = ["#0072B2", "#92C5DE", "#E69F00", "#009E73", "#CC79A7"]





def synthetic(n=100_000, fraud=0.15, seed=42):

    rng = np.random.RandomState(seed); nf = int(n * fraud); nl = n - nf

    legit = np.column_stack([rng.uniform(-74.05,-73.75,nl),rng.uniform(40.6,40.9,nl),

        rng.uniform(-74.05,-73.75,nl),rng.uniform(40.6,40.9,nl),rng.uniform(300,3600,nl),

        rng.uniform(0.5,20,nl),rng.uniform(5,80,nl),rng.randint(0,24,nl).astype(float)])

    fraud_=np.column_stack([rng.uniform(-74.05,-73.75,nf),rng.uniform(40.6,40.9,nf),

        rng.uniform(-74.05,-73.75,nf),rng.uniform(40.6,40.9,nf),rng.uniform(60,300,nf),

        rng.uniform(0.1,1.0,nf),rng.uniform(30,120,nf),rng.randint(0,24,nf).astype(float)])

    X=np.vstack([legit,fraud_]); y=np.concatenate([np.zeros(nl),np.ones(nf)])

    d=np.maximum(X[:,5],0.1); X=np.column_stack([X,X[:,6]/d,np.maximum(X[:,4],1.0)/d])

    i=rng.permutation(n); return X[i],y[i]





def load_data(path):

    if path is None: return synthetic()

    if path.endswith(".npz"):

        d = np.load(path); return d["X"].astype(float), d["y"].astype(float)

    df = pd.read_csv(path)

    y = df["label"].values.astype(float) if "label" in df.columns else df.iloc[:,-1].values.astype(float)

    X = (df.drop(columns=["label"]) if "label" in df.columns else df.iloc[:,:-1]).values.astype(float)

    return X, y





def run_sweep(a):

    from fl_base import make_validation_set

    from fl_runner import build_federation, run_federation

    from aggregators import AGGREGATOR_REGISTRY



    X, y = load_data(a.data)

    scaler = StandardScaler(); Xs = scaler.fit_transform(X)

    Xv, yv = make_validation_set(Xs, y, min(2000, len(Xs)), seed=42)



    rows = []

    for attack in a.attacks:

        for br in a.byz:

            for method in a.methods:

                server, clients, byz = build_federation(

                    Xs, y, scaler, Xv, yv,

                    n_clients=a.clients, clients_per_round=a.per_round,

                    byzantine_ratio=br, attack_type=attack,

                    noniid_alpha=0.5, seed=42)

                server.aggregator = AGGREGATOR_REGISTRY[method](n_clients=a.clients)

                hist = run_federation(server, clients, set(byz),

                                      n_rounds=a.rounds, verbose=False, seed=42)

                acc = np.mean([h["accuracy"] for h in hist[-5:]]) if hist else 0.0

                rows.append({"attack": attack, "byz": br, "method": method,

                             "accuracy": round(acc, 2)})

                print(f"  {attack:7s} byz={int(br*100):2d}% {method:11s} acc={acc:5.1f}%")

    df = pd.DataFrame(rows); df.to_csv(f"{a.out}.csv", index=False)

    print(f"wrote {a.out}.csv")

    return df





def plot_grid(df, out):

    attacks = list(dict.fromkeys(df["attack"]))

    methods = list(dict.fromkeys(df["method"]))

    byz = sorted(df["byz"].unique())

    ncol = 3 if len(attacks) > 4 else 2

    nrow = int(np.ceil(len(attacks) / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4*ncol, 2.7*nrow), squeeze=False)

    x = np.arange(len(byz)); w = 0.8 / len(methods)

    for idx, attack in enumerate(attacks):

        ax = axes[idx // ncol][idx % ncol]

        for k, meth in enumerate(methods):

            vals = [df[(df.attack==attack)&(df.byz==b)&(df.method==meth)]["accuracy"]

                    for b in byz]

            vals = [float(v.iloc[0]) if len(v) else np.nan for v in vals]

            ax.bar(x + (k-(len(methods)-1)/2)*w, vals, w,

                   color=BAR_COLORS[k % len(BAR_COLORS)], label=meth,

                   edgecolor="white", linewidth=0.5)

        ax.set_title(ATTACK_TITLE.get(attack, attack), fontsize=11)

        ax.set_xticks(x); ax.set_xticklabels([f"{int(b*100)}%" for b in byz], fontsize=9)

        ax.set_ylim(0, 100); ax.set_ylabel("Accuracy", fontsize=9)

        ax.grid(axis="y", lw=0.4, color="#dddddd"); ax.set_axisbelow(True)

        for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    for j in range(len(attacks), nrow*ncol):     # hide unused axes

        axes[j // ncol][j % ncol].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", ncol=len(methods),

               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Impact of malicious percentage", fontsize=12)

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    for ext in ("png", "pdf"):

        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")

    print(f"wrote {out}.png and .pdf")





def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--data", default=None)

    ap.add_argument("--methods", nargs="+", default=["Ours", "FedAvg"])

    ap.add_argument("--attacks", nargs="+", default=["lfp", "spf", "ooa", "grad-p"])

    ap.add_argument("--byz", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4])

    ap.add_argument("--clients", type=int, default=100)

    ap.add_argument("--per-round", type=int, default=20, dest="per_round")

    ap.add_argument("--rounds", type=int, default=50)

    ap.add_argument("--out", default="robustness")

    ap.add_argument("--replot", default=None, help="skip sweep; plot from this CSV")

    a = ap.parse_args()

    df = pd.read_csv(a.replot) if a.replot else run_sweep(a)

    plot_grid(df, a.out)





if __name__ == "__main__":

    main()