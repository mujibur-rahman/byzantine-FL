#!/usr/bin/env python3
"""
convergence_experiment.py — convergence speed of the aggregation algorithms.

Runs the full FL pipeline for every aggregator and records the global model's
accuracy/F1 at EACH round, then reports convergence-speed metrics and a
convergence-curve figure. Run either clean (--attack none) or under attack.

Metrics per method:
  final_acc      : mean accuracy over the last 5 rounds
  round@90%      : first round reaching 90% of that method's own final_acc
                   (relative convergence speed -- how fast it saturates)
  round@thresh   : first round reaching an absolute accuracy threshold (--thresh)
  auc            : mean accuracy across all rounds (normalised area under the
                   curve) -- rewards reaching high accuracy early and staying

RUN FROM THE REPO DIRECTORY (uses build_federation / AGGREGATOR_REGISTRY).

Usage:
  python3 convergence_experiment.py --data nyc_real100000.csv --out conv_nyc
  python3 convergence_experiment.py --data nyc_real100000.csv --attack lfp --byz 0.20 --out conv_nyc_lfp
  python3 convergence_experiment.py --replot conv_nyc_curves.csv --out conv_nyc   # redraw only

Outputs: <out>_curves.csv (method,round,acc,f1), <out>_summary.csv, <out>.png/.pdf
Deps: numpy, pandas, matplotlib, scikit-learn (+ repo modules).
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# fixed method order + CVD-safe styling; Ours emphasised
STYLE = {
    "FedAvg":      ("#999999", "-"),
    "TrimmedMean": ("#56B4E9", "-"),
    "RFVIR":       ("#009E73", "-"),
    "Multi-Krum":  ("#E69F00", "-"),
    "Bulyan":      ("#D55E00", "-"),
    "FLAME":       ("#CC79A7", "-"),
    "Ours":        ("#0072B2", "-"),
}


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


def run_curves(a):
    from fl_base import make_validation_set
    from fl_runner import build_federation, run_federation
    from aggregators import AGGREGATOR_REGISTRY
    X, y = load_data(a.data)
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    Xv, yv = make_validation_set(Xs, y, min(2000, len(Xs)), seed=42)
    attack = None if a.attack in (None, "none") else a.attack
    br = 0.0 if attack is None else a.byz

    rows = []
    for method in a.methods:
        # Reset the global RNG so every method starts from the SAME model
        # init and client partition — otherwise FLNeuralNet's init (which draws
        # from global numpy RNG) differs per method and the curves aren't
        # comparable (a method can "lose" purely on a worse random start).
        np.random.seed(42)
        server, clients, byz = build_federation(
            Xs, y, scaler, Xv, yv, n_clients=a.clients,
            clients_per_round=a.per_round, byzantine_ratio=br,
            attack_type=attack, noniid_alpha=0.5, seed=42)
        server.aggregator = AGGREGATOR_REGISTRY[method](n_clients=a.clients)
        hist = run_federation(server, clients, set(byz), n_rounds=a.rounds,
                              verbose=False, seed=42)
        for h in hist:
            rows.append({"method": method, "round": h["round"],
                         "acc": h["accuracy"], "f1": h["f1"]})
        fa = np.mean([h["accuracy"] for h in hist[-5:]]) if hist else 0.0
        print(f"  {method:12s} final_acc={fa:5.1f}%  ({len(hist)} rounds)")
    df = pd.DataFrame(rows); df.to_csv(f"{a.out}_curves.csv", index=False)
    print(f"wrote {a.out}_curves.csv")
    return df


def summarise(df, out, thresh):
    rows = []
    for m in [x for x in STYLE if x in df["method"].unique()]:
        sub = df[df.method == m].sort_values("round")
        acc = sub["acc"].values; rnd = sub["round"].values
        if len(acc) == 0:
            continue
        final = float(np.mean(acc[-5:]))
        # first round reaching 90% of own final
        tgt = 0.90 * final
        r90 = next((int(rnd[i]) for i in range(len(acc)) if acc[i] >= tgt), None)
        # first round reaching absolute threshold
        rth = next((int(rnd[i]) for i in range(len(acc)) if acc[i] >= thresh), None)
        auc = float(np.mean(acc))
        rows.append({"method": m, "final_acc": round(final, 1),
                     "round@90%": r90, f"round@{thresh:.0f}%": rth,
                     "auc": round(auc, 1)})
    s = pd.DataFrame(rows); s.to_csv(f"{out}_summary.csv", index=False)
    print("\n" + s.to_string(index=False)); print(f"\nwrote {out}_summary.csv")
    return s


def plot(df, out, thresh):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for m in STYLE:
        sub = df[df.method == m].sort_values("round")
        if sub.empty: continue
        color, ls = STYLE[m]
        lw = 2.8 if m == "Ours" else 1.6
        z = 6 if m == "Ours" else 3
        ax.plot(sub["round"], sub["acc"], ls, color=color, lw=lw, label=m, zorder=z)
    ax.axhline(thresh, ls=":", lw=1, color="#888888")
    ax.text(df["round"].max(), thresh + 0.6, f"{thresh:.0f}% threshold",
            ha="right", fontsize=8, color="#888888")
    ax.set_xlabel("Federated round"); ax.set_ylabel("Global model accuracy (%)")
    ax.set_title("Convergence speed by aggregation algorithm", fontsize=12)
    ax.grid(True, lw=0.4, color="#e8e8e8"); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower right")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=220, bbox_inches="tight")
    print(f"wrote {out}.png and .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--methods", nargs="+",
                    default=["FedAvg", "Multi-Krum", "TrimmedMean", "Bulyan",
                             "RFVIR", "FLAME", "Ours"])
    ap.add_argument("--attack", default="none", help="none|lfp|spf|ooa|grad-p")
    ap.add_argument("--byz", type=float, default=0.20)
    ap.add_argument("--clients", type=int, default=100)
    ap.add_argument("--per-round", type=int, default=20, dest="per_round")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--thresh", type=float, default=85.0, help="absolute acc threshold %")
    ap.add_argument("--out", default="convergence")
    ap.add_argument("--replot", default=None)
    a = ap.parse_args()
    d = os.path.dirname(a.out)
    if d: os.makedirs(d, exist_ok=True)
    df = pd.read_csv(a.replot) if a.replot else run_curves(a)
    summarise(df, a.out, a.thresh)
    plot(df, a.out, a.thresh)


if __name__ == "__main__":
    main()
