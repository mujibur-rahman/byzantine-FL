#!/usr/bin/env python3
"""
ablation_layers.py — layer ablation of the proposed framework. Remove each
defence layer (outlier / trust / Kappa) and measure accuracy + F1 UNDER ATTACK.
Scoped for a conference budget: one dataset, a couple of attacks, a few seeds,
mean±std.

Also tests the SEPARATION-GATED Kappa (aggregators.MultiLayerAggregator,
kappa_gate). Run twice to make the point:
  * on REAL heuristic-labeled data the gate should DEFER -> Full ≈ -Kappa
    (Kappa does no harm);
  * on SYNTHETIC ground-truth labels (omit --data) the gate should ENGAGE ->
    Full > -Kappa (Kappa helps when a real semantic signal exists).

Configs: Full / -Outlier / -Trust / -Kappa.

Usage:
  python3 ablation_layers.py --data nyc_real100000.csv --attacks lfp grad-p --out ablation_nyc          # real (gate on)
  python3 ablation_layers.py --attacks lfp grad-p --out ablation_synth                                  # synthetic ground-truth (gate on)
  python3 ablation_layers.py --data nyc_real100000.csv --no-gate --out ablation_nyc_ungated             # old behavior
Outputs: <out>.csv, <out>_summary.csv, <out>_latex.txt
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ORDER = ["Full", "-Outlier", "-Trust", "-Kappa"]
TOGGLE = {
    "Full":     dict(use_outlier=True,  use_trust=True,  use_kappa=True),
    "-Outlier": dict(use_outlier=False, use_trust=True,  use_kappa=True),
    "-Trust":   dict(use_outlier=True,  use_trust=False, use_kappa=True),
    "-Kappa":   dict(use_outlier=True,  use_trust=True,  use_kappa=False),
}


def synthetic(n=50000, fraud=0.17, seed=42):
    """Ground-truth labels: fraud is a fixed learnable feature region."""
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
    from aggregators import MultiLayerAggregator
    X, y = load_data(a.data)
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    Xv, yv = make_validation_set(Xs, y, min(2000, len(Xs)), seed=42)
    tag = "synthetic(ground-truth)" if a.data is None else a.data
    print(f"data={tag}  kappa_gate={'on' if a.gate else 'off'}")

    rows = []
    for attack in a.attacks:
        for cfg in ORDER:
            for seed in a.seeds:
                np.random.seed(seed)
                server, clients, byz = build_federation(
                    Xs, y, scaler, Xv, yv, n_clients=a.clients,
                    clients_per_round=a.per_round, byzantine_ratio=a.byz,
                    attack_type=attack, noniid_alpha=0.5, seed=seed)
                server.aggregator = MultiLayerAggregator(
                    n_clients=a.clients, kappa_gate=a.gate, **TOGGLE[cfg])
                hist = run_federation(server, clients, set(byz),
                                      n_rounds=a.rounds, verbose=False, seed=seed)
                acc = np.mean([h["accuracy"] for h in hist[-5:]]) if hist else 0.0
                f1 = np.mean([h["f1"] for h in hist[-5:]]) if hist else 0.0
                rows.append({"config": cfg, "attack": attack, "seed": seed,
                             "acc": acc, "f1": f1})
            sub = [r for r in rows if r["config"] == cfg and r["attack"] == attack]
            print(f"  {attack:7s} {cfg:9s} acc={np.mean([r['acc'] for r in sub]):5.1f}"
                  f"  f1={np.mean([r['f1'] for r in sub]):5.1f}")
    df = pd.DataFrame(rows); df.to_csv(f"{a.out}.csv", index=False)
    print(f"wrote {a.out}.csv")
    return df


def summarise(df, out):
    rows = []
    for attack in df["attack"].unique():
        fa = df[(df.attack == attack) & (df.config == "Full")]["acc"].mean()
        ff = df[(df.attack == attack) & (df.config == "Full")]["f1"].mean()
        for cfg in ORDER:
            s = df[(df.attack == attack) & (df.config == cfg)]
            am, as_ = s["acc"].mean(), s["acc"].std(ddof=0)
            fm, fs = s["f1"].mean(), s["f1"].std(ddof=0)
            da = "" if cfg == "Full" else f" ({am-fa:+.1f})"
            dfd = "" if cfg == "Full" else f" ({fm-ff:+.1f})"
            rows.append({"attack": attack, "config": cfg,
                         "acc": f"{am:.1f}±{as_:.1f}{da}", "f1": f"{fm:.1f}±{fs:.1f}{dfd}"})
    s = pd.DataFrame(rows); s.to_csv(f"{out}_summary.csv", index=False)
    print("\n" + s.to_string(index=False)); print(f"\nwrote {out}_summary.csv")

    NL = r" \\"
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize",
         r"\caption{Layer ablation (20\% Byzantine, mean$\pm$s.d.\ over "
         + str(df["seed"].nunique()) + r" seeds). $\Delta$ vs.\ Full in "
         r"parentheses. Separation-gated Kappa.}",
         r"\label{table:accuracy_ablation}",
         r"\begin{tabular}{llcc}", r"\hline",
         r"\textbf{Attack} & \textbf{Configuration} & \textbf{Acc.\ (\%)} & \textbf{F1 (\%)}" + NL,
         r"\hline"]
    for attack in df["attack"].unique():
        for cfg in ORDER:
            r = next(x for x in rows if x["attack"] == attack and x["config"] == cfg)
            lbl = attack if cfg == "Full" else ""
            L.append(f"{lbl} & {cfg} & {r['acc']} & {r['f1']}".replace("±", r"$\pm$") + NL)
        L.append(r"\hline")
    L += [r"\end{tabular}", r"\end{table}"]
    open(f"{out}_latex.txt", "w").write("\n".join(L) + "\n")
    print(f"wrote {out}_latex.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="omit for synthetic ground-truth labels")
    ap.add_argument("--attacks", nargs="+", default=["lfp", "grad-p"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--byz", type=float, default=0.20)
    ap.add_argument("--clients", type=int, default=100)
    ap.add_argument("--per-round", type=int, default=20, dest="per_round")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--gate", dest="gate", action="store_true", default=True,
                    help="separation-gated Kappa (default on)")
    ap.add_argument("--no-gate", dest="gate", action="store_false",
                    help="old always-on peer-relative Kappa")
    ap.add_argument("--out", default="ablation")
    ap.add_argument("--replot", default=None)
    a = ap.parse_args()
    d = os.path.dirname(a.out)
    if d:
        os.makedirs(d, exist_ok=True)
    df = pd.read_csv(a.replot) if a.replot else run(a)
    summarise(df, a.out)


if __name__ == "__main__":
    main()
