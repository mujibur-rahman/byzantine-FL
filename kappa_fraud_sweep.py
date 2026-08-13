#!/usr/bin/env python3
"""
kappa_fraud_sweep.py — sensitivity of semantic-divergence detection to class
prevalence, measured on GROUND-TRUTH labels.

For a grid of fraud rates, and for each attack, we measure the Cohen's-kappa gap
(benign - Byzantine) in the stable phase, reported as mean +/- std over seeds,
and plot gap-vs-prevalence. Uses the same FL primitives and attack application
as experiment6_semantic_divergence.py (full-model kappa on a class-balanced
D_val probe).

HONEST FRAMING (put in the caption): this uses synthetic *ground-truth* labels,
where fraud is a fixed, learnable region of feature space. Varying the fraud
rate therefore changes ONLY prevalence, isolating its effect. On real data,
raising prevalence via a feature-derived heuristic ALSO makes the class more
feature-separable (a confound); this sweep deliberately avoids that so the
prevalence effect is not entangled with label construction.

Usage:
  python3 kappa_fraud_sweep.py --out kfsweep
  python3 kappa_fraud_sweep.py --rates 0.05 0.10 0.15 0.20 0.25 \
      --seeds 42 43 44 --rounds 30 --out kfsweep
  python3 kappa_fraud_sweep.py --replot kfsweep_curves.csv --out kfsweep

Outputs: <out>_curves.csv (attack,fraud_rate,seed,kappa_gap,byz_k,ben_k),
         <out>.png/.pdf
Deps: numpy, pandas, matplotlib, scikit-learn (+ repo modules).
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score

ATTACKS = [("grad-p", "Grad-P", "#D55E00", ":"),
           ("spf",    "SPF",       "#E69F00", "--"),
           ("lfp",    "LFP",     "#0072B2", "-"),
           ("ooa",    "OOA",             "#009E73", "-.")]


def synthetic(n, fraud, seed):
    """Ground-truth labels: fraud is a FIXED distinct region of feature space,
    so it stays learnable at any prevalence — only the fraction changes."""
    rng = np.random.RandomState(seed)
    nf = int(n * fraud); nl = n - nf
    legit = np.column_stack([
        rng.uniform(-74.05, -73.75, nl), rng.uniform(40.6, 40.9, nl),
        rng.uniform(-74.05, -73.75, nl), rng.uniform(40.6, 40.9, nl),
        rng.uniform(300, 3600, nl), rng.uniform(0.5, 20, nl),
        rng.uniform(5, 80, nl), rng.randint(0, 24, nl).astype(float)])
    fr = np.column_stack([
        rng.uniform(-74.05, -73.75, nf), rng.uniform(40.6, 40.9, nf),
        rng.uniform(-74.05, -73.75, nf), rng.uniform(40.6, 40.9, nf),
        rng.uniform(60, 300, nf), rng.uniform(0.1, 1.0, nf),
        rng.uniform(30, 120, nf), rng.randint(0, 24, nf).astype(float)])
    X = np.vstack([legit, fr]); y = np.concatenate([np.zeros(nl), np.ones(nf)])
    d = np.maximum(X[:, 5], 0.1)
    X = np.column_stack([X, X[:, 6] / d, np.maximum(X[:, 4], 1.0) / d])
    i = rng.permutation(n)
    return X[i], y[i]


def balanced_probe(Xv, yv, size, seed=0):
    rng = np.random.RandomState(seed)
    pos = np.where(yv == 1)[0]; neg = np.where(yv == 0)[0]; half = size // 2
    def draw(p, k):
        return rng.choice(p, k, replace=len(p) < k) if len(p) else np.empty(0, int)
    idx = np.concatenate([draw(pos, half), draw(neg, size - half)])
    rng.shuffle(idx)
    return Xv[idx]


def kappa_gap(fraud, attack, seed, cfg):
    """Replicate exp6's per-round divergence, return (kappa_gap, byz_k, ben_k)
    over the stable phase (last 10 rounds)."""
    from fl_model import FLNeuralNet
    from fl_base import (partition_noniid, make_validation_set,
                         attack_label_flip, attack_gradient_poison,
                         attack_gps_spoof, is_malicious_round)
    np.random.seed(seed)
    X, y = synthetic(cfg["n"], fraud, seed)
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    Xv, yv = make_validation_set(Xs, y, min(2000, len(Xs)), seed=seed)
    Xk = balanced_probe(Xv, yv, cfg["kval"], seed=0)
    clients = partition_noniid(Xs, y, cfg["nc"], alpha=0.5, seed=seed)
    nb = max(1, int(cfg["nc"] * 0.2))
    byz_ids = set(range(cfg["nc"] - nb, cfg["nc"]))
    ben_ids = list(range(cfg["nc"] - nb))
    nfeat = Xs.shape[1]

    gm = FLNeuralNet(nfeat); gp = gm.get_params()
    byz_ks, ben_ks = [], []

    for rnd in range(cfg["rounds"]):
        rng = np.random.RandomState(seed + rnd)
        nbr = max(1, int(cfg["cpr"] * 0.2)); nbn = cfg["cpr"] - nbr
        sb = rng.choice(ben_ids, min(nbn, len(ben_ids)), replace=False).tolist()
        sm = rng.choice(list(byz_ids), min(nbr, len(byz_ids)), replace=False).tolist()
        sampled = sb + sm
        updates, isbyz = [], []
        for cid in sampled:
            Xc, yc = clients[cid]
            if len(Xc) == 0:
                continue
            ib = cid in byz_ids
            if ib:
                if attack == 'lfp':
                    yc = attack_label_flip(yc, flip_rate=0.30)
                elif attack == 'spf':
                    Xc = attack_gps_spoof(Xc, spoof_rate=0.30)
                elif attack == 'ooa' and not is_malicious_round(rnd):
                    ib = False
            m = FLNeuralNet(nfeat); m.set_params(gp.copy())
            before = m.get_params()
            m.train(Xc, yc, lr=0.01, epochs=5, batch_size=32)
            u = m.get_params() - before
            if ib and attack == 'grad-p':
                dw2, db2 = attack_gradient_poison(u[:-1], u[-1], scale=3.0)
                u = np.append(dw2, db2)
            updates.append(u); isbyz.append(ib)
        if not updates:
            continue
        # per-client full-model kappa on the balanced probe (stable phase only)
        if rnd >= cfg["rounds"] - 10:
            base = FLNeuralNet(nfeat); base.set_params(gp); pg = base.predict(Xk)
            for u, ib in zip(updates, isbyz):
                cm = FLNeuralNet(nfeat); cm.set_params(gp + u); pc = cm.predict(Xk)
                if len(np.unique(pc)) > 1 and len(np.unique(pg)) > 1:
                    k = cohen_kappa_score(pg, pc)
                else:
                    k = 1.0 if np.array_equal(pc, pg) else 0.0
                (byz_ks if ib else ben_ks).append(k)
        gp = gp + 0.01 * np.mean(updates, axis=0)

    bk = float(np.mean(byz_ks)) if byz_ks else np.nan
    nk = float(np.mean(ben_ks)) if ben_ks else np.nan
    return nk - bk, bk, nk


def run(a):
    cfg = dict(n=a.n, nc=a.clients, cpr=a.per_round, rounds=a.rounds, kval=a.kval)
    rows = []
    for fr in a.rates:
        for attack, _, _, _ in ATTACKS:
            for seed in a.seeds:
                gap, bk, nk = kappa_gap(fr, attack, seed, cfg)
                rows.append({"attack": attack, "fraud_rate": fr, "seed": seed,
                             "kappa_gap": gap, "byz_k": bk, "ben_k": nk})
            sub = [r for r in rows if r["attack"] == attack and r["fraud_rate"] == fr]
            gaps = [r["kappa_gap"] for r in sub]
            print(f"  fraud={fr:.0%}  {attack:6s}  κ gap={np.nanmean(gaps):+.3f}"
                  f"±{np.nanstd(gaps):.3f}")
    df = pd.DataFrame(rows); df.to_csv(f"{a.out}_curves.csv", index=False)
    print(f"wrote {a.out}_curves.csv")
    return df


def plot(df, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for attack, label, color, ls in ATTACKS:
        d = df[df.attack == attack]
        g = d.groupby("fraud_rate")["kappa_gap"]
        m = g.mean(); s = g.std(ddof=1).fillna(0.0)
        xs = m.index.values * 100
        lw = 2.8 if attack == "lfp" else 1.8
        ax.plot(xs, m.values, ls, color=color, lw=lw, label=label,
                marker="o", ms=4, zorder=6 if attack == "lfp" else 3)
        ax.fill_between(xs, m.values - s.values, m.values + s.values,
                        color=color, alpha=0.15, lw=0)
    ax.axhline(0, ls="-", lw=0.8, color="#888888")
    ax.set_xlabel("Fraud prevalence (%)", fontsize=15)
    ax.set_ylabel("Cohen's $\\kappa$ gap  (Benign $-$ Byzantine)", fontsize=15)
    #ax.set_title("Semantic-divergence detection vs. class prevalence "
    #             "(ground-truth labels)", fontsize=12)
    ax.grid(True, lw=0.4, color="#e8e8e8"); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=12.5, title="Attack")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=220, bbox_inches="tight")
    print(f"wrote {out}.png and .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", nargs="+", type=float,
                    default=[0.05, 0.10, 0.15, 0.20])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--n", type=int, default=50000,
                    help="use >=50000 so clients have enough data to learn the "
                         "boundary; the kappa separation only appears once the "
                         "minority class is actually learned (use 100000 to "
                         "match experiment6 exactly)")
    ap.add_argument("--clients", type=int, default=100)
    ap.add_argument("--per-round", type=int, default=20, dest="per_round")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--kval", type=int, default=500)
    ap.add_argument("--out", default="kfsweep")
    ap.add_argument("--replot", default=None)
    a = ap.parse_args()
    d = os.path.dirname(a.out)
    if d:
        os.makedirs(d, exist_ok=True)
    df = pd.read_csv(a.replot) if a.replot else run(a)
    plot(df, a.out)


if __name__ == "__main__":
    main()
