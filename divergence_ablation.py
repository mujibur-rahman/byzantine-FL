#!/usr/bin/env python3
"""
divergence_ablation.py
======================
Self-contained ablation: which divergence measure best detects Byzantine
clients from *semantic* (prediction-space) divergence in federated learning?

Compares, as Byzantine-detection signals between each client's model and the
global model on a server-held validation set D_val:

  - Cohen's kappa           : agreement on HARD predictions  (lower => suspicious)
  - KL divergence           : D_KL(client || global) on soft Bernoulli outputs
  - Symmetric KL (Jeffreys) : 0.5*(KL(c||g)+KL(g||c))
  - Renyi(alpha)            : D_alpha(client || global), alpha in {0.5, 2, inf}
                              (alpha->1 == KL; inf == worst-case log-ratio)
  (higher divergence => more suspicious)

Reported per attack type: detection AUC (Byzantine vs benign), averaged over
seeds (mean +/- std). Writes a CSV and a LaTeX table.

Dependencies:  numpy, pandas, scikit-learn   (pip install numpy pandas scikit-learn)

Usage:
  python3 divergence_ablation.py                       # synthetic NYC-taxi-like data
  python3 divergence_ablation.py --data nyc.npz        # real data: npz with X,y
  python3 divergence_ablation.py --data nyc.csv        # real data: csv, label col last or 'label'
  python3 divergence_ablation.py --seeds 5 --rounds 30 --clients 100 --byz 0.20
  python3 divergence_ablation.py --out results_nyc     # -> results_nyc.csv / .tex

Real-data note: X must be (n, 10) numeric features, y binary (0=legit,1=fraud),
matching the paper's feature space. Any dataset works; only NYC-taxi framing
is cosmetic.
"""

import argparse, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, roc_auc_score

EPS = 1e-6


# ────────────────────────────────────────────────────────────────────────────
# Model — faithful port of the paper's FLNeuralNet (10->64->128->64->1, Adam)
# ────────────────────────────────────────────────────────────────────────────
class MLP:
    HIDDEN = [64, 128, 64]

    def __init__(self, n_features=10, seed=None):
        if seed is not None:
            np.random.seed(seed)
        sizes = [n_features] + self.HIDDEN + [1]
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(np.random.uniform(-lim, lim, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    @staticmethod
    def _relu(x): return np.maximum(0, x)

    @staticmethod
    def _sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _forward(self, X):
        acts = [X]; a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            a = self._relu(z) if i < len(self.W) - 1 else self._sig(z)
            acts.append(a)
        return acts

    def _backward(self, acts, y):
        n = len(y)
        gW = [None] * len(self.W); gb = [None] * len(self.b)
        delta = ((acts[-1].flatten() - y) / n).reshape(-1, 1)
        gW[-1] = acts[-2].T @ delta; gb[-1] = delta.sum(0)
        for i in range(len(self.W) - 2, -1, -1):
            delta = (delta @ self.W[i + 1].T) * (acts[i + 1] > 0)
            gW[i] = acts[i].T @ delta; gb[i] = delta.sum(0)
        return gW, gb

    def _adam(self, gW, gb, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1; t = self._t
        for i in range(len(self.W)):
            self._mW[i] = b1 * self._mW[i] + (1 - b1) * gW[i]
            self._vW[i] = b2 * self._vW[i] + (1 - b2) * gW[i] ** 2
            self.W[i] -= lr * (self._mW[i] / (1 - b1 ** t)) / (np.sqrt(self._vW[i] / (1 - b2 ** t)) + eps)
            self._mb[i] = b1 * self._mb[i] + (1 - b1) * gb[i]
            self._vb[i] = b2 * self._vb[i] + (1 - b2) * gb[i] ** 2
            self.b[i] -= lr * (self._mb[i] / (1 - b1 ** t)) / (np.sqrt(self._vb[i] / (1 - b2 ** t)) + eps)

    def train(self, X, y, lr=0.01, epochs=5, batch=32):
        n = len(X)
        for _ in range(epochs):
            idx = np.random.permutation(n)
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                acts = self._forward(X[bi])
                gW, gb = self._backward(acts, y[bi])
                self._adam(gW, gb, lr)

    def proba(self, X): return self._forward(X)[-1].flatten()
    def predict(self, X): return (self.proba(X) >= 0.5).astype(int)
    def get_params(self):
        return np.concatenate([p.flatten() for WB in zip(self.W, self.b) for p in WB])
    def set_params(self, v):
        i = 0
        for k in range(len(self.W)):
            ws = self.W[k].size; bs = self.b[k].size
            self.W[k] = v[i:i + ws].reshape(self.W[k].shape); i += ws
            self.b[k] = v[i:i + bs]; i += bs


# ────────────────────────────────────────────────────────────────────────────
# Data, partitioning, attacks
# ────────────────────────────────────────────────────────────────────────────
def synthetic_data(n=100_000, fraud_rate=0.15, seed=42):
    rng = np.random.RandomState(seed)
    nf = int(n * fraud_rate); nl = n - nf
    legit = np.column_stack([
        rng.uniform(-74.05, -73.75, nl), rng.uniform(40.60, 40.90, nl),
        rng.uniform(-74.05, -73.75, nl), rng.uniform(40.60, 40.90, nl),
        rng.uniform(300, 3600, nl), rng.uniform(0.5, 20, nl),
        rng.uniform(5, 80, nl), rng.randint(0, 24, nl).astype(float)])
    fraud = np.column_stack([
        rng.uniform(-74.05, -73.75, nf), rng.uniform(40.60, 40.90, nf),
        rng.uniform(-74.05, -73.75, nf), rng.uniform(40.60, 40.90, nf),
        rng.uniform(60, 300, nf), rng.uniform(0.1, 1.0, nf),
        rng.uniform(30, 120, nf), rng.randint(0, 24, nf).astype(float)])
    X = np.vstack([legit, fraud]); y = np.concatenate([np.zeros(nl), np.ones(nf)])
    dist = np.maximum(X[:, 5], 0.1); dur = np.maximum(X[:, 4], 1.0)
    X = np.column_stack([X, X[:, 6] / dist, dur / dist])
    idx = rng.permutation(n)
    return X[idx], y[idx]


def load_data(path):
    if path.endswith(".npz"):
        d = np.load(path); return d["X"].astype(float), d["y"].astype(float)
    df = pd.read_csv(path)
    if "label" in df.columns:
        y = df["label"].values.astype(float); X = df.drop(columns=["label"]).values.astype(float)
    else:
        y = df.iloc[:, -1].values.astype(float); X = df.iloc[:, :-1].values.astype(float)
    return X, y


def partition_noniid(X, y, n_clients, alpha=0.5, seed=42):
    rng = np.random.RandomState(seed)
    parts = [[] for _ in range(n_clients)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        prop = rng.dirichlet(alpha * np.ones(n_clients))
        cuts = (np.cumsum(prop) * len(idx)).astype(int)[:-1]
        for i, s in enumerate(np.split(idx, cuts)):
            parts[i].append(s)
    out = []
    for i in range(n_clients):
        idx = np.concatenate(parts[i]) if parts[i] else np.array([], int)
        rng.shuffle(idx); out.append((X[idx], y[idx]))
    return out


def attack_label_flip(y, rate=0.30):
    y = y.copy(); n = int(len(y) * rate)
    idx = np.random.choice(len(y), n, replace=False); y[idx] = 1 - y[idx]; return y

def attack_gps_spoof(X, rate=0.30):
    X = X.copy(); n = int(len(X) * rate)
    idx = np.random.choice(len(X), n, replace=False)
    for col, (lo, hi) in zip(range(4), [(-80, -70), (35, 45), (-80, -70), (35, 45)]):
        X[idx, col] = np.random.uniform(lo, hi, n)
    return X

def attack_gradient_poison(upd, scale=3.0):
    return upd + np.random.randn(*upd.shape) * scale * np.linalg.norm(upd)

def is_malicious_round(r, period=5): return (r // period) % 2 == 1


# ────────────────────────────────────────────────────────────────────────────
# Divergence measures  (signal convention: higher => more suspicious)
# ────────────────────────────────────────────────────────────────────────────
def sig_kappa(pg_hard, pc_hard):
    if len(np.unique(pg_hard)) < 2 or len(np.unique(pc_hard)) < 2:
        k = 1.0 if np.array_equal(pg_hard, pc_hard) else 0.0
    else:
        k = float(cohen_kappa_score(pg_hard, pc_hard))
    return -k

def _kl(p, q):   # D_KL(P||Q), per-sample Bernoulli, mean
    p = np.clip(p, EPS, 1 - EPS); q = np.clip(q, EPS, 1 - EPS)
    return float((p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))).mean())

def sig_kl(pg, pc):       return _kl(pc, pg)                      # client || global
def sig_kl_sym(pg, pc):   return 0.5 * (_kl(pc, pg) + _kl(pg, pc))

def sig_renyi(pg, pc, a):
    p = np.clip(pc, EPS, 1 - EPS); q = np.clip(pg, EPS, 1 - EPS)
    if np.isinf(a):                                              # D_inf = log max ratio
        return float(np.log(np.maximum(p / q, (1 - p) / (1 - q))).mean())
    inner = p ** a * q ** (1 - a) + (1 - p) ** a * (1 - q) ** (1 - a)
    return float((1.0 / (a - 1.0) * np.log(np.clip(inner, EPS, None))).mean())


MEASURES = {
    "kappa":     lambda gh, gp, ch, cp: sig_kappa(gh, ch),
    "KL":        lambda gh, gp, ch, cp: sig_kl(gp, cp),
    "KL_sym":    lambda gh, gp, ch, cp: sig_kl_sym(gp, cp),
    "Renyi_0.5": lambda gh, gp, ch, cp: sig_renyi(gp, cp, 0.5),
    "Renyi_2":   lambda gh, gp, ch, cp: sig_renyi(gp, cp, 2.0),
    "Renyi_inf": lambda gh, gp, ch, cp: sig_renyi(gp, cp, np.inf),
}


# ────────────────────────────────────────────────────────────────────────────
# One FL run for one attack + seed -> per-measure AUC
# ────────────────────────────────────────────────────────────────────────────
def run(attack, X, y, cfg, seed):
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    rng0 = np.random.RandomState(seed)
    vidx = rng0.choice(len(Xs), min(cfg.val, len(Xs)), replace=False)
    Xv = Xs[vidx]
    clients = partition_noniid(Xs, y, cfg.clients, alpha=0.5, seed=seed)
    nbe = cfg.clients - int(cfg.clients * cfg.byz)
    byz = set(range(nbe, cfg.clients)); ben = list(range(nbe))
    nf = Xs.shape[1]
    per_round = cfg.per_round
    nb = max(1, int(per_round * cfg.byz)); ng = per_round - nb

    gp = MLP(nf, seed=seed).get_params()
    acc = {m: {i: [] for i in range(cfg.clients)} for m in MEASURES}
    for rnd in range(cfg.rounds):
        rng = np.random.RandomState(seed * 1000 + rnd)
        s = (rng.choice(ben, min(ng, len(ben)), replace=False).tolist()
             + rng.choice(list(byz), min(nb, len(byz)), replace=False).tolist())
        gm = MLP(nf); gm.set_params(gp.copy())
        gh, gpr = gm.predict(Xv), gm.proba(Xv)
        ups = []
        for cid in s:
            Xc, yc = clients[cid]
            if len(Xc) == 0: continue
            isb = cid in byz
            if isb and attack == "lfp": yc = attack_label_flip(yc)
            if isb and attack == "spf": Xc = attack_gps_spoof(Xc)
            if isb and attack == "ooa" and not is_malicious_round(rnd): isb = False
            m = MLP(nf); m.set_params(gp.copy())
            m.train(Xc, yc, lr=cfg.lr, epochs=5, batch=32)
            upd = m.get_params() - gp
            if (cid in byz) and attack == "grad-p":
                upd = attack_gradient_poison(upd); m.set_params(gp + upd)
            ch, cp = m.predict(Xv), m.proba(Xv)
            for mname, fn in MEASURES.items():
                acc[mname][cid].append(fn(gh, gpr, ch, cp))
            ups.append(upd)
        if ups: gp = gp + cfg.lr * np.mean(ups, axis=0)

    ids = [i for i in range(cfg.clients) if acc["kappa"][i]]
    ytrue = np.array([1 if i in byz else 0 for i in ids])
    aucs = {}
    for mname in MEASURES:
        score = np.array([np.mean(acc[mname][i]) for i in ids])
        aucs[mname] = roc_auc_score(ytrue, score) if len(np.unique(ytrue)) > 1 else np.nan
    return aucs


# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="npz (X,y) or csv; omit for synthetic")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--clients", type=int, default=100)
    ap.add_argument("--per-round", type=int, default=20, dest="per_round")
    ap.add_argument("--byz", type=float, default=0.20)
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out", default="divergence_ablation")
    cfg = ap.parse_args()

    if cfg.data:
        X, y = load_data(cfg.data); src = cfg.data
    else:
        X, y = synthetic_data(seed=42); src = "synthetic (NYC-taxi-like)"
    print(f"Data: {src}   X={X.shape}  fraud_rate={y.mean():.1%}")
    print(f"Config: clients={cfg.clients} per_round={cfg.per_round} "
          f"byz={cfg.byz:.0%} rounds={cfg.rounds} seeds={cfg.seeds}\n")

    attacks = ["lfp", "spf", "ooa", "grad-p"]
    rows = []
    for atk in attacks:
        runs = [run(atk, X, y, cfg, seed=42 + s) for s in range(cfg.seeds)]
        for mname in MEASURES:
            vals = np.array([r[mname] for r in runs], float)
            rows.append({"attack": atk, "measure": mname,
                         "AUC_mean": round(np.nanmean(vals), 3),
                         "AUC_std": round(np.nanstd(vals), 3)})
        best = max(MEASURES, key=lambda m: np.nanmean([r[m] for r in runs]))
        print(f"[{atk:7s}] " + "  ".join(
            f"{m}={np.nanmean([r[m] for r in runs]):.3f}" for m in MEASURES)
            + f"   -> best: {best}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{cfg.out}.csv", index=False)

    # LaTeX (AUC mean +/- std, measures as columns)
    piv = df.pivot(index="attack", columns="measure", values="AUC_mean").reindex(attacks)
    cols = list(MEASURES)
    with open(f"{cfg.out}.tex", "w") as f:
        f.write("\\begin{table}[t]\\centering\\small\n")
        f.write("\\caption{Divergence-measure ablation: Byzantine-detection AUC "
                "(higher is better) per attack type.}\n\\label{tab:divergence_ablation}\n")
        f.write("\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n")
        f.write("Attack & " + " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\\n\\midrule\n")
        for atk in attacks:
            best = piv.loc[atk].idxmax()
            cells = []
            for c in cols:
                v = piv.loc[atk, c]
                cells.append(f"\\textbf{{{v:.3f}}}" if c == best else f"{v:.3f}")
            f.write(f"{atk} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"\nWrote {cfg.out}.csv and {cfg.out}.tex")
    print("Reminder: run with --data <real NYC-TLC X,y> before using in the paper.")


if __name__ == "__main__":
    main()
