"""
experiment2_sensitivity.py
==========================
Experiment 2: Sensitivity analysis over τ, α, δ
Answers INFOCOM R2/R3: "how robust are results to parameter choices?"

Grid:
  τ (outlier threshold z-score) : [1.0, 1.5, 2.0, 2.5, 3.0]
  α (trust decay factor)        : [0.5, 0.6, 0.7, 0.8, 0.9]
  δ (Kappa threshold)           : [0.1, 0.15, 0.2, 0.25, 0.3]

One param varied at a time; others held at paper defaults:
  τ=2.0, α=0.8, δ=0.2

Attack: label flipping (lfp) at 20% Byzantine — the hardest case
        for semantic detection.

Run:  python3 experiment2_sensitivity.py
Output: results/sensitivity_tau.csv
        results/sensitivity_alpha.csv
        results/sensitivity_delta.csv
        results/sensitivity_summary.csv (for LaTeX table)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from dataset_config import get_dataset, DATASET_NAME, tag, results_dir
from fl_base import (generate_sc_dataset, partition_noniid,
                     make_validation_set, FLModel,
                     attack_label_flip, attack_gradient_poison,
                     attack_gps_spoof, is_malicious_round, compute_kappa)
from aggregators import MultiLayerAggregator, fedavg

os.makedirs(results_dir(), exist_ok=True)

# ── Shared config ────────────────────────────────────────────────────────────
N_CLIENTS        = 200
CLIENTS_PER_ROUND= 20
N_ROUNDS         = 40          # shorter for sensitivity sweep
LR               = 0.01
NONIID_ALPHA     = 0.5
BYZANTINE_RATIO  = 0.20
VAL_SIZE         = 2000
SEED             = 42
ATTACK           = 'lfp'      # label flipping — hardest for semantic detection

# Defaults
TAU_DEFAULT   = 2.0
ALPHA_DEFAULT = 0.8
DELTA_DEFAULT = 0.2

np.random.seed(SEED)

print("Preparing data for sensitivity analysis...")
X, y = get_dataset(seed=SEED)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_val, y_val = make_validation_set(X_scaled, y, VAL_SIZE, seed=SEED)

clients = partition_noniid(X_scaled, y, N_CLIENTS,
                           alpha=NONIID_ALPHA, seed=SEED)
n_byzantine = int(N_CLIENTS * BYZANTINE_RATIO)
n_benign    = N_CLIENTS - n_byzantine
byzantine_ids = list(range(n_benign, N_CLIENTS))
benign_ids    = list(range(n_benign))
n_features    = X_scaled.shape[1]


def run_ours(tau, alpha, delta, n_rounds=N_ROUNDS):
    """Run multi-layer aggregator with given hyperparams. Return final acc & F1."""
    global_model  = FLModel(n_features)
    global_params = global_model.get_params()
    aggregator    = MultiLayerAggregator(
        N_CLIENTS, tau=tau, alpha=alpha, delta=delta)

    accs = []
    for rnd in range(n_rounds):
        rng = np.random.RandomState(SEED + rnd)
        n_byz_round = max(1, int(CLIENTS_PER_ROUND * BYZANTINE_RATIO))
        n_ben_round = CLIENTS_PER_ROUND - n_byz_round

        sampled_b = rng.choice(benign_ids,
                    min(n_ben_round, len(benign_ids)), replace=False).tolist()
        sampled_m = rng.choice(byzantine_ids,
                    min(n_byz_round, len(byzantine_ids)), replace=False).tolist()
        sampled   = sampled_b + sampled_m

        updates = []
        for cid in sampled:
            Xc, yc = clients[cid]
            if len(Xc) == 0:
                continue
            is_byz = cid in byzantine_ids
            if is_byz and ATTACK == 'lfp':
                yc = attack_label_flip(yc, flip_rate=0.30)

            m = FLModel(n_features)
            m.set_params(global_params.copy())
            dw, db = m.local_train(Xc, yc, lr=LR, epochs=5, batch_size=32)
            updates.append(np.append(dw, db))

        if not updates:
            continue

        agg, _, _, _ = aggregator.aggregate(
            updates, global_params, X_val, y_val, scaler, sampled)
        global_params = global_params + LR * agg

    global_model.set_params(global_params)
    pred = global_model.predict(X_val)
    acc  = accuracy_score(y_val, pred) * 100
    f1   = f1_score(y_val, pred, zero_division=0) * 100
    return round(acc, 1), round(f1, 1)


# ── Sweep τ ──────────────────────────────────────────────────────────────────
TAU_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
print("\n[1/3] Sweeping τ (outlier threshold)...")
tau_rows = []
for tau in TAU_GRID:
    acc, f1 = run_ours(tau=tau, alpha=ALPHA_DEFAULT, delta=DELTA_DEFAULT)
    tag = " ← default" if tau == TAU_DEFAULT else ""
    print(f"  τ={tau:.1f}  Acc={acc}%  F1={f1}%{tag}")
    tau_rows.append({'τ': tau, 'Accuracy (%)': acc, 'F1 (%)': f1})
pd.DataFrame(tau_rows).to_csv(tag('sensitivity_tau.csv'), index=False)

# ── Sweep α ──────────────────────────────────────────────────────────────────
ALPHA_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]
print("\n[2/3] Sweeping α (trust decay factor)...")
alpha_rows = []
for alpha in ALPHA_GRID:
    acc, f1 = run_ours(tau=TAU_DEFAULT, alpha=alpha, delta=DELTA_DEFAULT)
    tag = " ← default" if alpha == ALPHA_DEFAULT else ""
    print(f"  α={alpha:.1f}  Acc={acc}%  F1={f1}%{tag}")
    alpha_rows.append({'α': alpha, 'Accuracy (%)': acc, 'F1 (%)': f1})
pd.DataFrame(alpha_rows).to_csv(tag('sensitivity_alpha.csv'), index=False)

# ── Sweep δ ──────────────────────────────────────────────────────────────────
DELTA_GRID = [0.10, 0.15, 0.20, 0.25, 0.30]
print("\n[3/3] Sweeping δ (Kappa detection threshold)...")
delta_rows = []
for delta in DELTA_GRID:
    acc, f1 = run_ours(tau=TAU_DEFAULT, alpha=ALPHA_DEFAULT, delta=delta)
    tag = " ← default" if delta == DELTA_DEFAULT else ""
    print(f"  δ={delta:.2f}  Acc={acc}%  F1={f1}%{tag}")
    delta_rows.append({'δ': delta, 'Accuracy (%)': acc, 'F1 (%)': f1})
pd.DataFrame(delta_rows).to_csv(tag('sensitivity_delta.csv'), index=False)

# ── Summary table for LaTeX ──────────────────────────────────────────────────
print("\n── Sensitivity Summary ─────────────────────────────────────────────")
print("\nτ sweep (α=0.8, δ=0.2):")
print(pd.DataFrame(tau_rows).to_string(index=False))
print("\nα sweep (τ=2.0, δ=0.2):")
print(pd.DataFrame(alpha_rows).to_string(index=False))
print("\nδ sweep (τ=2.0, α=0.8):")
print(pd.DataFrame(delta_rows).to_string(index=False))
print("\n[Copy these into Table IX (Sensitivity Analysis) in the LaTeX file]")
