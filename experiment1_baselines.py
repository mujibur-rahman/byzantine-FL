"""
experiment1_baselines.py
========================
Baseline comparison using PROPER federated server/client architecture.

Each method runs through:
  FederatedServer ← broadcasts global model
  FLClient        ← trains locally, returns only gradient update
  FederatedServer ← runs defence pipeline, aggregates, updates model

Methods compared:
  FedAvg, Multi-Krum, TrimmedMean, Bulyan, RFVIR, FLAME, Ours

Output: results/<dataset>/<dataset>_table_baselines.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import time

from dataset_config import get_dataset, DATASET_NAME, tag, results_dir
from fl_base import partition_noniid, make_validation_set
from fl_runner import build_federation, run_federation
from aggregators import AGGREGATOR_REGISTRY

os.makedirs(results_dir(), exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
N_CLIENTS        = 100
CLIENTS_PER_ROUND= 20
N_ROUNDS         = 50
LOCAL_EPOCHS     = 5
LR               = 0.01
NONIID_ALPHA     = 0.5
BYZANTINE_RATIO  = 0.20
VAL_SIZE         = 2000
SEED             = 42

ATTACK_TYPES = ['grad-p', 'spf', 'lfp', 'ooa']
METHODS      = list(AGGREGATOR_REGISTRY.keys())   # all 7 methods

np.random.seed(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
print(f"\n[{DATASET_NAME}] Baseline Comparison Experiment")
X, y = get_dataset(seed=SEED)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_val, y_val = make_validation_set(X_scaled, y, VAL_SIZE, seed=SEED)

# ── Run one experiment ────────────────────────────────────────────────────────
def run_experiment(method_name, attack_type, n_rounds=N_ROUNDS):
    """
    Build a full federation with the specified aggregator and attack,
    run for n_rounds, return final accuracy.
    """
    aggregator_cls = AGGREGATOR_REGISTRY[method_name]

    server, clients, byz_ids = build_federation(
        X_scaled, y, scaler, X_val, y_val,
        n_clients        = N_CLIENTS,
        clients_per_round= CLIENTS_PER_ROUND,
        byzantine_ratio  = BYZANTINE_RATIO,
        attack_type      = attack_type,
        noniid_alpha     = NONIID_ALPHA,
        lr               = LR,
        local_epochs     = LOCAL_EPOCHS,
        seed             = SEED,
    )

    # Override the server's aggregator with the method under test
    server.aggregator = aggregator_cls(n_clients=N_CLIENTS)

    history = run_federation(
        server, clients, set(byz_ids),
        n_rounds = n_rounds,
        verbose  = False,
        seed     = SEED,
    )

    final_acc = history[-1]['accuracy'] if history else 0.0
    final_f1  = history[-1]['f1']       if history else 0.0
    return final_acc, final_f1, history


# ── No-attack baseline ────────────────────────────────────────────────────────
print("\nRunning no-attack baseline (Ours, clean data)...")
server, clients, byz_ids = build_federation(
    X_scaled, y, scaler, X_val, y_val,
    n_clients=N_CLIENTS, clients_per_round=CLIENTS_PER_ROUND,
    byzantine_ratio=0.0,   # no Byzantine clients
    attack_type=None,
    noniid_alpha=NONIID_ALPHA, lr=LR, local_epochs=LOCAL_EPOCHS,
    seed=SEED,
)
no_attack_history = run_federation(
    server, clients, set(),
    n_rounds=N_ROUNDS, verbose=False, seed=SEED)
no_attack_acc = no_attack_history[-1]['accuracy'] if no_attack_history else 0.0
print(f"  No-attack accuracy: {no_attack_acc:.1f}%")

# ── Main comparison ───────────────────────────────────────────────────────────
print(f"\nRunning {len(METHODS)} methods × {len(ATTACK_TYPES)} attacks "
      f"× {N_ROUNDS} rounds each...")
print(f"Dataset: {DATASET_NAME} | Byzantine: {BYZANTINE_RATIO*100:.0f}% | "
      f"Non-IID α: {NONIID_ALPHA}\n")

rows = []
for method in METHODS:
    row = {'Method': method}
    for attack in ATTACK_TYPES:
        t0 = time.time()
        acc, f1, hist = run_experiment(method, attack)
        elapsed = time.time() - t0
        row[attack]          = round(acc, 1)
        row[f'{attack}_f1']  = round(f1, 1)
        print(f"  {method:18s} | {attack:7s} | "
              f"Acc={acc:5.1f}%  F1={f1:5.1f}%  [{elapsed:.1f}s]")
    row['No Attack'] = round(no_attack_acc, 1)
    rows.append(row)

df = pd.DataFrame(rows)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = tag('table_baselines.csv')
df.to_csv(out_path, index=False)

print(f"\n── Results ({DATASET_NAME}) ─────────────────────────────────────")
print(df[['Method'] + ATTACK_TYPES + ['No Attack']].to_string(index=False))
print(f"\nSaved to: {out_path}")
print("[Copy accuracy values into Table VI in the LaTeX file]")
