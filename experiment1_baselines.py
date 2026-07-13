"""
experiment1_baselines.py
========================
Experiment 1: FLAME + updated baseline comparison
Reproduces Table VI from the paper, adding FLAME as a new row.

Run:  python3 experiment1_baselines.py
Output: results/table_baselines.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import time

from dataset_config import get_dataset, DATASET_NAME, tag, results_dir
from fl_model import FLNeuralNet
from fl_base import (generate_sc_dataset, partition_noniid,
                     make_validation_set,
                     attack_label_flip, attack_gradient_poison,
                     attack_gps_spoof, is_malicious_round)
from aggregators import (fedavg, krum, trimmed_mean, bulyan,
                         rfvir, flame, MultiLayerAggregator)

os.makedirs(results_dir(), exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
N_TOTAL_CLIENTS  = 500        # Scaled down for CPU; paper uses 50K (TFF)
CLIENTS_PER_ROUND= 20
N_ROUNDS         = 50
LOCAL_EPOCHS     = 5
LR               = 0.01
BATCH_SIZE       = 32
NONIID_ALPHA     = 0.5        # "strong skew" level from Table V
BYZANTINE_RATIO  = 0.20       # 20% malicious
VAL_SIZE         = 2000
N_SAMPLES        = 100_000
SEED             = 42

np.random.seed(SEED)

ATTACK_TYPES = ['grad-p', 'spf', 'lfp', 'ooa']

METHODS = ['FedAvg', 'Multi-Krum', 'TrimmedMean', 'Bulyan',
           'RFVIR', 'FLAME', 'Ours']

# ── Data prep ────────────────────────────────────────────────────────────────
X, y = get_dataset(seed=SEED)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_val, y_val = make_validation_set(X_scaled, y, VAL_SIZE, seed=SEED)

clients = partition_noniid(X_scaled, y, N_TOTAL_CLIENTS,
                           alpha=NONIID_ALPHA, seed=SEED)

n_byzantine = int(N_TOTAL_CLIENTS * BYZANTINE_RATIO)
n_benign    = N_TOTAL_CLIENTS - n_byzantine
byzantine_ids = list(range(n_benign, N_TOTAL_CLIENTS))
benign_ids    = list(range(n_benign))

n_features = X_scaled.shape[1]

# ── Helper: run one full experiment ─────────────────────────────────────────
def run_experiment(method_name, attack_type, n_rounds=N_ROUNDS):
    global_model = FLNeuralNet(n_features)
    global_params = global_model.get_params()

    if method_name == 'Ours':
        aggregator = MultiLayerAggregator(
            N_TOTAL_CLIENTS, tau=2.0, alpha=0.8, delta=0.2)

    accs = []

    for rnd in range(n_rounds):
        # Sample clients
        rng = np.random.RandomState(SEED + rnd)
        sampled_benign   = rng.choice(benign_ids,
                            min(int(CLIENTS_PER_ROUND * (1-BYZANTINE_RATIO)),
                                len(benign_ids)), replace=False).tolist()
        sampled_byzantine= rng.choice(byzantine_ids,
                            min(int(CLIENTS_PER_ROUND * BYZANTINE_RATIO),
                                len(byzantine_ids)), replace=False).tolist()
        sampled = sampled_benign + sampled_byzantine

        updates = []
        for cid in sampled:
            Xc, yc = clients[cid]
            if len(Xc) == 0:
                continue

            is_byz = cid in byzantine_ids

            # Apply attack to data or gradient
            if is_byz:
                if attack_type == 'lfp':
                    yc = attack_label_flip(yc, flip_rate=0.30)
                elif attack_type == 'spf':
                    Xc = attack_gps_spoof(Xc, spoof_rate=0.30)
                elif attack_type == 'ooa':
                    if not is_malicious_round(rnd):
                        is_byz = False  # behave honestly this round

            m = FLNeuralNet(n_features)
            m.set_params(global_params.copy())
            params_before = m.get_params()
            m.train(Xc, yc, lr=LR,
                                   epochs=LOCAL_EPOCHS,
                                   batch_size=BATCH_SIZE)
            update = m.get_params() - params_before

            if is_byz and attack_type == 'grad-p':
                noise = np.random.randn(*update.shape) * 3.0 * np.linalg.norm(update)
                update = update + noise

            updates.append(update)

        if len(updates) == 0:
            continue

        # Aggregate
        if method_name == 'FedAvg':
            agg = fedavg(updates)
        elif method_name == 'Multi-Krum':
            agg = krum(updates, f=max(1, len(sampled_byzantine)), m=3)
        elif method_name == 'TrimmedMean':
            agg = trimmed_mean(updates, trim_ratio=0.1)
        elif method_name == 'Bulyan':
            agg = bulyan(updates, f=max(1, len(sampled_byzantine)))
        elif method_name == 'RFVIR':
            agg = rfvir(updates, global_params)
        elif method_name == 'FLAME':
            agg = flame(updates, noise_std=0.001)
        elif method_name == 'Ours':
            agg, _, _, _ = aggregator.aggregate(
                updates, global_params, X_val, y_val, scaler, sampled)

        global_params = global_params + LR * agg
        global_model.set_params(global_params)

        # Evaluate on validation set
        pred = global_model.predict(X_val)
        acc  = accuracy_score(y_val, pred) * 100
        accs.append(acc)

    return accs[-1] if accs else 0.0, accs


# ── Run all combinations ─────────────────────────────────────────────────────
print(f"\nRunning {len(METHODS)} methods × {len(ATTACK_TYPES)} attacks "
      f"× {N_ROUNDS} rounds each...")
print(f"Byzantine ratio: {BYZANTINE_RATIO*100:.0f}%  |  "
      f"Non-IID alpha: {NONIID_ALPHA}  |  "
      f"Clients/round: {CLIENTS_PER_ROUND}\n")

results = {}
for method in METHODS:
    results[method] = {}
    for attack in ATTACK_TYPES:
        t0 = time.time()
        final_acc, acc_curve = run_experiment(method, attack)
        elapsed = time.time() - t0
        results[method][attack] = final_acc
        print(f"  {method:15s} | {attack:7s} | "
              f"Acc={final_acc:5.1f}%  [{elapsed:.1f}s]")

# No-attack baseline
print("\nRunning no-attack baseline...")
no_attack_accs = {}
for method in ['FedAvg', 'Ours']:
    final_acc, _ = run_experiment.__wrapped__(method, 'lfp') \
        if hasattr(run_experiment, '__wrapped__') else (None, None)

# Clean no-attack
def run_no_attack(method_name):
    global_model = FLNeuralNet(n_features)
    global_params = global_model.get_params()
    if method_name == 'Ours':
        aggregator = MultiLayerAggregator(N_TOTAL_CLIENTS)
    for rnd in range(N_ROUNDS):
        rng = np.random.RandomState(SEED + rnd)
        sampled = rng.choice(benign_ids,
                   min(CLIENTS_PER_ROUND, len(benign_ids)),
                   replace=False).tolist()
        updates = []
        for cid in sampled:
            Xc, yc = clients[cid]
            m = FLNeuralNet(n_features)
            m.set_params(global_params.copy())
            params_before = m.get_params()
            m.train(Xc, yc, lr=LR, epochs=LOCAL_EPOCHS)
            updates.append(m.get_params() - params_before)
        if not updates:
            continue
        if method_name == 'FedAvg':
            agg = fedavg(updates)
        elif method_name == 'Ours':
            agg, _, _, _ = aggregator.aggregate(
                updates, global_params, X_val, y_val, scaler, sampled)
        global_params = global_params + LR * agg
    global_model.set_params(global_params)
    pred = global_model.predict(X_val)
    return accuracy_score(y_val, pred) * 100

no_attack_acc = run_no_attack('Ours')
print(f"  No-attack (Ours): {no_attack_acc:.1f}%")

# ── Build output table ───────────────────────────────────────────────────────
rows = []
for method in METHODS:
    row = {'Method': method}
    for attack in ATTACK_TYPES:
        row[attack] = f"{results[method][attack]:.1f}"
    row['No Attack'] = f"{no_attack_acc:.1f}"
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(tag('table_baselines.csv'), index=False)

print("\n── Results Table (Accuracy %) ──────────────────────────────────")
print(df.to_string(index=False))
print("\nSaved to results/table_baselines.csv")
print("\n[Copy these numbers into Table VI in the LaTeX file]")
