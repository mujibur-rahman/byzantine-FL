"""
experiment3_layer_analysis.py
==============================
Experiment 3: Per-layer confusion breakdown
Answers RAID R253B: "is Kappa catching cases that trust misses?"

For each attack type, shows:
  - Which clients are detected by Outlier only
  - Which by Trust only
  - Which by Kappa only
  - Which by all three
  - Which are missed (false negatives)

Output: results/layer_confusion.csv
        results/layer_confusion_latex.txt

Also produces precision/recall per layer per attack type.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from dataset_config import get_dataset, DATASET_NAME, tag, results_dir
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              accuracy_score)

from fl_base import (generate_sc_dataset, partition_noniid,
                     make_validation_set, FLModel, compute_kappa,
                     attack_label_flip, attack_gradient_poison,
                     attack_gps_spoof, is_malicious_round)

os.makedirs(results_dir(), exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
N_CLIENTS        = 100
CLIENTS_PER_ROUND= 20
N_ROUNDS         = 30
LR               = 0.01
NONIID_ALPHA     = 0.5
BYZANTINE_RATIO  = 0.20
VAL_SIZE         = 2000
SEED             = 42
TAU, ALPHA, DELTA = 2.0, 0.8, 0.2
TRUST_THRESH      = 0.3

np.random.seed(SEED)

print("Preparing data for layer analysis...")
X, y = get_dataset(seed=SEED)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_val, y_val = make_validation_set(X_scaled, y, VAL_SIZE, seed=SEED)

clients = partition_noniid(X_scaled, y, N_CLIENTS,
                           alpha=NONIID_ALPHA, seed=SEED)
n_byzantine = int(N_CLIENTS * BYZANTINE_RATIO)
n_benign    = N_CLIENTS - n_byzantine
byzantine_ids = set(range(n_benign, N_CLIENTS))
benign_ids    = list(range(n_benign))
n_features    = X_scaled.shape[1]

ATTACK_TYPES = ['grad-p', 'spf', 'lfp', 'ooa']


def run_layer_analysis(attack_type):
    """
    Run FL for N_ROUNDS and track, per client per round:
      - outlier_flag  (Layer 1)
      - low_trust     (Layer 2: trust < TRUST_THRESH after update)
      - low_kappa     (Layer 3: kappa < DELTA)
      - is_byzantine  (ground truth)
    """
    global_model  = FLModel(n_features)
    global_params = global_model.get_params()
    trust_scores  = {i: 0.5 for i in range(N_CLIENTS)}

    # Accumulators over all rounds
    all_records = []  # list of dicts per client-round

    for rnd in range(N_ROUNDS):
        rng = np.random.RandomState(SEED + rnd)
        n_byz_round = max(1, int(CLIENTS_PER_ROUND * BYZANTINE_RATIO))
        n_ben_round = CLIENTS_PER_ROUND - n_byz_round

        sampled_b = rng.choice(list(range(n_benign)),
                    min(n_ben_round, n_benign), replace=False).tolist()
        sampled_m = rng.choice(list(byzantine_ids),
                    min(n_byz_round, len(byzantine_ids)), replace=False).tolist()
        sampled   = sampled_b + sampled_m

        updates = []
        is_byz_list = []

        for cid in sampled:
            Xc, yc = clients[cid]
            if len(Xc) == 0:
                continue
            is_byz = cid in byzantine_ids

            if is_byz:
                if attack_type == 'lfp':
                    yc = attack_label_flip(yc, flip_rate=0.30)
                elif attack_type == 'spf':
                    Xc = attack_gps_spoof(Xc, spoof_rate=0.30)
                elif attack_type == 'ooa' and not is_malicious_round(rnd):
                    is_byz = False

            m = FLModel(n_features)
            m.set_params(global_params.copy())
            dw, db = m.local_train(Xc, yc, lr=LR, epochs=5, batch_size=32)
            update = np.append(dw, db)

            if is_byz and attack_type == 'grad-p':
                dw2, db2 = attack_gradient_poison(update[:-1], update[-1], scale=3.0)
                update = np.append(dw2, db2)

            updates.append((cid, update, cid in byzantine_ids))

        if not updates:
            continue

        # Compute norms for outlier detection
        all_updates = [u for _, u, _ in updates]
        norms = np.array([np.linalg.norm(u) for u in all_updates])
        if norms.std() < 1e-10:
            z_scores = np.zeros(len(norms))
        else:
            z_scores = np.abs((norms - norms.mean()) / norms.std())

        agg_updates = []
        for idx, (cid, update, is_byz_gt) in enumerate(updates):
            # Layer 1
            outlier = bool(z_scores[idx] > TAU)

            # Layer 3: Kappa
            candidate = global_params + LR * update
            kappa = compute_kappa(candidate, global_params, X_val, scaler)
            low_kappa = kappa < DELTA

            # Layer 2: Trust update
            if outlier or low_kappa:
                trust_scores[cid] = max(0.0, trust_scores[cid] - 0.05)
            else:
                trust_scores[cid] = min(1.0, trust_scores[cid] + 0.05)
            low_trust = trust_scores[cid] < TRUST_THRESH

            all_records.append({
                'round':       rnd,
                'client_id':   cid,
                'is_byzantine': is_byz_gt,
                'outlier':      outlier,
                'low_kappa':    low_kappa,
                'low_trust':    low_trust,
                'flagged':      outlier or low_kappa or low_trust,
                'kappa':        kappa,
                'trust':        trust_scores[cid],
            })

            # Aggregate weight
            if not outlier and trust_scores[cid] >= TRUST_THRESH:
                w = trust_scores[cid] * max(kappa, 0.0)
                agg_updates.append((w, update))

        if agg_updates:
            weights = np.array([w for w, _ in agg_updates])
            if weights.sum() > 1e-10:
                weights /= weights.sum()
            agg = sum(w * u for w, (w_, u) in zip(weights, agg_updates))
            global_params = global_params + LR * agg

    df = pd.DataFrame(all_records)
    return df


# ── Run and compute per-layer detection stats ────────────────────────────────
print(f"\nRunning layer analysis across {len(ATTACK_TYPES)} attack types...")

summary_rows = []

for attack in ATTACK_TYPES:
    print(f"\n  Attack: {attack}")
    df = run_layer_analysis(attack)

    # Focus on last 10 rounds (stable phase)
    df_stable = df[df['round'] >= N_ROUNDS - 10]

    # Ground truth
    y_true = df_stable['is_byzantine'].astype(int).values

    # Detection by each layer alone
    y_outlier   = df_stable['outlier'].astype(int).values
    y_kappa     = df_stable['low_kappa'].astype(int).values
    y_trust     = df_stable['low_trust'].astype(int).values
    y_combined  = df_stable['flagged'].astype(int).values

    def safe_metrics(y_true, y_pred):
        if y_pred.sum() == 0:
            return 0.0, 0.0, 0.0
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        return round(p*100,1), round(r*100,1), round(f*100,1)

    p_out, r_out, f_out = safe_metrics(y_true, y_outlier)
    p_kap, r_kap, f_kap = safe_metrics(y_true, y_kappa)
    p_tru, r_tru, f_tru = safe_metrics(y_true, y_trust)
    p_com, r_com, f_com = safe_metrics(y_true, y_combined)

    # Unique detections (caught by Kappa but NOT by outlier — the key claim)
    only_kappa = (y_kappa == 1) & (y_outlier == 0) & (y_trust == 0)
    only_trust = (y_trust == 1) & (y_outlier == 0) & (y_kappa == 0)
    only_outlier = (y_outlier == 1) & (y_trust == 0) & (y_kappa == 0)

    kappa_unique = int((only_kappa & (y_true == 1)).sum())
    trust_unique = int((only_trust & (y_true == 1)).sum())
    outlier_unique = int((only_outlier & (y_true == 1)).sum())

    print(f"    Outlier only : P={p_out}% R={r_out}% F1={f_out}%  "
          f"| Unique TP={outlier_unique}")
    print(f"    Trust only   : P={p_tru}% R={r_tru}% F1={f_tru}%  "
          f"| Unique TP={trust_unique}")
    print(f"    Kappa only   : P={p_kap}% R={r_kap}% F1={f_kap}%  "
          f"| Unique TP={kappa_unique}")
    print(f"    Combined     : P={p_com}% R={r_com}% F1={f_com}%")

    summary_rows.append({
        'Attack': attack,
        'Outlier P': p_out, 'Outlier R': r_out, 'Outlier F1': f_out,
        'Trust P':   p_tru, 'Trust R':   r_tru, 'Trust F1':   f_tru,
        'Kappa P':   p_kap, 'Kappa R':   r_kap, 'Kappa F1':   f_kap,
        'Combined P':p_com, 'Combined R':r_com, 'Combined F1':f_com,
        'Kappa unique TP': kappa_unique,
        'Trust unique TP': trust_unique,
        'Outlier unique TP': outlier_unique,
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(tag('layer_confusion.csv'), index=False)

# ── LaTeX table output ───────────────────────────────────────────────────────
latex = r"""\begin{table}[htbp]
\centering
\scriptsize
\caption{Per-Layer Detection Analysis: F1-Score (\%) and Unique True Positives per Attack Type (20\% Byzantine)}
\label{tab:layer_analysis}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{lcccc|c}
\toprule
\textbf{Attack} & \textbf{Outlier} & \textbf{Trust} & \textbf{Kappa} & \textbf{Combined} & \textbf{Kappa-only TPs} \\
 & F1 (\%) & F1 (\%) & F1 (\%) & F1 (\%) & (missed by L1+L2) \\
\midrule
"""
for row in summary_rows:
    latex += (f"{row['Attack']} & {row['Outlier F1']} & "
              f"{row['Trust F1']} & {row['Kappa F1']} & "
              f"\\textbf{{{row['Combined F1']}}} & "
              f"{row['Kappa unique TP']} \\\\\n")

latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

with open(tag('layer_confusion_latex.txt'), 'w') as f:
    f.write(latex)

print("\nSaved results/layer_confusion.csv and layer_confusion_latex.txt")
print("[The 'Kappa-only TPs' column is the key evidence for Corollary 1]")
