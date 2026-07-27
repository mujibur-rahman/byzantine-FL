"""
experiment6_semantic_divergence.py
===================================
Dedicated semantic divergence analysis — the core empirical
proof of the paper's primary claim.

Answers: "Does Cohen's Kappa actually measure semantic divergence,
          or just prediction noise?"

For each attack type, across training rounds, this experiment records:

  Per-client per-round:
    κᵢ                    — Kappa score (our detection signal)
    p_fraud_i             — fraction of D_val client predicts as fraud
    p_fraud_global        — fraction D_val global model predicts as fraud
    semantic_shift        — |p_fraud_i - p_fraud_global|  (the divergence)
    gradient_distance     — ‖Δwᵢ - w_med‖₂  (gradient-level signal)

  Key finding to demonstrate:
    For label-flipping and GPS-spoofing attacks:
      - gradient_distance remains LOW (attacker passes gradient filter)
      - semantic_shift becomes HIGH (attacker's predictions diverge)
      - κᵢ drops below δ     (Kappa catches what gradient filter misses)

    This is Corollary 1 demonstrated empirically:
      "There exist adversaries that bypass gradient filters
       yet are caught by Kappa."

Output:
  results/<dataset>/semantic_divergence_per_round.csv
  results/<dataset>/semantic_divergence_summary.csv
  results/<dataset>/semantic_divergence_latex.txt

Run:
  python3 experiment6_semantic_divergence.py --dataset nyc-taxi
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, accuracy_score

from dataset_config import get_dataset, DATASET_NAME, tag, results_dir
from fl_model import FLNeuralNet
from fl_base import (partition_noniid, make_validation_set,
                     attack_label_flip, attack_gradient_poison,
                     attack_gps_spoof, is_malicious_round)

os.makedirs(results_dir(), exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
N_CLIENTS         = 100
CLIENTS_PER_ROUND = 20
N_ROUNDS          = 30
LR                = 0.01
NONIID_ALPHA      = 0.5
BYZANTINE_RATIO   = 0.20
VAL_SIZE          = 2000
SEED              = 42
DELTA             = 0.2    # Kappa threshold (legacy; detection now peer-relative)
TAU               = 2.0    # outlier z-score threshold
KAPPA_VAL_SIZE    = 500    # subsample of D_val for the kappa check
K_MAD             = 1.5    # peer-relative flag: kappa < median - K_MAD*1.4826*MAD

np.random.seed(SEED)

ATTACK_TYPES = ['grad-p', 'spf', 'lfp', 'ooa']

print(f"[{DATASET_NAME}] Semantic Divergence Analysis")
print(f"{'='*60}")

# ── Data ──────────────────────────────────────────────────────────────────────
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

# ── Global fraud rate on val set (ground truth reference) ────────────────────
global_fraud_rate = y_val.mean()
print(f"Val set fraud rate: {global_fraud_rate:.1%}\n")


def run_divergence_analysis(attack_type):
    """
    Run FL for N_ROUNDS. For every client every round, record:
      - κᵢ (Kappa between client model and global model)
      - semantic_shift (|p_fraud_client - p_fraud_global|)
      - gradient_distance (L2 norm from coordinate-wise median)
      - is_byzantine (ground truth label)
      - flagged_by_kappa (κᵢ < δ)
      - flagged_by_gradient (dist > τ·σ)
    """
    global_model  = FLNeuralNet(n_features)
    global_params = global_model.get_params()

    records = []

    for rnd in range(N_ROUNDS):
        rng = np.random.RandomState(SEED + rnd)
        n_byz_round = max(1, int(CLIENTS_PER_ROUND * BYZANTINE_RATIO))
        n_ben_round = CLIENTS_PER_ROUND - n_byz_round

        sampled_b = rng.choice(benign_ids,
                    min(n_ben_round, len(benign_ids)),
                    replace=False).tolist()
        sampled_m = rng.choice(list(byzantine_ids),
                    min(n_byz_round, len(byzantine_ids)),
                    replace=False).tolist()
        sampled   = sampled_b + sampled_m

        # ── Local training for each client ────────────────────────────
        updates = []
        is_byz_flags = []

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

            m = FLNeuralNet(n_features)
            m.set_params(global_params.copy())
            params_before = m.get_params()
            m.train(Xc, yc, lr=LR, epochs=5, batch_size=32)
            update = m.get_params() - params_before

            if is_byz and attack_type == 'grad-p':
                dw2, db2 = attack_gradient_poison(
                    update[:-1], update[-1], scale=3.0)
                update = np.append(dw2, db2)

            updates.append(update)
            is_byz_flags.append(is_byz)

        if not updates:
            continue

        # ── Coordinate-wise median for gradient distance ──────────────
        U     = np.vstack(updates)
        w_med = np.median(U, axis=0)
        norms = np.array([np.linalg.norm(u) for u in updates])
        if norms.std() > 1e-10:
            z_scores = np.abs((norms - norms.mean()) / norms.std())
        else:
            z_scores = np.zeros(len(norms))

        # ── Global model predictions on D_val (kappa on a subsample) ──
        if len(X_val) > KAPPA_VAL_SIZE:
            _ki = np.random.RandomState(0).choice(
                len(X_val), KAPPA_VAL_SIZE, replace=False)
            Xk = X_val[_ki]
        else:
            Xk = X_val
        gm = FLNeuralNet(n_features)
        gm.set_params(global_params.copy())
        pred_global    = gm.predict(Xk)
        p_fraud_global = pred_global.mean()   # fraction predicted fraud

        # ── Pass 1: per-client Kappa on the client's ACTUAL local model ──
        # candidate = global + update (full model), NOT global + LR*update:
        # the lr-scaled step barely moves predictions, so kappa saturates
        # near 1 and only huge (grad-poison) updates register. The full model
        # exposes real prediction-space divergence for label/semantic attacks.
        kappas, shifts, gdists, pfc = [], [], [], []
        for update in updates:
            cm = FLNeuralNet(n_features)
            cm.set_params(global_params + update)
            pred_client = cm.predict(Xk)
            if (len(np.unique(pred_client)) > 1 and
                    len(np.unique(pred_global)) > 1):
                k = cohen_kappa_score(pred_global, pred_client)
            else:
                k = 1.0 if np.array_equal(pred_client, pred_global) else 0.0
            kappas.append(k)
            pfc.append(pred_client.mean())
            shifts.append(abs(pred_client.mean() - p_fraud_global))
            gdists.append(np.linalg.norm(update - w_med))
        kappas = np.array(kappas)
        # Peer-relative lower-tail flag (removes per-round convergence shift).
        _med = np.median(kappas)
        _mad = np.median(np.abs(kappas - _med)) + 1e-9
        kappa_flag = kappas < (_med - K_MAD * 1.4826 * _mad)

        # ── Pass 2: record + aggregate ────────────────────────────────
        agg_update = np.zeros(len(global_params))
        weight_sum = 0.0

        for idx, (update, is_byz_gt) in enumerate(
                zip(updates, is_byz_flags)):
            kappa            = float(kappas[idx])
            p_fraud_client   = pfc[idx]
            semantic_shift   = shifts[idx]
            gradient_dist    = gdists[idx]
            flagged_kappa    = bool(kappa_flag[idx])
            flagged_gradient = z_scores[idx] > TAU

            records.append({
                'dataset':           DATASET_NAME,
                'attack':            attack_type,
                'round':             rnd,
                'client_id':         sampled[idx],
                'is_byzantine':      is_byz_gt,
                'kappa':             round(kappa, 4),
                'semantic_shift':    round(semantic_shift, 4),
                'p_fraud_client':    round(p_fraud_client, 4),
                'p_fraud_global':    round(p_fraud_global, 4),
                'gradient_dist':     round(gradient_dist, 4),
                'flagged_kappa':     flagged_kappa,
                'flagged_gradient':  flagged_gradient,
                'z_score':           round(z_scores[idx], 4),
            })

            # Simple aggregation (no filtering — we want to see full divergence)
            agg_update += update
            weight_sum += 1.0

        if weight_sum > 0:
            global_params = global_params + LR * (agg_update / weight_sum)

    return pd.DataFrame(records)


# ── Run all attacks ───────────────────────────────────────────────────────────
all_records = []
summary_rows = []

for attack in ATTACK_TYPES:
    print(f"\nAttack: {attack}")
    df = run_divergence_analysis(attack)
    all_records.append(df)

    # ── Summary statistics ────────────────────────────────────────────
    # Focus on stable phase (last 10 rounds)
    df_stable = df[df['round'] >= N_ROUNDS - 10]

    byz  = df_stable[df_stable['is_byzantine']]
    ben  = df_stable[~df_stable['is_byzantine']]

    # Key metrics
    byz_kappa_mean    = byz['kappa'].mean()
    ben_kappa_mean    = ben['kappa'].mean()
    byz_shift_mean    = byz['semantic_shift'].mean()
    ben_shift_mean    = ben['semantic_shift'].mean()
    byz_grad_mean     = byz['gradient_dist'].mean()
    ben_grad_mean     = ben['gradient_dist'].mean()

    # The core claim: Kappa catches what gradient misses
    # = byzantine clients with LOW gradient distance but HIGH semantic shift
    low_grad_high_kappa_miss = byz[
        (byz['z_score'] <= TAU) &    # passed gradient filter
        (byz['flagged_kappa'])       # caught by Kappa (peer-relative)
    ]
    kappa_unique_detections = len(low_grad_high_kappa_miss)
    total_byz_samples       = len(byz)

    print(f"  Byzantine clients  — κ={byz_kappa_mean:.3f}  "
          f"shift={byz_shift_mean:.3f}  grad={byz_grad_mean:.3f}")
    print(f"  Benign clients     — κ={ben_kappa_mean:.3f}  "
          f"shift={ben_shift_mean:.3f}  grad={ben_grad_mean:.3f}")
    print(f"  Kappa-only detections (passed gradient filter): "
          f"{kappa_unique_detections}/{total_byz_samples} "
          f"({100*kappa_unique_detections/max(total_byz_samples,1):.1f}%)")

    summary_rows.append({
        'Attack':              attack,
        'Byz κ (mean)':        round(byz_kappa_mean, 3),
        'Ben κ (mean)':        round(ben_kappa_mean, 3),
        'κ gap':               round(ben_kappa_mean - byz_kappa_mean, 3),
        'Byz shift (mean)':    round(byz_shift_mean, 3),
        'Ben shift (mean)':    round(ben_shift_mean, 3),
        'Byz grad dist':       round(byz_grad_mean, 3),
        'Ben grad dist':       round(ben_grad_mean, 3),
        'Grad dist gap':       round(byz_grad_mean - ben_grad_mean, 3),
        'Kappa-only det (%)':  round(
            100*kappa_unique_detections/max(total_byz_samples,1), 1),
    })


# ── Save full records ─────────────────────────────────────────────────────────
df_all = pd.concat(all_records, ignore_index=True)
df_all.to_csv(tag('semantic_divergence_per_round.csv'), index=False)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(tag('semantic_divergence_summary.csv'), index=False)

# ── LaTeX table ───────────────────────────────────────────────────────────────
latex = r"""\begin{table}[htbp]
\centering
\scriptsize
\caption{Semantic Divergence Analysis: Cohen's $\kappa$ vs Gradient Distance
per Attack Type (20\% Byzantine, """ + DATASET_NAME + r""", last 10 rounds)}
\label{tab:semantic_divergence}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{lcccc|cc|c}
\toprule
& \multicolumn{2}{c}{\textbf{Cohen's $\kappa$}}
& \multicolumn{2}{c|}{\textbf{Semantic Shift}}
& \multicolumn{2}{c|}{\textbf{Gradient Distance}}
& \textbf{Kappa-only} \\
\textbf{Attack}
& Byz & Ben & Byz & Ben & Byz & Ben & \textbf{Det. (\%)} \\
\midrule
"""

for row in summary_rows:
    latex += (
        f"{row['Attack']} & "
        f"{row['Byz κ (mean)']} & {row['Ben κ (mean)']} & "
        f"{row['Byz shift (mean)']} & {row['Ben shift (mean)']} & "
        f"{row['Byz grad dist']} & {row['Ben grad dist']} & "
        f"\\textbf{{{row['Kappa-only det (%)']}\\%}} \\\\\n"
    )

latex += r"""\bottomrule
\end{tabular}
\vspace{1mm}
\begin{minipage}{0.48\textwidth}
\scriptsize
\textbf{Semantic Shift} = $|p^{\text{fraud}}_{\text{client}} -
p^{\text{fraud}}_{\text{global}}|$: difference in fraud prediction
rate on $\mathcal{D}_{val}$.
\textbf{Kappa-only Det.} = Byzantine clients that passed the gradient
filter ($z$-score $\leq \tau$) yet were caught by Kappa ($\kappa_i < \delta$),
directly validating Corollary~1.
\end{minipage}
\end{table}
"""

with open(tag('semantic_divergence_latex.txt'), 'w') as f:
    f.write(latex)

print(f"\n{'='*60}")
print(f"Summary — Semantic Divergence Analysis ({DATASET_NAME})")
print(f"{'='*60}")
print(df_summary.to_string(index=False))
print(f"\nSaved:")
print(f"  {tag('semantic_divergence_per_round.csv')}")
print(f"  {tag('semantic_divergence_summary.csv')}")
print(f"  {tag('semantic_divergence_latex.txt')}")
print(f"\nKey column to cite in paper:")
print(f"  'Kappa-only Det. (%)' — percentage of Byzantine clients")
print(f"  that passed the gradient filter but were caught by Kappa.")
print(f"  This is the direct empirical proof of Corollary 1.")
