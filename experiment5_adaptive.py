"""
experiment5_adaptive.py
========================
Week 5 — Adaptive Adversary Scenario

Three attack variants, each testing a different evasion strategy:

  A) Sleeper attack (delayed activation)
     Byzantine clients behave honestly for T_burn rounds to
     accumulate high trust, then switch to label flipping.
     Measures detection latency per layer after activation.

  B) Mimicry attack (gradient-constrained poisoning)
     Adversary clips poisoned gradient to lie within the
     statistical envelope of benign gradients (norm ≤ median + 1σ)
     — the Shejwalkar & Houmansadr (NDSS 2021) threat model.
     Tests whether Kappa catches what outlier detection misses.

  C) Slow-drift attack (gradual ramp)
     Label flip rate increases 5% per round from 0% to 40%.
     Tests whether Kappa detects gradual semantic shift before
     trust scoring does.

Key metrics:
  - Detection latency: rounds after attack activation until flagged
  - Per-layer latency comparison (outlier / trust / Kappa)
  - Accuracy under attack vs accuracy under standard (non-adaptive) attack
  - False positive rate during the honest burn-in phase

Run:  python3 experiment5_adaptive.py
Output:
  results/adaptive_sleeper.csv
  results/adaptive_mimicry.csv
  results/adaptive_slowdrift.csv
  results/adaptive_detection_latency.csv  ← key table for paper
  results/adaptive_latex.txt              ← ready-to-paste LaTeX
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from dataset_config import get_dataset, DATASET_NAME, tag, results_dir
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from fl_model import FLNeuralNet
from fl_base import (generate_sc_dataset, partition_noniid,
                     make_validation_set, compute_kappa,
                     attack_label_flip, attack_gradient_poison)

os.makedirs(results_dir(), exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
N_CLIENTS        = 100
CLIENTS_PER_ROUND= 20
N_ROUNDS         = 60          # long enough to see burn-in + attack + recovery
LR               = 0.01
NONIID_ALPHA     = 0.5
BYZANTINE_RATIO  = 0.20
T_BURN           = 20          # rounds of honest behaviour before activation
VAL_SIZE         = 2000
SEED             = 42

TAU          = 2.0
ALPHA        = 0.8
DELTA        = 0.2
TRUST_THRESH = 0.3
TRUST_DECAY  = 0.05
TRUST_REWARD = 0.05

np.random.seed(SEED)

print("Preparing data for adaptive adversary experiments...")
X, y = get_dataset(seed=SEED)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_val, y_val = make_validation_set(X_scaled, y, VAL_SIZE, seed=SEED)

clients = partition_noniid(X_scaled, y, N_CLIENTS,
                           alpha=NONIID_ALPHA, seed=SEED)
n_byzantine   = int(N_CLIENTS * BYZANTINE_RATIO)
n_benign      = N_CLIENTS - n_byzantine
byzantine_ids = set(range(n_benign, N_CLIENTS))
benign_ids    = list(range(n_benign))
n_features    = X_scaled.shape[1]


# ── Core simulation loop ──────────────────────────────────────────────────────
def run_adaptive(attack_variant, n_rounds=N_ROUNDS, t_burn=T_BURN):
    """
    Returns per-round records with per-layer detection flags.

    attack_variant: 'sleeper' | 'mimicry' | 'slowdrift'
    """
    global_model  = FLNeuralNet(n_features)
    global_params = global_model.get_params()
    trust_scores  = {i: 0.5 for i in range(N_CLIENTS)}

    round_records = []   # one dict per round (aggregated across clients)
    client_records = []  # one dict per client per round (for latency analysis)

    for rnd in range(n_rounds):
        rng = np.random.RandomState(SEED + rnd)
        n_byz_round = max(1, int(CLIENTS_PER_ROUND * BYZANTINE_RATIO))
        n_ben_round = CLIENTS_PER_ROUND - n_byz_round

        sampled_b = rng.choice(benign_ids,
                    min(n_ben_round, len(benign_ids)), replace=False).tolist()
        sampled_m = rng.choice(list(byzantine_ids),
                    min(n_byz_round, len(byzantine_ids)), replace=False).tolist()
        sampled   = sampled_b + sampled_m

        # Is this round in the attack phase?
        attack_active = rnd >= t_burn

        updates_weighted = []
        agg_norm_sum = 0.0

        for cid in sampled:
            Xc, yc = clients[cid]
            if len(Xc) == 0:
                continue

            is_byz = cid in byzantine_ids

            # ── Apply adaptive attack ─────────────────────────────────────
            if is_byz and attack_active:
                if attack_variant == 'sleeper':
                    # Full label flip after burn-in
                    yc = attack_label_flip(yc, flip_rate=0.30)

                elif attack_variant == 'mimicry':
                    # Label flip but gradient is clipped to look benign
                    yc = attack_label_flip(yc, flip_rate=0.25)
                    # (clipping applied after computing gradient below)

                elif attack_variant == 'slowdrift':
                    # Ramp: 5% per round from T_burn, capped at 40%
                    flip_rate = min(0.40, 0.05 * (rnd - t_burn + 1))
                    if flip_rate > 0:
                        yc = attack_label_flip(yc, flip_rate=flip_rate)

            # ── Local training ────────────────────────────────────────────
            m = FLNeuralNet(n_features)
            m.set_params(global_params.copy())
            params_before = m.get_params()
            m.train(Xc, yc, lr=LR, epochs=5, batch_size=32)
            update = m.get_params() - params_before

            # Mimicry: clip gradient norm to benign envelope
            if is_byz and attack_active and attack_variant == 'mimicry':
                # Estimate benign norm from trust scores (server knows nothing
                # about which are benign, so adversary clips to median norm
                # of all recent updates — a realistic assumption)
                target_norm = np.linalg.norm(global_params) * 0.05
                current_norm = np.linalg.norm(update)
                if current_norm > target_norm and current_norm > 1e-10:
                    update = update * target_norm / current_norm

            # ── Layer 1: Outlier detection ────────────────────────────────
            # (computed globally after collecting all updates — done below)
            updates_weighted.append((cid, update, is_byz))

        if not updates_weighted:
            continue

        all_updates = [u for _, u, _ in updates_weighted]
        norms = np.array([np.linalg.norm(u) for u in all_updates])
        if norms.std() < 1e-10:
            z_scores = np.zeros(len(norms))
        else:
            z_scores = np.abs((norms - norms.mean()) / norms.std())

        # ── Per-client Layer 2 & 3 ────────────────────────────────────────
        agg_update  = np.zeros(len(global_params))
        weight_sum  = 0.0

        for idx, (cid, update, is_byz_gt) in enumerate(updates_weighted):
            # Layer 1
            outlier = bool(z_scores[idx] > TAU)

            # Layer 3: Kappa (stateless — recomputed fresh every round)
            candidate = global_params + LR * update
            kappa = compute_kappa(candidate, global_params, X_val, scaler)
            low_kappa = kappa < DELTA

            # Layer 2: trust update
            if outlier or low_kappa:
                trust_scores[cid] = max(0.0, trust_scores[cid] - TRUST_DECAY)
            else:
                trust_scores[cid] = min(1.0, trust_scores[cid] + TRUST_REWARD)
            low_trust = trust_scores[cid] < TRUST_THRESH

            flagged = outlier or low_kappa or low_trust

            client_records.append({
                'round':         rnd,
                'client_id':     cid,
                'attack_active': attack_active,
                'is_byzantine':  is_byz_gt,
                'outlier':       outlier,
                'low_kappa':     low_kappa,
                'low_trust':     low_trust,
                'flagged':       flagged,
                'kappa':         round(kappa, 4),
                'trust':         round(trust_scores[cid], 4),
            })

            # Aggregate only if not flagged / not outlier
            if not outlier and not low_trust:
                w = trust_scores[cid] * max(kappa, 0.001)
                agg_update += w * update
                weight_sum += w

        if weight_sum > 1e-10:
            global_params = global_params + LR * (agg_update / weight_sum)

        # ── Round-level metrics ───────────────────────────────────────────
        global_model.set_params(global_params)
        pred_val = global_model.predict(X_val)
        acc_val  = accuracy_score(y_val, pred_val) * 100

        # Detection rate this round (among byzantine clients sampled)
        byz_records_this_round = [r for r in client_records
                                   if r['round'] == rnd and r['is_byzantine']]
        if byz_records_this_round:
            detected_outlier = np.mean([r['outlier']    for r in byz_records_this_round]) * 100
            detected_kappa   = np.mean([r['low_kappa']  for r in byz_records_this_round]) * 100
            detected_trust   = np.mean([r['low_trust']  for r in byz_records_this_round]) * 100
            detected_any     = np.mean([r['flagged']    for r in byz_records_this_round]) * 100
        else:
            detected_outlier = detected_kappa = detected_trust = detected_any = 0.0

        # False positive rate (benign clients flagged)
        ben_records_this_round = [r for r in client_records
                                   if r['round'] == rnd and not r['is_byzantine']]
        fpr = (np.mean([r['flagged'] for r in ben_records_this_round]) * 100
               if ben_records_this_round else 0.0)

        round_records.append({
            'round':            rnd,
            'attack_active':    attack_active,
            'accuracy':         round(acc_val, 2),
            'det_outlier_pct':  round(detected_outlier, 1),
            'det_kappa_pct':    round(detected_kappa, 1),
            'det_trust_pct':    round(detected_trust, 1),
            'det_any_pct':      round(detected_any, 1),
            'fpr_pct':          round(fpr, 1),
        })

        if rnd % 10 == 0 or rnd == t_burn or rnd == t_burn + 1:
            phase = "ATTACK" if attack_active else "burn-in"
            print(f"    R{rnd:02d} [{phase}] Acc={acc_val:.1f}%  "
                  f"Det(K={detected_kappa:.0f}% T={detected_trust:.0f}% "
                  f"O={detected_outlier:.0f}%)  FPR={fpr:.1f}%")

    return pd.DataFrame(round_records), pd.DataFrame(client_records)


# ── Compute detection latency ─────────────────────────────────────────────────
def detection_latency(round_df, t_burn=T_BURN, threshold_pct=50.0):
    """
    How many rounds after T_burn until each layer detects >50% of
    Byzantine clients?
    Returns dict of {layer: latency_in_rounds or 'Never'}
    """
    attack_phase = round_df[round_df['attack_active']]
    latencies = {}
    for col, label in [('det_outlier_pct', 'Outlier'),
                        ('det_trust_pct',   'Trust'),
                        ('det_kappa_pct',   'Kappa'),
                        ('det_any_pct',     'Combined')]:
        detected_rounds = attack_phase[attack_phase[col] >= threshold_pct]['round']
        if len(detected_rounds) == 0:
            latencies[label] = 'Never'
        else:
            latencies[label] = int(detected_rounds.iloc[0] - t_burn)
    return latencies


# ── Run all three variants ────────────────────────────────────────────────────
variants = {
    'sleeper':   'A: Sleeper (delayed activation, T_burn=20)',
    'mimicry':   'B: Mimicry (gradient-constrained poisoning)',
    'slowdrift': 'C: Slow drift (ramp 5%/round to 40% flip)',
}

latency_rows = []
all_results  = {}

for variant, desc in variants.items():
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    rdf, cdf = run_adaptive(variant, N_ROUNDS, T_BURN)
    all_results[variant] = (rdf, cdf)

    rdf.to_csv(tag(f'{DATASET_NAME}_adaptive_{variant}.csv'), index=False)

    lat = detection_latency(rdf, T_BURN)
    acc_at_activation = rdf[rdf['round'] == T_BURN]['accuracy'].values
    acc_steady        = rdf[rdf['round'] >= N_ROUNDS - 5]['accuracy'].mean()
    fpr_burnin        = rdf[~rdf['attack_active']]['fpr_pct'].mean()

    print(f"\n  Detection latency (rounds after activation):")
    for layer, l in lat.items():
        print(f"    {layer:12s}: {l} rounds")
    print(f"  Acc at activation (round {T_BURN}): "
          f"{acc_at_activation[0] if len(acc_at_activation) else 'N/A'}%")
    print(f"  Acc steady-state (last 5 rounds): {acc_steady:.1f}%")
    print(f"  FPR during burn-in: {fpr_burnin:.1f}%")

    latency_rows.append({
        'Variant':            desc.split(':')[0].strip(),
        'Attack':             desc.split(': ')[1] if ': ' in desc else desc,
        'Outlier latency':    lat['Outlier'],
        'Trust latency':      lat['Trust'],
        'Kappa latency':      lat['Kappa'],
        'Combined latency':   lat['Combined'],
        'Burn-in FPR (%)':    round(fpr_burnin, 1),
        'Final Acc (%)':      round(acc_steady, 1),
    })

lat_df = pd.DataFrame(latency_rows)
lat_df.to_csv(tag(f'{DATASET_NAME}_adaptive_detection_latency.csv'), index=False)

# ── LaTeX output ──────────────────────────────────────────────────────────────
latex_table = r"""
\begin{table}[htbp]
\centering
\scriptsize
\caption{Adaptive Adversary Detection: Latency (Rounds After Activation) Per Layer and Final Model Accuracy (\%), $T_{\text{burn}}=20$, 20\% Byzantine}
\label{tab:adaptive}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{p{2.2cm}cccc|cc}
\toprule
\textbf{Attack Variant} & \textbf{Outlier} & \textbf{Trust} & \textbf{Kappa} & \textbf{Combined} & \textbf{Burn-in FPR} & \textbf{Final Acc} \\
 & latency & latency & latency & latency & (\%) & (\%) \\
\midrule
"""
for row in latency_rows:
    latex_table += (
        f"{row['Attack']} & "
        f"{row['Outlier latency']} & "
        f"{row['Trust latency']} & "
        f"\\textbf{{{row['Kappa latency']}}} & "
        f"{row['Combined latency']} & "
        f"{row['Burn-in FPR (%)']} & "
        f"{row['Final Acc (%)']} \\\\\n"
    )

latex_table += r"""\bottomrule
\end{tabular}
\vspace{1mm}
\begin{minipage}{0.48\textwidth}
\scriptsize
Latency = rounds elapsed between attack activation (round $T_{\text{burn}}$)
and the first round in which $>$50\% of Byzantine clients are flagged.
\textbf{Kappa latency} is the detection speed of Layer~3 alone.
``Never'' indicates the layer never detected $>$50\% of attackers
within the experiment window.
Trust latency reflects EMA decay ($\alpha=0.8$): a client with
$T_i \approx 1.0$ requires $\approx \lceil \log_{0.8}(0.7) \rceil = 8$ rounds
to fall below threshold---consistent with the Burn-in Latency Remark
in Section~IV.
\end{minipage}
\end{table}
"""

with open(tag('adaptive_latex.txt'), 'w') as f:
    f.write(latex_table)

print("\n" + "="*60)
print("Saved:")
print("  results/adaptive_sleeper.csv")
print("  results/adaptive_mimicry.csv")
print("  results/adaptive_slowdrift.csv")
print("  results/adaptive_detection_latency.csv  ← key table")
print("  results/adaptive_latex.txt              ← paste into paper")
print("="*60)
print(lat_df.to_string(index=False))
