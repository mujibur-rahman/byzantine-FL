"""
experiment4_overhead.py
========================
Experiment 4: Computational overhead analysis
Answers INFOCOM R1/R2: "no measurements of computational overhead"

Measures per-round wall-clock time and complexity for each aggregator.
Reports:
  - Avg time per round (ms)
  - Peak memory estimate (MB)
  - Big-O complexity (theoretical, listed)
  - Time overhead vs FedAvg (×)

Run:  python3 experiment4_overhead.py
Output: results/overhead.csv
        results/overhead_latex.txt
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import time
import tracemalloc
from dataset_config import DATASET_NAME, tag, results_dir
from sklearn.preprocessing import StandardScaler

from fl_model import FLNeuralNet
from fl_base import (generate_sc_dataset, partition_noniid,
                     make_validation_set)
from aggregators import (fedavg, krum, trimmed_mean, bulyan,
                         rfvir, flame, MultiLayerAggregator)

os.makedirs(results_dir(), exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
N_CLIENTS_LIST   = [20, 50, 100, 200]   # vary n to show scaling
N_FEATURES       = 10
N_REPEATS        = 30                    # repeat each timing
VAL_SIZE         = 2000
SEED             = 42
np.random.seed(SEED)

print("Preparing data for overhead measurement...")
X, y = generate_sc_dataset(50_000, seed=SEED)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_val, y_val = make_validation_set(X_scaled, y, VAL_SIZE, seed=SEED)

THEORETICAL_COMPLEXITY = {
    'FedAvg':      'O(nd)',
    'Multi-Krum':  'O(n²d)',
    'TrimmedMean': 'O(nd log n)',
    'Bulyan':      'O(n²d)',
    'RFVIR':       'O(nd)',
    'FLAME':       'O(nd + n² )',
    'Ours':        'O(nd + n·m)',  # m = |D_val|
}

rows = []
for n_clients in N_CLIENTS_LIST:
    print(f"\n  n_clients = {n_clients}")

    # Generate random updates of shape (n_clients, n_features+1)
    updates = [np.random.randn(N_FEATURES + 1) * 0.01
               for _ in range(n_clients)]
    global_params = np.zeros(N_FEATURES + 1)

    aggregator = MultiLayerAggregator(n_clients)

    def time_method(fn, repeats=N_REPEATS):
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)  # ms
        return np.mean(times), np.std(times)

    # ── Measure each method ──────────────────────────────────────────────────
    methods_fns = {
        'FedAvg':      lambda: fedavg(updates),
        'Multi-Krum':  lambda: krum(updates, f=n_clients//5, m=3),
        'TrimmedMean': lambda: trimmed_mean(updates, trim_ratio=0.1),
        'Bulyan':      lambda: bulyan(updates, f=n_clients//5),
        'RFVIR':       lambda: rfvir(updates, global_params),
        'FLAME':       lambda: flame(updates, noise_std=0.001),
        'Ours':        lambda: aggregator.aggregate(
                            updates, global_params,
                            X_val, y_val, scaler,
                            list(range(n_clients))),
    }

    fedavg_time = None
    for method, fn in methods_fns.items():
        # Warm up
        fn()
        mean_ms, std_ms = time_method(fn)

        if method == 'FedAvg':
            fedavg_time = mean_ms

        overhead_x = mean_ms / fedavg_time if fedavg_time else 1.0

        print(f"    {method:15s}: {mean_ms:7.2f} ± {std_ms:.2f} ms  "
              f"({overhead_x:.1f}× FedAvg)  "
              f"[{THEORETICAL_COMPLEXITY[method]}]")

        rows.append({
            'n_clients':    n_clients,
            'Method':       method,
            'Avg (ms)':     round(mean_ms, 2),
            'Std (ms)':     round(std_ms, 2),
            'Overhead (×)': round(overhead_x, 2),
            'Complexity':   THEORETICAL_COMPLEXITY[method],
        })

df = pd.DataFrame(rows)
df.to_csv(tag('overhead.csv'), index=False)

# ── LaTeX table (n=100 slice, most relevant) ─────────────────────────────────
df_100 = df[df['n_clients'] == 100].copy()

latex = r"""\begin{table}[htbp]
\centering
\scriptsize
\caption{Computational Overhead per Aggregation Round ($n=100$ clients, $d=10$ parameters, $|\mathcal{D}_{val}|=2000$)}
\label{tab:overhead}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Avg Time (ms)} & \textbf{Overhead (×FedAvg)} & \textbf{Complexity} \\
\midrule
"""
for _, row in df_100.iterrows():
    bold_open  = r'\textbf{' if row['Method'] == 'Ours' else ''
    bold_close = r'}'        if row['Method'] == 'Ours' else ''
    latex += (f"{bold_open}{row['Method']}{bold_close} & "
              f"{row['Avg (ms)']} & "
              f"{row['Overhead (×)']} & "
              f"\\texttt{{{row['Complexity']}}} \\\\\n")
latex += r"""\bottomrule
\end{tabular}
\vspace{1mm}
\begin{minipage}{0.45\textwidth}
\scriptsize
$n$: clients per round; $d$: model parameter dimension;
$m$: validation set size. Our method's additional cost over FedAvg
arises from the Kappa forward pass on $\mathcal{D}_{val}$,
which is $O(nm)$ and runs on the server only.
\end{minipage}
\end{table}
"""

with open(tag('overhead_latex.txt'), 'w') as f:
    f.write(latex)

print("\nSaved results/overhead.csv and overhead_latex.txt")
print("[Copy overhead_latex.txt content into Section V of the LaTeX file]")
