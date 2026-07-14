# Byzantine-Resilient FL — Project Documentation

**Paper:** *Semantic Divergence Detection via Cohen's Kappa for
Byzantine-Resilient Federated Learning in Spatial Crowdsourcing Systems*
**Target journal:** Information Fusion (Elsevier)
**Status:** Experiments running — results pending for 4 datasets

---

## Project structure

```
byzantine-FL/
├── CLAUDE.md                        ← this file
│
├── Core FL framework
│   ├── fl_model.py                  ← neural network model (FLNeuralNet)
│   ├── fl_base.py                   ← shared utilities, attacks, compute_kappa
│   ├── fl_client.py                 ← FLClient class (local training)
│   ├── fl_server.py                 ← FederatedServer class (defence + aggregation)
│   ├── fl_runner.py                 ← build_federation() + run_federation()
│   └── aggregators.py               ← all 7 aggregation methods + wrapper classes
│
├── Dataset loaders
│   ├── dataset_config.py            ← central config, get_dataset(), tag(), results_dir()
│   ├── nyc_taxi_loader.py           ← NYC TLC FHVHV parquet loader
│   ├── geolife_loader.py            ← Microsoft Geolife PLT trajectory loader
│   ├── foursquare_loader.py         ← Foursquare TSMC2014 check-in loader
│   └── yelp_loader.py               ← Yelp business + check-in JSON loader
│
├── Experiments
│   ├── experiment1_baselines.py     ← Table VI: 7-method comparison + FLAME
│   ├── experiment2_sensitivity.py   ← Table IX: τ, α, δ sensitivity sweep
│   ├── experiment3_layer_analysis.py← Table X: per-layer detection breakdown
│   ├── experiment4_overhead.py      ← Table XI: computational overhead
│   ├── experiment5_adaptive.py      ← Table XII: sleeper/mimicry/slow-drift
│   ├── experiment6_semantic_divergence.py ← Table XIII: core Kappa claim proof
│   └── run_all.sh                   ← runs all 6 experiments for one dataset
│
├── Utilities
│   └── find_datasets.py             ← auto-detect dataset paths on disk
│
└── Paper (LaTeX)
    ├── byzantine-FL-final.tex       ← current manuscript (use this one)
    ├── cover_letter.tex             ← Information Fusion cover letter
    ├── highlights.tex               ← Elsevier 5-bullet highlights
    └── submission_checklist.txt     ← pre-submission checklist
```

---

## Architecture

### How the federated system works

```
FederatedServer
    │
    │  broadcast(global_params)
    ▼
FLClient × N                         (private data never leaves)
    │  local_train_and_submit()
    │  → update = local_params - global_params
    ▼
FederatedServer.receive_update()
    │
    ▼
FederatedServer.aggregate()
    │
    ├── Layer 1: Outlier Detection
    │   coordinate-wise median, z-score threshold τ=2.0
    │   targets: gradient poisoning, Sybil attacks
    │
    ├── Layer 2: Trust Scoring
    │   EMA trust: T_i(t) = α·T_i(t−1) + (1−α)·S_i(t), α=0.8
    │   targets: on-off attacks, drifting adversaries
    │
    └── Layer 3: Kappa Filtering  ← PRIMARY CONTRIBUTION
        Cohen's Kappa between client model and global model on D_val
        κ_i = (p_o − p_e) / (1 − p_e), threshold δ=0.2
        targets: label flipping, semantic poisoning
        detection latency: 1 round (vs ~6 rounds for trust scoring)
    │
    ▼
Aggregation: θ_{t+1} = Σ W_i·θ_i(t) / Σ W_i
    where W_i = T_i(t) · max(κ_i, 0)  if not flagged
           W_i = 0                      if flagged
```

### Key architectural decision

`FLClient.local_train_and_submit()` returns only the gradient update
vector. Raw data never leaves the client. Byzantine clients apply
attacks inside this method before returning:
- `lfp`   — label flipping (data-level, before training)
- `spf`   — GPS spoofing (data-level, feature corruption)
- `ooa`   — on-off (data-level, alternating rounds)
- `grad-p` — gradient poisoning (update-level, after training)

---

## Model

**FLNeuralNet** (`fl_model.py`)
- Architecture: `10 → 64 → 128 → 64 → 1`
- Activations: ReLU (hidden), Sigmoid (output)
- Optimiser: Adam (β₁=0.9, β₂=0.999)
- Parameters: 17,345
- Matches paper: 3-layer CNN adapted for tabular SC data

```python
from fl_model import FLNeuralNet
net = FLNeuralNet(n_features=10)
net.train(X, y, lr=0.01, epochs=5, batch_size=32)
pred = net.predict(X)
params = net.get_params()    # flat vector, length 17345
net.set_params(params)       # restore from flat vector
```

---

## Datasets

All four datasets return the same interface:
```python
X, y = get_dataset(seed=42)
# X: (n, 10) float array
# y: (n,)    binary label (0=legitimate, 1=fraudulent)
```

**Feature space (consistent across all datasets):**

| Index | Feature | NYC Taxi | Geolife | Foursquare | Yelp |
|-------|---------|----------|---------|------------|------|
| 0 | pickup_lat | GPS lat | traj start lat | check-in A lat | biz A lat |
| 1 | pickup_lon | GPS lon | traj start lon | check-in A lon | biz A lon |
| 2 | dropoff_lat | GPS lat | traj end lat | check-in B lat | biz B lat |
| 3 | dropoff_lon | GPS lon | traj end lon | check-in B lon | biz B lon |
| 4 | duration_s | trip time | traj duration | time between check-ins | dist/speed |
| 5 | distance | trip miles | km displacement | km between venues | km between biz |
| 6 | cost_proxy | fare ($) | speed (km/h) | category score | rating × price |
| 7 | hour_of_day | pickup hour | start hour | check-in hour | peak hour |
| 8 | cost_per_unit | fare/mile | speed/km | dist/hour | demand density |
| 9 | duration_per_unit | dur/mile | disp ratio | check-in density | review velocity |

**Expected fraud rates:**

| Dataset | Fraud rate | Geographic distribution |
|---------|-----------|------------------------|
| NYC Taxi | 15% | New York City |
| Geolife | 12% | Beijing |
| Foursquare | 13% | New York City (check-ins) |
| Yelp | 13% | Las Vegas, Phoenix, Toronto, Charlotte, Pittsburgh |

**Download paths** (edit in `dataset_config.py` → `DATASET_REGISTRY`):
```
~/data/nyc-taxi/fhvhv_tripdata_2025-01.parquet
~/data/nyc-taxi/taxi_zone_lookup.csv
~/data/geolife/Data/
~/data/foursquare/dataset_TSMC2014_NYC.txt
~/data/yelp/yelp_academic_dataset_business.json
~/data/yelp/yelp_academic_dataset_checkin.json
```

If real data not found → synthetic fallback runs automatically.
To find where your files are: `python3 find_datasets.py`

---

## Running experiments

### Setup

```bash
pip install numpy pandas scikit-learn scipy pyarrow fastparquet
```

### Run one dataset

```bash
bash run_all.sh --dataset nyc-taxi      # or geolife / foursquare / yelp
```

### Run individual experiments

```bash
python3 experiment1_baselines.py        --dataset nyc-taxi
python3 experiment2_sensitivity.py      --dataset nyc-taxi
python3 experiment3_layer_analysis.py   --dataset nyc-taxi
python3 experiment4_overhead.py         --dataset nyc-taxi
python3 experiment5_adaptive.py         --dataset nyc-taxi
python3 experiment6_semantic_divergence.py --dataset nyc-taxi
```

### Output structure

All results are tagged with the dataset name:
```
results/
├── nyc-taxi/
│   ├── nyc-taxi_table_baselines.csv          → Table VI
│   ├── nyc-taxi_sensitivity_tau.csv          → Table IX (τ sweep)
│   ├── nyc-taxi_sensitivity_alpha.csv        → Table IX (α sweep)
│   ├── nyc-taxi_sensitivity_delta.csv        → Table IX (δ sweep)
│   ├── nyc-taxi_layer_confusion.csv          → Table X
│   ├── nyc-taxi_layer_confusion_latex.txt    → paste into paper
│   ├── nyc-taxi_overhead.csv                 → Table XI
│   ├── nyc-taxi_overhead_latex.txt           → paste into paper
│   ├── nyc-taxi_adaptive_*.csv               → Table XII
│   ├── nyc-taxi_adaptive_detection_latency.csv
│   ├── nyc-taxi_adaptive_latex.txt           → paste into paper
│   ├── nyc-taxi_semantic_divergence_summary.csv  → Table XIII
│   └── nyc-taxi_semantic_divergence_latex.txt    → paste into paper
├── geolife/
├── foursquare/
└── yelp/
```

---

## Experiment descriptions

### Experiment 1 — Baseline comparison (`experiment1_baselines.py`)
Runs all 7 aggregation methods through the proper `FederatedServer`/`FLClient`
architecture. Each method is tested against 4 attack types at 20% Byzantine.

**Methods:** FedAvg, Multi-Krum, TrimmedMean, Bulyan, RFVIR, FLAME, Ours

**Key citations:**
- FedAvg: McMahan et al. ICML 2017
- Krum: Blanchard et al. NeurIPS 2017
- TrimmedMean: Yin et al. ICML 2018
- Bulyan: El Mhamdi et al. ICML 2018
- RFVIR: Wang et al. Information Fusion 2024
- FLAME: Nguyen et al. USENIX Security 2022
- Ours: this paper

### Experiment 2 — Sensitivity analysis (`experiment2_sensitivity.py`)
Sweeps each hyperparameter independently while holding others at default.

| Parameter | Default | Range tested |
|-----------|---------|-------------|
| τ (outlier z-score) | 2.0 | 1.0, 1.5, 2.0, 2.5, 3.0 |
| α (trust decay) | 0.8 | 0.5, 0.6, 0.7, 0.8, 0.9 |
| δ (Kappa threshold) | 0.2 | 0.10, 0.15, 0.20, 0.25, 0.30 |

Attack: label flipping (lfp), 20% Byzantine.

### Experiment 3 — Per-layer detection (`experiment3_layer_analysis.py`)
Records which layer catches each Byzantine client independently.
**Key output column:** `Kappa-only TPs` — Byzantine clients that passed
gradient filter and trust but were caught by Kappa.
This is the empirical proof of **Corollary 1**.

### Experiment 4 — Overhead (`experiment4_overhead.py`)
Wall-clock timing per aggregation round across client counts 20/50/100/200.
Reports complexity comparison: our O(nd + nm) vs Krum/Bulyan O(n²d).

### Experiment 5 — Adaptive adversary (`experiment5_adaptive.py`)
Three attack variants designed to evade gradient-level defences:
- **Sleeper**: honest for 20 rounds, then activates label flipping
- **Mimicry**: label flipping + gradient clipped to benign norm envelope
- **Slow drift**: flip rate ramps 5%/round from 0% to 40%

**Key result:** Kappa detects sleeper activation in 1 round.
Trust scoring requires ~6 rounds (derived in Remark 2, Section IV).

### Experiment 6 — Semantic divergence (`experiment6_semantic_divergence.py`)
The core empirical proof of the primary claim. For every client every round:
- `κ_i` — Kappa score (detection signal)
- `semantic_shift` — |p_fraud_client − p_fraud_global| on D_val
- `gradient_dist` — ‖Δw_i − w_med‖₂ (gradient-level signal)

**Demonstrates:** for label flipping and GPS spoofing, gradient distance
remains low while semantic shift becomes high → Kappa catches what
gradient filters cannot. Directly proves Corollary 1 empirically.

---

## Formal contributions (from paper)

**Proposition 1 (Kappa-Divergence Bound):**
If client i induces prediction disagreement d_i ≥ ε, then:
```
κ_i ≤ 1 − ε / (1 − p_e)  where p_e = Σ p_k²
```
Setting δ = 1 − ε/(1 − p_e) guarantees detection in 1 round.

**Corollary 1 (Complementarity):**
There exist adversaries that bypass gradient filters yet are caught by
Kappa — specifically, any attacker with gradient norm within the benign
envelope but semantic prediction divergence above ε.

**Remark 2 (Adaptive adversary latency):**
Trust scoring requires k* ≈ ⌈log(0.3)/log(0.8)⌉ = 6 rounds after
a sleeper activates before excluding them. Kappa detects in 1 round
(stateless — recomputed fresh each round, independent of trust history).

---

## Aggregation methods (API)

All methods share the same interface via wrapper classes in `aggregators.py`:

```python
from aggregators import AGGREGATOR_REGISTRY

agg = AGGREGATOR_REGISTRY['Ours'](n_clients=100, tau=2.0, alpha=0.8, delta=0.2)
# or: 'FedAvg', 'Multi-Krum', 'TrimmedMean', 'Bulyan', 'RFVIR', 'FLAME'

agg_update, outlier_flags, kappas, trust_scores = agg.aggregate(
    updates,       # list of gradient vectors
    global_params, # current global model params
    X_val, y_val,  # server-held validation set D_val
    scaler,        # fitted StandardScaler
    client_ids,    # list of client indices
)
```

Simple methods (FedAvg, Krum, etc.) return `None` for
`outlier_flags`, `kappas`, `trust_scores` — they do not
perform semantic analysis.

---

## Using the server/client API directly

```python
from fl_runner import build_federation, run_federation
from aggregators import AGGREGATOR_REGISTRY

server, clients, byz_ids = build_federation(
    X_scaled, y, scaler, X_val, y_val,
    n_clients         = 100,
    clients_per_round = 20,
    byzantine_ratio   = 0.20,
    attack_type       = 'lfp',     # 'lfp' | 'grad-p' | 'spf' | 'ooa'
    noniid_alpha      = 0.5,
    lr                = 0.01,
    local_epochs      = 5,
    async_mode        = False,
    seed              = 42,
)

# Swap aggregator to compare methods
server.aggregator = AGGREGATOR_REGISTRY['FedAvg'](n_clients=100)

history = run_federation(
    server, clients, set(byz_ids),
    n_rounds = 50,
    p_churn  = 0.10,   # 10% client dropout per round
    verbose  = True,
    seed     = 42,
)

# history is a list of per-round dicts:
# { round, accuracy, f1, n_flagged, mean_kappa, mean_trust, agg_time_ms }
```

---

## Known issues / watch points

| Issue | File | Notes |
|-------|------|-------|
| Geolife slow with real data | `geolife_loader.py` | Uses fast mode (start+end only). Increase `max_files_per_user` if more records needed. |
| Dataset paths not found | `dataset_config.py` | Run `find_datasets.py` to auto-detect paths on your machine. |
| Kappa near zero in early rounds | `aggregators.py` | Expected — global model predicts all-one-class early. Kappa stabilises by round 5–10. |
| Neural net slower than LR proxy | `fl_model.py` | ~17K params vs 11. Reduce `LOCAL_EPOCHS` or `N_CLIENTS` for quick tests. |
| Foursquare: few trip pairs | `foursquare_loader.py` | Increase `max_gap_hours` or `n_pairs_per_biz` if record count is low. |

---

## Paper → code mapping

| Paper section | Code location |
|---------------|--------------|
| Section III — FL workflow (Fig 1) | `fl_server.py` → `run_round()` |
| Section III — Byzantine attacks | `fl_client.py` → `local_train_and_submit()` |
| Section IV.B — Outlier detection (Eq 1,2) | `aggregators.py` → `MultiLayerAggregator.aggregate()` Layer 1 |
| Section IV.C — Trust scoring (Eq 3,4) | `aggregators.py` → `MultiLayerAggregator.aggregate()` Layer 2 |
| Section IV.D — Kappa filtering (Eq 5) | `fl_base.py` → `compute_kappa()` |
| Section IV.E — Integrated pipeline (Alg 1) | `aggregators.py` → `MultiLayerAggregator.aggregate()` |
| Section IV — Proposition 1 | `fl_base.py` → `compute_kappa()` + `aggregators.py` threshold `delta` |
| Section IV — Remark 2 (latency) | `experiment5_adaptive.py` → `run_adaptive()` |
| Section V — Datasets | `dataset_config.py` + `*_loader.py` files |
| Section V — Attack scenarios (Table II) | `fl_base.py` → `attack_*()` functions |
| Section VI — Table VI | `experiment1_baselines.py` |
| Section VI — Table IX | `experiment2_sensitivity.py` |
| Section VI — Table X | `experiment3_layer_analysis.py` |
| Section VI — Table XI | `experiment4_overhead.py` |
| Section VI — Table XII | `experiment5_adaptive.py` |
| Section VI — Table XIII | `experiment6_semantic_divergence.py` |

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_clients | 100 (sim) / 20K–50K (paper) | Total clients in federation |
| clients_per_round | 20 | Sampled per round |
| local_epochs | 5 | Epochs of local training |
| lr | 0.01 | Learning rate (Adam) |
| batch_size | 32 | Mini-batch size |
| n_rounds | 50 | FL training rounds |
| noniid_alpha | 0.5 | Dirichlet Non-IID level |
| byzantine_ratio | 0.20 | Fraction of malicious clients |
| τ (tau) | 2.0 | Outlier z-score threshold |
| α (alpha) | 0.8 | Trust decay factor |
| δ (delta) | 0.2 | Kappa detection threshold |
| trust_thresh | 0.3 | Trust exclusion threshold |
| val_size | 2,000 | Server validation set size |

---

## Dependencies

```
numpy >= 1.24
pandas >= 2.0
scikit-learn >= 1.3
scipy >= 1.11
pyarrow >= 14.0      # for NYC Taxi parquet
fastparquet >= 2023  # optional parquet backend
```

Install: `pip install numpy pandas scikit-learn scipy pyarrow fastparquet`

---

## After experiments complete

1. Replace all `--` placeholders in `byzantine-FL-final.tex` with CSV values
2. Paste `*_latex.txt` content into corresponding table environments
3. Replace Yelp with Foursquare dataset label in Section V
4. Convert `\documentclass{IEEEtran}` → `\documentclass[review]{elsarticle}`
5. Add 8 new BibTeX entries (listed at bottom of `.tex` file as comments)
6. Fill in `cover_letter.tex` (name, institution, ORCID, funding)
7. Submit to Information Fusion via Editorial Manager

See `submission_checklist.txt` for the full pre-submission checklist.
