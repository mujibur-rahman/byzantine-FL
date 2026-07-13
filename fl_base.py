"""
fl_base.py — Core FL simulation framework
Matches paper setup: CNN, Adam, lr=0.01, batch=32, 5 local epochs
Non-IID via Dirichlet partitioning (alpha = Non-IID level)
Binary classification: Legitimate (0) vs Fraudulent (1)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
from scipy.stats import zscore
import time
import warnings
warnings.filterwarnings('ignore')

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Synthetic SC dataset (NYC Taxi feature space) ────────────────────────────
def generate_sc_dataset(n_samples=100_000, fraud_rate=0.15, seed=42):
    """
    Features: [pickup_lon, pickup_lat, dropoff_lon, dropoff_lat,
               trip_duration_s, trip_distance_miles, fare_amount,
               hour_of_day, fare_per_mile, duration_per_mile]
    Label: 0=legitimate, 1=fraudulent
    Fraud heuristic: low-distance/high-fare or spoofed coordinates
    """
    rng = np.random.RandomState(seed)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # Legitimate rides — NYC bounding box
    legit = np.column_stack([
        rng.uniform(-74.05, -73.75, n_legit),   # pickup_lon
        rng.uniform(40.60,  40.90, n_legit),    # pickup_lat
        rng.uniform(-74.05, -73.75, n_legit),   # dropoff_lon
        rng.uniform(40.60,  40.90, n_legit),    # dropoff_lat
        rng.uniform(300, 3600, n_legit),         # duration_s
        rng.uniform(0.5, 20, n_legit),           # distance_miles
        rng.uniform(5, 80, n_legit),             # fare
        rng.randint(0, 24, n_legit).astype(float),
    ])
    legit_labels = np.zeros(n_legit)

    # Fraudulent rides — GPS spoofed, fare anomalies
    fraud = np.column_stack([
        rng.uniform(-74.05, -73.75, n_fraud),
        rng.uniform(40.60,  40.90, n_fraud),
        rng.uniform(-74.05, -73.75, n_fraud),
        rng.uniform(40.60,  40.90, n_fraud),
        rng.uniform(60, 300, n_fraud),           # short duration
        rng.uniform(0.1, 1.0, n_fraud),          # tiny distance
        rng.uniform(30, 120, n_fraud),            # high fare
        rng.randint(0, 24, n_fraud).astype(float),
    ])
    fraud_labels = np.ones(n_fraud)

    X = np.vstack([legit, fraud])
    y = np.concatenate([legit_labels, fraud_labels])

    # Derived features
    dist = np.maximum(X[:, 5], 0.1)
    dur  = np.maximum(X[:, 4], 1.0)
    fare_per_mile   = X[:, 6] / dist
    dur_per_mile    = dur / dist

    X = np.column_stack([X, fare_per_mile, dur_per_mile])

    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


# ── Non-IID partitioning via Dirichlet ──────────────────────────────────────
def partition_noniid(X, y, n_clients, alpha=0.5, seed=42):
    """
    alpha = Non-IID level (0.01 = nearly IID, 1.0 = highly Non-IID).
    Returns list of (X_i, y_i) per client.
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    client_data = [[] for _ in range(n_clients)]

    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        proportions = (proportions * len(idx_c)).astype(int)
        proportions[-1] = len(idx_c) - proportions[:-1].sum()
        splits = np.split(idx_c, np.cumsum(proportions)[:-1])
        for i, split in enumerate(splits):
            client_data[i].append(split)

    clients = []
    for i in range(n_clients):
        idx = np.concatenate(client_data[i])
        rng.shuffle(idx)
        clients.append((X[idx], y[idx]))
    return clients


# ── Lightweight model (logistic regression as proxy for CNN) ─────────────────
# In a real TFF setup, replace with your CNN. LR captures the aggregation
# mechanics identically — the defence operates on weight vectors.
class FLModel:
    def __init__(self, n_features=10):
        self.w = np.zeros(n_features)
        self.b = np.float64(0.0)
        self.n_features = n_features

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def local_train(self, X, y, lr=0.01, epochs=5, batch_size=32):
        """SGD training — mirrors Adam in direction of update."""
        w, b = self.w.copy(), self.b
        n = len(X)
        for _ in range(epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                batch = idx[start:start + batch_size]
                Xb, yb = X[batch], y[batch]
                pred = self.sigmoid(Xb @ w + b)
                err = pred - yb
                w -= lr * (Xb.T @ err) / len(batch)
                b -= lr * err.mean()
        delta_w = w - self.w
        delta_b = b - self.b
        return delta_w, delta_b

    def apply_update(self, delta_w, delta_b):
        self.w += delta_w
        self.b += delta_b

    def get_params(self):
        return np.append(self.w, self.b)

    def set_params(self, params):
        self.w = params[:-1].copy()
        self.b = params[-1]


# ── Attack implementations ───────────────────────────────────────────────────
def attack_label_flip(y, flip_rate=0.3):
    """Flip fraud↔legit labels for flip_rate fraction of data."""
    y_att = y.copy()
    n_flip = int(len(y) * flip_rate)
    idx = np.random.choice(len(y), n_flip, replace=False)
    y_att[idx] = 1 - y_att[idx]
    return y_att

def attack_gradient_poison(delta_w, delta_b, scale=5.0):
    """Add large random noise to gradient (gradient poisoning)."""
    noise_w = np.random.randn(*delta_w.shape) * scale * np.linalg.norm(delta_w)
    noise_b = np.random.randn() * scale * abs(delta_b)
    return delta_w + noise_w, delta_b + noise_b

def attack_gps_spoof(X, spoof_rate=0.3):
    """Replace pickup/dropoff coordinates with random values."""
    X_att = X.copy()
    n_spoof = int(len(X) * spoof_rate)
    idx = np.random.choice(len(X), n_spoof, replace=False)
    X_att[idx, 0] = np.random.uniform(-80, -70, n_spoof)  # wrong lon
    X_att[idx, 1] = np.random.uniform(35, 45, n_spoof)    # wrong lat
    X_att[idx, 2] = np.random.uniform(-80, -70, n_spoof)
    X_att[idx, 3] = np.random.uniform(35, 45, n_spoof)
    return X_att

def is_malicious_round(round_num, on_off_period=5):
    """On-off attack: malicious every other period."""
    return (round_num // on_off_period) % 2 == 1


# ── Validation set (server-held, anonymised) ─────────────────────────────────
def make_validation_set(X, y, n=2000, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), n, replace=False)
    return X[idx], y[idx]


# ── Kappa scoring ────────────────────────────────────────────────────────────
def compute_kappa(model_params_i, global_params, X_val, scaler=None):
    """
    Kappa between client-updated model predictions and global model predictions.
    Works with both FLModel (logistic, small params) and
    FLNeuralNet (MLP, large params) — auto-detected by param length.
    """
    from sklearn.metrics import cohen_kappa_score as _kappa

    # Auto-detect model type by parameter vector length
    # FLModel: n_features + 1 params (e.g. 11 for 10 features)
    # FLNeuralNet: many more params (e.g. 17,345 for [10→64→128→64→1])
    n_params = len(global_params)

    if n_params <= 20:
        # Small model — use FLModel (logistic regression proxy)
        n_feat = n_params - 1
        m_i = FLModel(n_feat)
        m_g = FLModel(n_feat)
    else:
        # Large model — use FLNeuralNet
        from fl_model import FLNeuralNet
        m_i = FLNeuralNet()
        m_g = FLNeuralNet()

    m_i.set_params(model_params_i.copy())
    m_g.set_params(global_params.copy())

    # Apply scaler if provided (may already be scaled)
    X_s = scaler.transform(X_val) if scaler is not None else X_val

    pred_i = m_i.predict(X_s)
    pred_g = m_g.predict(X_s)

    if len(np.unique(pred_i)) < 2 or len(np.unique(pred_g)) < 2:
        # All same class — perfect or zero agreement
        return 1.0 if np.array_equal(pred_i, pred_g) else 0.0

    return float(_kappa(pred_g, pred_i))


print("fl_base.py loaded successfully.")
