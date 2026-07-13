"""
aggregators.py — All aggregation methods
FedAvg, Krum, TrimmedMean, Bulyan, RFVIR, FLAME, Ours (Multi-Layer)
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score
import time

# ── FedAvg ───────────────────────────────────────────────────────────────────
def fedavg(updates, weights=None):
    if weights is None:
        weights = np.ones(len(updates)) / len(updates)
    weights = np.array(weights) / np.sum(weights)
    return sum(w * u for w, u in zip(weights, updates))


# ── Coordinate-wise Median ───────────────────────────────────────────────────
def coord_median(updates):
    return np.median(np.vstack(updates), axis=0)


# ── Krum ─────────────────────────────────────────────────────────────────────
def krum(updates, f=None, m=1):
    """
    f: number of Byzantine clients (if None, estimated as len//4)
    m: number of updates to select (Multi-Krum if m>1)
    Blanchard et al. NeurIPS 2017.
    """
    n = len(updates)
    if f is None:
        f = n // 4
    scores = []
    U = np.vstack(updates)
    for i in range(n):
        dists = np.sum((U - U[i]) ** 2, axis=1)
        dists[i] = np.inf
        nearest = np.sort(dists)[:n - f - 2]
        scores.append(nearest.sum())
    selected = np.argsort(scores)[:m]
    return np.mean(U[selected], axis=0)


# ── Trimmed Mean ─────────────────────────────────────────────────────────────
def trimmed_mean(updates, trim_ratio=0.1):
    """
    Coordinate-wise trimmed mean.
    Yin et al. ICML 2018.
    """
    U = np.vstack(updates)
    n = len(updates)
    k = max(1, int(n * trim_ratio))
    sorted_U = np.sort(U, axis=0)
    trimmed = sorted_U[k:n-k, :]
    return trimmed.mean(axis=0)


# ── Bulyan ───────────────────────────────────────────────────────────────────
def bulyan(updates, f=None):
    """
    El Mhamdi et al. ICML 2018.
    Step 1: Krum selection of n-2f updates.
    Step 2: Coordinate-wise trimmed mean of selected.
    """
    n = len(updates)
    if f is None:
        f = n // 4
    m = n - 2 * f
    m = max(1, m)
    # Step 1: Multi-Krum to select m updates
    U = np.vstack(updates)
    scores = []
    for i in range(n):
        dists = np.sum((U - U[i]) ** 2, axis=1)
        dists[i] = np.inf
        nearest = np.sort(dists)[:n - f - 2]
        scores.append(nearest.sum())
    selected_idx = np.argsort(scores)[:m]
    selected = U[selected_idx]
    # Step 2: Trimmed mean
    k = max(1, int(m * 0.1))
    sorted_sel = np.sort(selected, axis=0)
    if 2*k < m:
        trimmed = sorted_sel[k:m-k, :]
    else:
        trimmed = sorted_sel
    return trimmed.mean(axis=0)


# ── RFVIR ────────────────────────────────────────────────────────────────────
def rfvir(updates, global_params, threshold_percentile=70):
    """
    Wang et al. Information Fusion 2024.
    Simplified: cosine similarity + gram matrix filtering.
    Clients below similarity threshold are excluded.
    """
    U = np.vstack(updates)
    g = global_params

    sims = []
    for u in updates:
        norm_u = np.linalg.norm(u)
        norm_g = np.linalg.norm(g)
        if norm_u < 1e-10 or norm_g < 1e-10:
            sims.append(0.0)
        else:
            sims.append(np.dot(u, g) / (norm_u * norm_g))

    sims = np.array(sims)
    threshold = np.percentile(sims, 100 - threshold_percentile)
    keep = sims >= threshold
    if keep.sum() == 0:
        keep = np.ones(len(updates), dtype=bool)
    return U[keep].mean(axis=0)


# ── FLAME ────────────────────────────────────────────────────────────────────
def flame(updates, noise_std=0.001, clip_norm=None):
    """
    Nguyen et al. USENIX Security 2022.
    Step 1: HDBSCAN clustering — simplified here with cosine-distance
            based k-means (2 clusters: benign / malicious).
    Step 2: Select largest cluster.
    Step 3: Clip updates to median norm.
    Step 4: Add Gaussian noise scaled to clip_norm.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    U = np.vstack(updates)
    U_norm = normalize(U)

    # Cluster into 2 groups (benign / adversarial)
    if len(updates) < 4:
        return U.mean(axis=0)

    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = km.fit_predict(U_norm)

    # Keep the larger cluster (assumed benign majority)
    counts = np.bincount(labels)
    benign_label = np.argmax(counts)
    selected = U[labels == benign_label]

    # Clip to median norm
    norms = np.linalg.norm(selected, axis=1)
    clip = np.median(norms)
    if clip_norm is not None:
        clip = min(clip, clip_norm)

    clipped = []
    for row in selected:
        n = np.linalg.norm(row)
        if n > clip:
            clipped.append(row * clip / n)
        else:
            clipped.append(row)

    agg = np.mean(clipped, axis=0)

    # Add noise
    agg += np.random.randn(*agg.shape) * noise_std

    return agg


# ── Ours: Multi-Layer (Outlier + Trust + Kappa) ──────────────────────────────
class MultiLayerAggregator:
    """
    Three-layer Byzantine-resilient aggregation.
    Layer 1: Coordinate-wise median outlier detection (z-score threshold τ)
    Layer 2: Exponential moving average trust scoring (decay α)
    Layer 3: Cohen's Kappa semantic divergence filtering (threshold δ)
    """

    def __init__(self, n_clients, tau=2.0, alpha=0.8, delta=0.2,
                 trust_decay=0.05, trust_reward=0.05, trust_thresh=0.3):
        self.tau         = tau          # outlier z-score threshold
        self.alpha       = alpha        # EMA trust decay
        self.delta       = delta        # Kappa threshold
        self.trust_decay = trust_decay
        self.trust_reward= trust_reward
        self.trust_thresh= trust_thresh
        self.trust_scores = np.ones(n_clients) * 0.5
        self.n_clients   = n_clients

    def aggregate(self, updates, global_params, X_val, y_val, scaler,
                  client_ids, lr=0.01):
        """
        updates     : list of gradient vectors (one per client)
        global_params : current global model params
        X_val, y_val  : server-held validation set
        scaler        : fitted StandardScaler
        client_ids    : indices into self.trust_scores
        """
        from fl_base import FLModel, compute_kappa
        n_feat = len(global_params) - 1
        n = len(updates)

        # ── Layer 1: Outlier detection ────────────────────────────────────
        norms = np.array([np.linalg.norm(u) for u in updates])
        if norms.std() < 1e-10:
            outlier_flags = np.zeros(n, dtype=bool)
        else:
            z_scores = np.abs((norms - norms.mean()) / norms.std())
            outlier_flags = z_scores > self.tau

        # ── Layer 2 & 3: Trust + Kappa ────────────────────────────────────
        weights = []
        kappas  = []

        for j, (u, cid) in enumerate(zip(updates, client_ids)):
            # Kappa: apply this client's update to global model
            candidate_params = global_params.copy()
            candidate_params += lr * u
            kappa = compute_kappa(candidate_params, global_params,
                                  X_val, scaler)
            kappas.append(kappa)

            # Trust update
            if outlier_flags[j] or kappa < self.delta:
                self.trust_scores[cid] = max(
                    0.0, self.trust_scores[cid] - self.trust_decay)
            else:
                self.trust_scores[cid] = min(
                    1.0, self.trust_scores[cid] + self.trust_reward)

            # Aggregation weight
            if outlier_flags[j] or self.trust_scores[cid] < self.trust_thresh:
                weights.append(0.0)
            else:
                weights.append(self.trust_scores[cid] * max(kappa, 0.0))

        weights = np.array(weights)
        kappas  = np.array(kappas)

        if weights.sum() < 1e-10:
            # Fallback: use all non-outlier updates equally
            weights = (~outlier_flags).astype(float)
            if weights.sum() < 1e-10:
                weights = np.ones(n)

        weights /= weights.sum()

        U = np.vstack(updates)
        agg = (weights[:, None] * U).sum(axis=0)

        return agg, outlier_flags, kappas, self.trust_scores[client_ids].copy()


print("aggregators.py loaded successfully.")
