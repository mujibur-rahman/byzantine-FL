from fl_model import FLNeuralNet
from fl_base import compute_kappa
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
                 trust_decay=0.05, trust_reward=0.05, trust_thresh=0.3,
                 kappa_val_size=500,
                 use_outlier=True, use_trust=True, use_kappa=True):
        self.tau         = tau          # outlier z-score threshold
        self.alpha       = alpha        # EMA trust decay
        self.delta       = delta        # Kappa threshold
        self.trust_decay = trust_decay
        self.trust_reward= trust_reward
        self.trust_thresh= trust_thresh
        # ── Layer toggles (for the ablation study, run_ablation.py) ────────
        # use_outlier : Layer 1 (coordinate-wise z-score outlier detection)
        # use_trust   : Layer 2 (EMA trust scoring + trust-threshold exclusion)
        # use_kappa   : Layer 3 (Cohen's-Kappa semantic-divergence flag)
        # Disabling a layer removes ITS contribution only: a disabled Layer 2
        # means unflagged clients get weight 1.0 (not their trust score), never
        # 0; a disabled Layer 1/3 means that flag can never fire. All other
        # behaviour (sampling, partition, warm-up, staleness) is unchanged.
        # Defaults are all True, so every non-ablation caller is unaffected.
        self.use_outlier = use_outlier
        self.use_trust   = use_trust
        self.use_kappa   = use_kappa
        # Per-aggregation exclusion log: list of (client_ids, excluded_bool)
        # tuples, one entry per aggregate() call. "excluded" means the client's
        # final aggregation weight was zero (caught by whichever layers are on).
        # Lets an external harness score detection consistently with the active
        # toggles, instead of re-deriving it from raw kappas/flags. Reset here
        # so each fresh aggregator instance starts with an empty log.
        self.flag_log    = []
        # Kappa is evaluated on a fixed random subsample of D_val of this size
        # (the layer cost is n inferences over the val set; subsampling cuts the
        # dominant overhead constant with negligible detection impact). None or
        # 0 => use the full validation set.
        self.kappa_val_size = kappa_val_size
        # Peer-relative Kappa flag: a client is suspicious when its kappa is a
        # lower-tail outlier among the round's peers (kappa < median -
        # k_mad*1.4826*MAD). Dataset-agnostic — replaces the absolute delta,
        # which never gated once kappa was un-saturated. delta is kept as a
        # fallback when there are too few clients for a stable MAD.
        self.k_mad       = 1.5
        # Warm-up: for the first `warmup_rounds` aggregations the global model
        # is still unstable, so kappa is noisy/degenerate and filtering does
        # more harm than good (drops honest clients, slows convergence). During
        # warm-up we skip the outlier/kappa flags and aggregate all clients
        # (still by trust), then engage detection once the model has settled.
        self.warmup_rounds = 5
        self._agg_count    = 0
        self.trust_scores = np.ones(n_clients) * 0.5
        self.n_clients   = n_clients

    def aggregate(self, updates, global_params, X_val, y_val, scaler,
                  client_ids, lr=0.01, stalenesses=None, staleness_a=0.5,
                  ref_params_list=None):
        """
        updates     : list of gradient vectors (one per client)
        global_params : current global model params
        X_val, y_val  : server-held validation set
        scaler        : fitted StandardScaler
        client_ids    : indices into self.trust_scores
        stalenesses : optional list of per-update staleness (server-model
                      versions elapsed since the client trained). When given,
                      each aggregation weight is decayed by
                      (1 + staleness) ** (-staleness_a), i.e. FedBuff/FedAsync
                      staleness weighting. None (default) = synchronous, no
                      decay, so existing sync callers are unaffected.
        staleness_a : decay exponent for the staleness weight (a>0).
        ref_params_list : optional per-update reference params (the global
                      VERSION each client trained on). When given, Kappa for
                      client j is evaluated against ref_params_list[j] instead
                      of the current global_params — staleness-corrected Kappa,
                      so a stale-but-honest client is not flagged for drift it
                      never saw. None (default) = compare against current
                      global (synchronous behaviour, unchanged).
        """
        n_feat = len(global_params) - 1
        n = len(updates)

        # ── Subsample D_val for the Kappa layer (overhead reduction) ──────
        # The Kappa layer runs n inferences over the validation set, so its
        # cost is O(n * |D_val|). A fixed random subsample keeps detection
        # essentially unchanged while cutting that constant.
        Xk, yk = X_val, y_val
        if (X_val is not None and self.kappa_val_size
                and len(X_val) > self.kappa_val_size):
            idx = np.random.RandomState(0).choice(
                len(X_val), self.kappa_val_size, replace=False)
            Xk = X_val[idx]
            yk = None if y_val is None else y_val[idx]

        # ── Layer 1: Outlier detection ────────────────────────────────────
        norms = np.array([np.linalg.norm(u) for u in updates])
        if (not self.use_outlier) or norms.std() < 1e-10:
            outlier_flags = np.zeros(n, dtype=bool)
        else:
            z_scores = np.abs((norms - norms.mean()) / norms.std())
            outlier_flags = z_scores > self.tau

        # ── Layer 2 & 3: Trust + Kappa ────────────────────────────────────
        weights = []
        kappas  = []

        # ── Pass 1: Kappa on each client's ACTUAL local model (base + update),
        # not a tiny lr-scaled step (which saturates kappa near 1 and makes the
        # threshold inert). Reference is the version the client trained on when
        # ref_params_list is supplied (staleness-corrected), else current.
        # When the Kappa layer is disabled (ablation) we skip the computation
        # entirely — it is the dominant per-round cost — and report neutral
        # kappas so the return signature and downstream logging are unchanged.
        if self.use_kappa:
            for j, (u, cid) in enumerate(zip(updates, client_ids)):
                base = (ref_params_list[j] if ref_params_list is not None
                        else global_params)
                kappas.append(compute_kappa(base + u, base, Xk, scaler))
            kappas = np.array(kappas)
            # Peer-relative lower-tail Kappa flag (dataset-agnostic). Falls back
            # to the absolute delta when too few clients for a stable MAD.
            if n >= 5:
                med = np.median(kappas)
                mad = np.median(np.abs(kappas - med)) + 1e-9
                kappa_flag = kappas < (med - self.k_mad * 1.4826 * mad)
            else:
                kappa_flag = kappas < self.delta
        else:
            kappas = np.ones(n)
            kappa_flag = np.zeros(n, dtype=bool)

        # ── Warm-up: while the global model is still unstable, kappa is noisy/
        # degenerate, so filtering drops honest clients and slows convergence.
        # Skip flagging for the first `warmup_rounds` aggregations.
        self._agg_count += 1
        if self._agg_count <= self.warmup_rounds:
            outlier_flags = np.zeros(n, dtype=bool)
            kappa_flag    = np.zeros(n, dtype=bool)

        # ── Pass 2: trust EMA + weight. Kappa DETECTS (via the flag), it does
        # NOT scale honest clients — weighting unflagged clients by raw kappa
        # penalises honest non-IID clients (kappa~0.5-0.7) and wrecks
        # convergence. Unflagged clients are weighted by trust only.
        #
        # Layer toggles: `flagged` collapses only the ACTIVE detection layers
        # (Layer 1 outlier, Layer 3 kappa). Layer 2 (trust) contributes the
        # trust-threshold exclusion and the trust weight; when disabled,
        # unflagged clients get weight 1.0 and there is no trust exclusion, so
        # the layer's multiplicative contribution is exactly 1.0 (never 0).
        excluded = np.zeros(n, dtype=bool)
        for j, cid in enumerate(client_ids):
            flagged = bool((self.use_outlier and outlier_flags[j])
                           or (self.use_kappa and kappa_flag[j]))
            S = 0.0 if flagged else 1.0
            # Keep the trust EMA state current regardless, so toggling trust
            # off then on across an ablation grid never leaves stale state;
            # it only *affects the weight* when use_trust is True.
            self.trust_scores[cid] = (self.alpha * self.trust_scores[cid]
                                      + (1.0 - self.alpha) * S)
            if flagged:
                w = 0.0
            elif self.use_trust:
                w = (0.0 if self.trust_scores[cid] < self.trust_thresh
                     else self.trust_scores[cid])
            else:
                w = 1.0
            weights.append(w)
            excluded[j] = (w == 0.0)

        weights = np.array(weights)
        # Record this round's exclusion decision for external detection scoring
        # (consistent with whichever layers are currently enabled).
        self.flag_log.append((list(client_ids), excluded.copy()))

        # ── Staleness weighting (buffered semi-async / FedBuff) ───────────
        if stalenesses is not None:
            stale = np.asarray(stalenesses, dtype=float)
            weights = weights * (1.0 + stale) ** (-staleness_a)

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


# ══════════════════════════════════════════════════════════════════════════════
# Aggregator wrapper classes
# Each wraps a standalone aggregation function with the same interface
# as MultiLayerAggregator so FederatedServer can use any of them.
#
# Interface:
#   .aggregate(updates, global_params, X_val, y_val, scaler, client_ids)
#   returns (agg_update, outlier_flags, kappas, trust_scores)
# ══════════════════════════════════════════════════════════════════════════════

class _SimpleAggregator:
    """
    Base wrapper — turns a standalone aggregation function into
    a class with the same interface as MultiLayerAggregator.
    outlier_flags, kappas, trust_scores are all None for simple methods.
    """
    delta      = 0.2    # dummy — not used
    trust_thresh = 0.3  # dummy — not used

    def __init__(self, n_clients, **kwargs):
        self.n_clients = n_clients
        self.trust_scores = np.ones(n_clients)

    def _agg(self, updates, global_params):
        raise NotImplementedError

    def aggregate(self, updates, global_params,
                  X_val=None, y_val=None, scaler=None,
                  client_ids=None, lr=0.01,
                  stalenesses=None, staleness_a=0.5, ref_params_list=None):
        # Simple baselines ignore staleness (no staleness-aware weighting);
        # accepted only so the async driver can call them uniformly.
        agg = self._agg(updates, global_params)
        n   = len(updates)
        dummy_flags  = np.zeros(n, dtype=bool)
        dummy_kappas = np.ones(n)
        dummy_trust  = np.ones(n)
        return agg, dummy_flags, dummy_kappas, dummy_trust


class FedAvgAggregator(_SimpleAggregator):
    """FedAvg — Blanchard et al. standard averaging."""
    def _agg(self, updates, global_params):
        return fedavg(updates)


class KrumAggregator(_SimpleAggregator):
    """Multi-Krum — Blanchard et al. NeurIPS 2017."""
    def __init__(self, n_clients, f=None, m=3, **kwargs):
        super().__init__(n_clients)
        self.f = f or max(1, n_clients // 5)
        self.m = m

    def _agg(self, updates, global_params):
        return krum(updates, f=self.f, m=self.m)


class TrimmedMeanAggregator(_SimpleAggregator):
    """Coordinate-wise trimmed mean — Yin et al. ICML 2018."""
    def __init__(self, n_clients, trim_ratio=0.1, **kwargs):
        super().__init__(n_clients)
        self.trim_ratio = trim_ratio

    def _agg(self, updates, global_params):
        return trimmed_mean(updates, trim_ratio=self.trim_ratio)


class BulyanAggregator(_SimpleAggregator):
    """Bulyan — El Mhamdi et al. ICML 2018."""
    def __init__(self, n_clients, f=None, **kwargs):
        super().__init__(n_clients)
        self.f = f or max(1, n_clients // 5)

    def _agg(self, updates, global_params):
        return bulyan(updates, f=self.f)


class RFVIRAggregator(_SimpleAggregator):
    """RFVIR — Wang et al. Information Fusion 2024."""
    def _agg(self, updates, global_params):
        return rfvir(updates, global_params)


class FLAMEAggregator(_SimpleAggregator):
    """FLAME — Nguyen et al. USENIX Security 2022."""
    def __init__(self, n_clients, noise_std=0.001, **kwargs):
        super().__init__(n_clients)
        self.noise_std = noise_std

    def _agg(self, updates, global_params):
        return flame(updates, noise_std=self.noise_std)


# Registry: method name → aggregator class
AGGREGATOR_REGISTRY = {
    'FedAvg':      FedAvgAggregator,
    'Multi-Krum':  KrumAggregator,
    'TrimmedMean': TrimmedMeanAggregator,
    'Bulyan':      BulyanAggregator,
    'RFVIR':       RFVIRAggregator,
    'FLAME':       FLAMEAggregator,
    'Ours':        MultiLayerAggregator,
}

print("aggregators.py loaded successfully.")
