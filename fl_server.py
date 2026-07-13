"""
fl_server.py
============
Federated Learning Server.

Responsibilities:
  - Holds the global model
  - Selects clients each round
  - Receives client updates
  - Runs the three-layer defence pipeline
  - Aggregates into the updated global model
  - Broadcasts the updated model back to clients

In simulation mode (same process), communication is via
in-memory objects. The class structure mirrors what would
be a real gRPC/REST boundary in production deployment.
"""

import numpy as np
import time
from collections import deque
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from fl_client import FLClient
from fl_model  import FLNeuralNet
from aggregators import MultiLayerAggregator


class FederatedServer:
    """
    Central federated server — honest-but-unaware.

    Parameters
    ----------
    n_features      : int   Input feature dimension
    n_clients       : int   Total number of registered clients
    clients_per_round: int  How many clients to sample each round
    aggregator      : MultiLayerAggregator (or any aggregator)
    X_val, y_val    : np.ndarray  Server-held validation set D_val
    scaler          : fitted StandardScaler
    lr              : float  Global learning rate for applying updates
    async_mode      : bool  If True, accept updates with staleness
    max_staleness   : int   Max rounds a stale update is accepted
    """

    def __init__(self, n_features, n_clients,
                 clients_per_round=20,
                 aggregator=None,
                 X_val=None, y_val=None,
                 scaler=None,
                 lr=0.01,
                 async_mode=False,
                 max_staleness=5):

        self.n_features       = n_features
        self.n_clients        = n_clients
        self.clients_per_round= clients_per_round
        self.lr               = lr
        self.async_mode       = async_mode
        self.max_staleness    = max_staleness

        # Global model
        self.global_model  = FLNeuralNet(n_features)
        self.global_params = self.global_model.get_params()
        self.round         = 0

        # Validation set (D_val — server-held for Kappa evaluation)
        self.X_val  = X_val
        self.y_val  = y_val
        self.scaler = scaler

        # Aggregator
        self.aggregator = aggregator or MultiLayerAggregator(
            n_clients, tau=2.0, alpha=0.8, delta=0.2)

        # Async buffer: pending updates not yet aggregated
        self._async_buffer = deque()

        # Round history for logging
        self.history = []

    # ── Client selection ──────────────────────────────────────────────────────
    def select_clients(self, client_pool, rng=None):
        """
        Sample clients_per_round clients from pool.
        In async mode, also drain any pending buffer updates.
        """
        if rng is None:
            rng = np.random.RandomState(self.round)

        n = min(self.clients_per_round, len(client_pool))
        selected = rng.choice(len(client_pool), n, replace=False)
        return [client_pool[i] for i in selected]

    # ── Broadcast global model to clients ────────────────────────────────────
    def broadcast(self, clients):
        """
        Send current global model parameters to each client.
        In production: serialise and send over network.
        Here: direct parameter copy (simulates network delivery).
        """
        for client in clients:
            client.receive_global_model(
                self.global_params.copy(),
                round_number=self.round
            )

    # ── Receive and buffer client update ─────────────────────────────────────
    def receive_update(self, client_id, update, sent_at_round):
        """
        Accept an update from a client.
        In async mode, checks staleness before buffering.
        In sync mode, buffers all updates for the current round.
        """
        staleness = self.round - sent_at_round
        if self.async_mode and staleness > self.max_staleness:
            return False   # reject stale update
        self._async_buffer.append({
            'client_id':  client_id,
            'update':     update,
            'staleness':  staleness,
            'received_at':self.round,
        })
        return True

    # ── Aggregate all buffered updates ────────────────────────────────────────
    def aggregate(self):
        """
        Run three-layer defence + weighted aggregation on buffered updates.
        Clears the buffer after aggregation.
        Returns dict of per-round metrics.
        """
        if not self._async_buffer:
            return None

        pending   = list(self._async_buffer)
        self._async_buffer.clear()

        updates    = [p['update']    for p in pending]
        client_ids = [p['client_id'] for p in pending]
        stalenesses= [p['staleness'] for p in pending]

        t_start = time.perf_counter()

        # ── Three-layer defence ───────────────────────────────────────
        agg_update, outlier_flags, kappas, trust_scores = \
            self.aggregator.aggregate(
                updates,
                self.global_params,
                self.X_val,
                self.y_val,
                self.scaler,
                client_ids,
                lr=self.lr,
            )

        # ── Apply aggregated update to global model ───────────────────
        self.global_params = self.global_params + self.lr * agg_update
        self.global_model.set_params(self.global_params)

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        # ── Evaluate on validation set ────────────────────────────────
        pred_val = self.global_model.predict(self.X_val)
        acc  = accuracy_score(self.y_val, pred_val) * 100
        f1   = f1_score(self.y_val, pred_val, zero_division=0) * 100

        # ── Per-client semantic divergence logging ────────────────────
        per_client = []
        for i, (cid, upd, kap, trust, outlier) in enumerate(
                zip(client_ids, updates, kappas,
                    trust_scores, outlier_flags)):
            per_client.append({
                'client_id':    cid,
                'kappa':        round(float(kap), 4),
                'trust':        round(float(trust), 4),
                'outlier':      bool(outlier),
                'flagged':      bool(outlier or kap < self.aggregator.delta
                                    or trust < self.aggregator.trust_thresh),
                'staleness':    stalenesses[i],
            })

        metrics = {
            'round':        self.round,
            'n_updates':    len(updates),
            'n_flagged':    sum(1 for c in per_client if c['flagged']),
            'accuracy':     round(acc, 2),
            'f1':           round(f1, 2),
            'agg_time_ms':  round(elapsed_ms, 2),
            'mean_kappa':   round(float(np.mean(kappas)), 4),
            'mean_trust':   round(float(np.mean(trust_scores)), 4),
            'per_client':   per_client,
        }

        self.history.append(metrics)
        self.round += 1

        return metrics

    # ── One complete synchronous FL round ────────────────────────────────────
    def run_round(self, client_pool, rng=None):
        """
        Convenience method for synchronous FL:
          1. Select clients
          2. Broadcast global model
          3. Each client trains locally and submits update
          4. Aggregate
          5. Return metrics
        """
        selected = self.select_clients(client_pool, rng)
        self.broadcast(selected)

        for client in selected:
            update = client.local_train_and_submit()
            if update is not None:
                self.receive_update(
                    client.client_id,
                    update,
                    sent_at_round=self.round
                )

        return self.aggregate()

    # ── Async round: clients submit independently ─────────────────────────────
    def run_async_round(self, client_pool, rng=None,
                        participation_prob=0.3):
        """
        Asynchronous FL round:
          - Each client independently decides to participate
            with probability participation_prob (simulates churn)
          - Server aggregates whatever updates arrived
        """
        if rng is None:
            rng = np.random.RandomState(self.round)

        # Broadcast to all clients (they all have current model)
        self.broadcast(client_pool)

        # Each client submits independently with some probability
        participants = [c for c in client_pool
                        if rng.random() < participation_prob]

        if not participants:
            self.round += 1
            return None

        for client in participants:
            update = client.local_train_and_submit()
            if update is not None:
                self.receive_update(
                    client.client_id,
                    update,
                    sent_at_round=self.round
                )

        return self.aggregate()

    @property
    def global_accuracy(self):
        if not self.history:
            return 0.0
        return self.history[-1]['accuracy']

    @property
    def convergence_round(self, threshold=80.0):
        """First round accuracy exceeded threshold."""
        for m in self.history:
            if m['accuracy'] >= threshold:
                return m['round']
        return None
