"""
fl_client.py
============
Federated Learning Client.

Each client:
  - Holds its own local data partition (private, never shared)
  - Receives the global model from the server
  - Trains locally for several epochs
  - Computes the gradient update (local_model - global_model)
  - Returns only the update (never raw data) to the server

Byzantine clients additionally apply one of four attack strategies
before returning the update.

In simulation: communication is via direct Python calls.
In production: receive_global_model and local_train_and_submit
               would be gRPC endpoints.
"""

import numpy as np
from fl_model import FLNeuralNet
from fl_base  import (attack_label_flip, attack_gradient_poison,
                       attack_gps_spoof, is_malicious_round)


class FLClient:
    """
    Federated learning client — one per driver/rider in SC system.

    Parameters
    ----------
    client_id    : int    Unique identifier
    X_local      : ndarray  Local private data features
    y_local      : ndarray  Local private data labels
    n_features   : int    Feature dimension (must match server model)
    is_byzantine : bool   Whether this client is an adversary
    attack_type  : str    'lfp' | 'grad-p' | 'spf' | 'ooa' | None
    lr           : float  Local learning rate
    local_epochs : int    Epochs of local training per round
    batch_size   : int    Local mini-batch size
    """

    def __init__(self, client_id, X_local, y_local,
                 n_features,
                 is_byzantine=False,
                 attack_type=None,
                 lr=0.01,
                 local_epochs=5,
                 batch_size=32):

        self.client_id    = client_id
        self.X_local      = X_local
        self.y_local      = y_local
        self.n_features   = n_features
        self.is_byzantine = is_byzantine
        self.attack_type  = attack_type
        self.lr           = lr
        self.local_epochs = local_epochs
        self.batch_size   = batch_size

        # Local model — initialised with zeros, overwritten on broadcast
        self.local_model  = FLNeuralNet(n_features)

        # Current global model params received from server
        self._global_params = None
        self._round_number  = 0

        # Training history (for client-side logging)
        self.update_history = []

    # ── Receive broadcast from server ─────────────────────────────────────────
    def receive_global_model(self, global_params, round_number=0):
        """
        Server calls this to deliver the current global model.
        Client copies it to both local model and reference params.

        In production: deserialise from network payload.
        """
        self._global_params = global_params.copy()
        self._round_number  = round_number
        self.local_model.set_params(global_params.copy())

    # ── Local training ────────────────────────────────────────────────────────
    def local_train_and_submit(self):
        """
        Core FL client operation:
          1. (Byzantine) Apply attack to local data or later to gradient
          2. Train local model for local_epochs
          3. Compute update = local_params - global_params
          4. (Byzantine grad-p) Poison the gradient update
          5. Return update to server (never returns raw data)

        Returns
        -------
        update : np.ndarray  Gradient update vector, or None if no data
        """
        if self._global_params is None:
            return None
        if len(self.X_local) == 0:
            return None

        X_train = self.X_local.copy()
        y_train = self.y_local.copy()

        # ── Byzantine data-level attacks ──────────────────────────────
        if self.is_byzantine and self.attack_type:

            if self.attack_type == 'lfp':
                # Label flipping: invert fraud/legit labels
                y_train = attack_label_flip(y_train, flip_rate=0.30)

            elif self.attack_type == 'spf':
                # GPS spoofing: corrupt spatial features
                X_train = attack_gps_spoof(X_train, spoof_rate=0.30)

            elif self.attack_type == 'ooa':
                # On-off: only attack on odd periods
                if not is_malicious_round(self._round_number):
                    pass   # honest round — no modification

        # ── Local training (never leaves this device) ──────────────────
        self.local_model.set_params(self._global_params.copy())
        self.local_model.train(
            X_train, y_train,
            lr=self.lr,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
        )

        # ── Compute gradient update ────────────────────────────────────
        local_params  = self.local_model.get_params()
        update        = local_params - self._global_params

        # ── Byzantine gradient-level attack ───────────────────────────
        if self.is_byzantine and self.attack_type == 'grad-p':
            # Gradient poisoning: add large adversarial noise
            noise = np.random.randn(*update.shape) * 3.0 * np.linalg.norm(update)
            update = update + noise

        # ── Log update norm (for monitoring) ──────────────────────────
        self.update_history.append({
            'round':       self._round_number,
            'update_norm': float(np.linalg.norm(update)),
            'is_attack':   (self.is_byzantine and
                            self.attack_type is not None and
                            (self.attack_type != 'ooa' or
                             is_malicious_round(self._round_number))),
        })

        return update

    # ── Churn simulation (client goes offline) ────────────────────────────────
    def is_available(self, availability_prob=0.9, rng=None):
        """
        Simulate client dropout (churn).
        Returns False if client is offline this round.
        """
        if rng is None:
            rng = np.random.RandomState(self._round_number + self.client_id)
        return rng.random() < availability_prob

    def __repr__(self):
        role = 'Byzantine' if self.is_byzantine else 'Benign'
        atk  = f' [{self.attack_type}]' if self.attack_type else ''
        return (f"FLClient(id={self.client_id}, {role}{atk}, "
                f"n={len(self.X_local)})")
