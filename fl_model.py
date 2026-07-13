"""
fl_model.py
===========
Neural network model for federated learning.

Implements the CNN architecture described in the paper:
  - 3 fully-connected layers (paper's CNN adapted for tabular SC data)
  - ReLU activations in hidden layers
  - Sigmoid output for binary fraud classification
  - Adam optimiser
  - Matches paper hyperparameters: lr=0.01, batch=32, epochs=5

Note on CNN vs MLP for tabular data:
  The paper describes a CNN with convolutional layers for image-like
  spatial data. For the 10-feature tabular SC data (lat/lon, duration,
  fare, etc.), a multi-layer perceptron (MLP) is architecturally
  equivalent — convolutional kernels on 1D feature vectors reduce to
  dense layers. The architecture here uses hidden sizes [64, 128, 64]
  matching the paper's filter counts, with the same Adam + ReLU setup.

FLNeuralNet exposes:
  train(X, y, lr, epochs, batch_size)  — local training
  predict(X)                           — binary predictions
  predict_proba(X)                     — fraud probability
  get_params()                         — flat parameter vector
  set_params(params)                   — load flat parameter vector

The flat parameter representation is what gets sent as the
gradient update: update = local_params - global_params
"""

import numpy as np


class FLNeuralNet:
    """
    3-layer MLP for binary SC fraud classification.
    Architecture: input(10) → 64 → 128 → 64 → 1
    Activations:  ReLU → ReLU → ReLU → Sigmoid
    """

    HIDDEN_SIZES = [64, 128, 64]

    def __init__(self, n_features=10, hidden_sizes=None, seed=None):
        self.n_features  = n_features
        self.hidden_sizes= hidden_sizes or self.HIDDEN_SIZES

        if seed is not None:
            np.random.seed(seed)

        # ── Initialise weights (Xavier uniform) ───────────────────────
        self.weights = []   # list of weight matrices
        self.biases  = []   # list of bias vectors

        layer_sizes = [n_features] + self.hidden_sizes + [1]

        for i in range(len(layer_sizes) - 1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            limit   = np.sqrt(6.0 / (fan_in + fan_out))
            W = np.random.uniform(-limit, limit, (fan_in, fan_out))
            b = np.zeros(fan_out)
            self.weights.append(W)
            self.biases.append(b)

        # Adam optimiser state
        self._adam_m_w = [np.zeros_like(W) for W in self.weights]
        self._adam_v_w = [np.zeros_like(W) for W in self.weights]
        self._adam_m_b = [np.zeros_like(b) for b in self.biases]
        self._adam_v_b = [np.zeros_like(b) for b in self.biases]
        self._adam_t   = 0

    # ── Forward pass ──────────────────────────────────────────────────────────
    def _relu(self, x):
        return np.maximum(0, x)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _forward(self, X):
        """
        Returns activations at each layer.
        activations[0] = input X
        activations[-1] = output (sigmoid)
        """
        activations = [X]
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            if i < len(self.weights) - 1:
                a = self._relu(z)
            else:
                a = self._sigmoid(z)   # output layer
            activations.append(a)
        return activations

    # ── Backward pass ─────────────────────────────────────────────────────────
    def _backward(self, activations, y):
        """
        Backpropagation. Returns (grad_W list, grad_b list).
        Loss: binary cross-entropy.
        """
        n = len(y)
        grad_W = [None] * len(self.weights)
        grad_b = [None] * len(self.biases)

        # Output layer gradient
        # dL/dz_out = (a_out - y) for binary cross-entropy + sigmoid
        delta = (activations[-1].flatten() - y) / n
        delta = delta.reshape(-1, 1)

        grad_W[-1] = activations[-2].T @ delta
        grad_b[-1] = delta.sum(axis=0)

        # Hidden layers (backprop through ReLU)
        for i in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[i + 1].T) * \
                    (activations[i + 1] > 0)   # ReLU derivative
            grad_W[i] = activations[i].T @ delta
            grad_b[i] = delta.sum(axis=0)

        return grad_W, grad_b

    # ── Adam update step ──────────────────────────────────────────────────────
    def _adam_update(self, grad_W, grad_b, lr,
                     beta1=0.9, beta2=0.999, eps=1e-8):
        self._adam_t += 1
        t = self._adam_t

        for i in range(len(self.weights)):
            # Weights
            self._adam_m_w[i] = beta1 * self._adam_m_w[i] + \
                                 (1 - beta1) * grad_W[i]
            self._adam_v_w[i] = beta2 * self._adam_v_w[i] + \
                                 (1 - beta2) * grad_W[i] ** 2
            m_hat = self._adam_m_w[i] / (1 - beta1 ** t)
            v_hat = self._adam_v_w[i] / (1 - beta2 ** t)
            self.weights[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

            # Biases
            self._adam_m_b[i] = beta1 * self._adam_m_b[i] + \
                                 (1 - beta1) * grad_b[i]
            self._adam_v_b[i] = beta2 * self._adam_v_b[i] + \
                                 (1 - beta2) * grad_b[i] ** 2
            m_hat = self._adam_m_b[i] / (1 - beta1 ** t)
            v_hat = self._adam_v_b[i] / (1 - beta2 ** t)
            self.biases[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ── Local training ────────────────────────────────────────────────────────
    def train(self, X, y, lr=0.01, epochs=5, batch_size=32):
        """
        Train locally for `epochs` passes over the data.
        Uses mini-batch Adam optimisation.
        """
        n = len(X)
        for _ in range(epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                batch_idx = idx[start:start + batch_size]
                Xb = X[batch_idx]
                yb = y[batch_idx]

                activations    = self._forward(Xb)
                grad_W, grad_b = self._backward(activations, yb)
                self._adam_update(grad_W, grad_b, lr)

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict_proba(self, X):
        activations = self._forward(X)
        return activations[-1].flatten()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    # ── Parameter serialisation (what gets sent as gradient update) ───────────
    def get_params(self):
        """Flatten all weights and biases to a single vector."""
        parts = []
        for W, b in zip(self.weights, self.biases):
            parts.append(W.flatten())
            parts.append(b.flatten())
        return np.concatenate(parts)

    def set_params(self, params):
        """Load a flat parameter vector back into weights and biases."""
        idx = 0
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            w_size = W.size
            b_size = b.size
            self.weights[i] = params[idx:idx + w_size].reshape(W.shape)
            idx += w_size
            self.biases[i]  = params[idx:idx + b_size]
            idx += b_size

    @property
    def n_params(self):
        return sum(W.size + b.size
                   for W, b in zip(self.weights, self.biases))

    def __repr__(self):
        sizes = [self.n_features] + self.hidden_sizes + [1]
        arch  = ' → '.join(str(s) for s in sizes)
        return f"FLNeuralNet({arch}, params={self.n_params:,})"


if __name__ == '__main__':
    net = FLNeuralNet(n_features=10)
    print(net)
    X = np.random.randn(100, 10)
    y = (np.random.rand(100) > 0.85).astype(float)
    print(f"Before training — accuracy: "
          f"{((net.predict(X) == y).mean()*100):.1f}%")
    net.train(X, y, lr=0.01, epochs=10)
    print(f"After training  — accuracy: "
          f"{((net.predict(X) == y).mean()*100):.1f}%")
    params = net.get_params()
    print(f"Parameter vector length: {len(params):,}")
    net2 = FLNeuralNet(10)
    net2.set_params(params)
    print(f"Params restored correctly: "
          f"{np.allclose(net2.get_params(), params)}")
