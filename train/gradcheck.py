"""Numerical gradient check: is the hand-derived backprop actually correct?

Compares each analytic dLoss/dW against the finite-difference estimate
(Loss(w+h) - Loss(w-h)) / 2h. If the chain rule was applied correctly these
agree to ~1e-7. This is the sanity check that catches a dropped transpose or a
missing f'(z).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network import MLP, cross_entropy

rng = np.random.default_rng(1)

for act in ("relu", "sigmoid", "tanh"):
    net = MLP([12, 7, 5, 4], activation=act, seed=3)
    X = rng.normal(size=(6, 12))
    Y = np.zeros((6, 4)); Y[np.arange(6), rng.integers(0, 4, 6)] = 1.0
    L2 = 1e-3

    def loss():
        probs, _ = net.forward(X)
        reg = sum(0.5 * L2 * np.sum(w * w) for w in net.W)
        return cross_entropy(probs, Y) + reg

    _, cache = net.forward(X)
    dW, db = net.backward(cache, Y, l2=L2)

    worst = 0.0
    h = 1e-5
    for l in range(len(net.W)):
        for _ in range(60):  # spot-check 60 random weights per layer
            i = rng.integers(net.W[l].shape[0]); j = rng.integers(net.W[l].shape[1])
            orig = net.W[l][i, j]
            net.W[l][i, j] = orig + h; lp = loss()
            net.W[l][i, j] = orig - h; lm = loss()
            net.W[l][i, j] = orig
            num = (lp - lm) / (2 * h)
            ana = dW[l][i, j]
            denom = max(1e-9, abs(num) + abs(ana))
            worst = max(worst, abs(num - ana) / denom)
        for _ in range(20):  # and the biases
            k = rng.integers(net.b[l].shape[0])
            orig = net.b[l][k]
            net.b[l][k] = orig + h; lp = loss()
            net.b[l][k] = orig - h; lm = loss()
            net.b[l][k] = orig
            num = (lp - lm) / (2 * h)
            ana = db[l][k]
            denom = max(1e-9, abs(num) + abs(ana))
            worst = max(worst, abs(num - ana) / denom)

    status = "PASS" if worst < 1e-5 else "FAIL"
    print(f"{act:8s} worst relative error {worst:.2e}   {status}")
