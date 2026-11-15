"""Train the MLP with mini-batch SGD and export the weights for the web app.

Usage: python3 train.py <mnist_data_dir> <output_dir>
"""

import json
import os
import sys
import time

import numpy as np

# numpy 2.0 on Apple's Accelerate BLAS leaks bogus floating-point exception
# flags out of matmul: a plain ones @ 0.01 raises "overflow encountered in
# matmul" while every value stays finite. The warnings are a platform artifact,
# not instability, so we mute that one message and assert finiteness ourselves
# below -- real divergence still stops the run.
import warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network import MLP, cross_entropy          # noqa: E402
from data import load_mnist, one_hot, random_shift  # noqa: E402

# ---- hyperparameters ------------------------------------------------------
SIZES = [784, 32, 16, 10]   # small hidden layers on purpose: 32 and 16 units
                            # are few enough to draw every neuron on screen
ACTIVATION = "relu"
EPOCHS = 60
BATCH = 64                  # mini-batch: the compromise between pure SGD
                            # (noisy, one example at a time) and full batch
                            # (accurate but slow)
LR0 = 0.4                   # initial learning rate (eta)
LR_DECAY = 0.95             # lecture's advice: start with big steps, then
                            # shrink eta to settle into the minimum
L2 = 1e-5
SEED = 0


def main(data_dir, out_dir):
    rng = np.random.default_rng(SEED)
    Xtr_all, ytr_all, Xte, yte = load_mnist(data_dir)

    # Held-out dev set, used only to watch for overfitting (early stopping).
    perm = rng.permutation(len(Xtr_all))
    dev_idx, tr_idx = perm[:5000], perm[5000:]
    Xdev, ydev = Xtr_all[dev_idx], ytr_all[dev_idx]
    Xtr, ytr = Xtr_all[tr_idx], ytr_all[tr_idx]
    Ytr = one_hot(ytr)

    net = MLP(SIZES, activation=ACTIVATION, seed=SEED)
    print(f"architecture {SIZES}  activation={ACTIVATION}")
    print(f"parameters   {sum(w.size for w in net.W) + sum(b.size for b in net.b):,}")
    print(f"train {len(Xtr):,} | dev {len(Xdev):,} | test {len(Xte):,}\n")

    best = {"acc": -1.0, "epoch": -1, "W": None, "b": None}
    lr = LR0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(Xtr))
        Xshift = random_shift(Xtr, rng)      # fresh jitter every epoch
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            xb, yb = Xshift[idx], Ytr[idx]

            probs, cache = net.forward(xb)           # forward pass
            if not np.isfinite(probs).all():         # genuine divergence check
                raise RuntimeError(f"non-finite output at epoch {epoch}; lr {lr} too high")
            epoch_loss += cross_entropy(probs, yb)   # how wrong were we
            dW, db = net.backward(cache, yb, l2=L2)  # backprop the blame
            net.sgd_step(dW, db, lr)                 # w <- w - eta * grad
            n_batches += 1

        dev_acc = net.accuracy(Xdev, ydev)
        print(f"epoch {epoch:2d}  lr {lr:.4f}  loss {epoch_loss/n_batches:.4f}  "
              f"dev acc {dev_acc*100:.2f}%")

        # Early stopping: remember the parameters from the best dev epoch, so a
        # later epoch that starts memorizing training noise can't cost us.
        if dev_acc > best["acc"]:
            best = {"acc": dev_acc, "epoch": epoch,
                    "W": [w.copy() for w in net.W], "b": [b.copy() for b in net.b]}
        lr *= LR_DECAY

    net.W, net.b = best["W"], best["b"]
    test_acc = net.accuracy(Xte, yte)
    print(f"\nbest dev epoch {best['epoch']} ({best['acc']*100:.2f}%)")
    print(f"TEST ACCURACY  {test_acc*100:.2f}%  on {len(yte):,} unseen digits")
    print(f"trained in {time.time()-t0:.1f}s")

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "weights.npz"),
             *net.W, *net.b)

    # Export for the browser. Round to 4 decimals to keep the file small --
    # the accuracy cost is nil and it saves roughly half the bytes.
    payload = {
        "sizes": SIZES,
        "activation": ACTIVATION,
        "test_accuracy": round(test_acc, 4),
        "best_epoch": best["epoch"],
        "W": [[round(float(v), 4) for v in w.ravel()] for w in net.W],
        "b": [[round(float(v), 4) for v in bb.ravel()] for bb in net.b],
    }
    path = os.path.join(out_dir, "weights.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"wrote {path}  ({os.path.getsize(path)/1024:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
