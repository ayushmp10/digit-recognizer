"""Load MNIST from the raw IDX files, plus the preprocessing the browser mirrors."""

import gzip
import struct
import numpy as np


def _read_idx(path):
    """Parse an IDX file. Pure byte parsing -- these files are data, not code."""
    with gzip.open(path, "rb") as fh:
        magic, = struct.unpack(">I", fh.read(4))
        dtype_code = (magic >> 8) & 0xFF
        n_dims = magic & 0xFF
        if dtype_code != 0x08:
            raise ValueError(f"expected unsigned byte data, got code {dtype_code:#x}")
        dims = struct.unpack(f">{n_dims}I", fh.read(4 * n_dims))
        buf = fh.read(int(np.prod(dims)))
        return np.frombuffer(buf, dtype=np.uint8).reshape(dims)


def load_mnist(data_dir):
    """Return (X_train, y_train, X_test, y_test) with X flattened to (n, 784) in [0,1]."""
    import os
    j = lambda name: os.path.join(data_dir, name)
    Xtr = _read_idx(j("train-images-idx3-ubyte.gz"))
    ytr = _read_idx(j("train-labels-idx1-ubyte.gz"))
    Xte = _read_idx(j("t10k-images-idx3-ubyte.gz"))
    yte = _read_idx(j("t10k-labels-idx1-ubyte.gz"))

    # Layer 0 of the network is one unit per pixel, so the 28x28 image becomes
    # a flat 784-vector. Scaling to [0,1] keeps the first linear scores small.
    Xtr = Xtr.reshape(len(Xtr), -1).astype(np.float32) / 255.0
    Xte = Xte.reshape(len(Xte), -1).astype(np.float32) / 255.0
    return Xtr, ytr.astype(np.int64), Xte, yte.astype(np.int64)


def one_hot(y, n_classes=10):
    """Truth vector y for cross-entropy: 1 at the correct digit, 0 elsewhere."""
    out = np.zeros((len(y), n_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def random_shift(X, rng, max_px=2, prob=0.5):
    """Jitter a fraction of the images by up to +/-2 px.

    MNIST digits sit dead center; a digit drawn with a mouse will not. Training
    on shifted copies makes the net tolerate the difference instead of relying
    on a pixel landing in exactly the right cell.

    Only `prob` of the batch gets jittered, so the net still sees plenty of
    perfectly centered digits -- augmenting everything trades away accuracy on
    centered input for robustness we don't need that much of.
    """
    imgs = X.reshape(-1, 28, 28)
    out = imgs.copy()
    hit = rng.random(len(imgs)) < prob
    dy, dx = rng.integers(-max_px, max_px + 1, size=(2, len(imgs)))
    for i in np.flatnonzero(hit):
        out[i] = np.roll(np.roll(imgs[i], dy[i], axis=0), dx[i], axis=1)
    return out.reshape(len(imgs), -1)
