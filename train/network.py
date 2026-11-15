"""
A feed-forward neural network (Multi-Layer Perceptron) written from scratch.

Every piece of this file maps to CIS 5240 Week 12:
  - Neural Networks (Part 1)        -> layers, activation functions, "deep", fully connected
  - Neural Networks (Part 2)        -> forward pass, chain rule, error backpropagation
  - Stochastic Gradient Descent 1/2 -> loss, learning rate, update rule, mini-batches,
                                       L2 regularization, softmax for multi-class

No autograd, no framework. The chain rule is spelled out by hand so you can
point at the exact line that implements each idea from lecture.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. ACTIVATION FUNCTIONS            (Part 1: "Activation Functions")
# ---------------------------------------------------------------------------
# A neuron does NOT output its raw linear score z = w . x + b. It applies a
# non-linear function f to z, and the result is the activation a = f(z).
#
# Every activation here is paired with its derivative, because Part 2 is blunt
# about the requirement: backprop must compute a partial derivative at every
# node. A perceptron's step function has no derivative at the jump, so the
# chain rule breaks and the network cannot learn. Smooth f -> learnable net.

def sigmoid(z):
    """Squashes any real number into (0, 1):  y = 1 / (1 + e^-z)."""
    # Split on the sign of z so we never exponentiate a large positive number
    # (which would overflow). Mathematically identical to 1/(1+e^-z).
    out = np.empty_like(z)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def d_sigmoid(z):
    """d/dz sigmoid(z) = s(z) * (1 - s(z)). Smooth everywhere -> differentiable."""
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z):
    """Like sigmoid, but centered at 0 with range (-1, 1)."""
    return np.tanh(z)


def d_tanh(z):
    """d/dz tanh(z) = 1 - tanh^2(z)."""
    return 1.0 - np.tanh(z) ** 2


def relu(z):
    """Rectified Linear Unit: max(0, z).

    Near-linear for positive z, so it avoids the heavy squashing that sigmoid
    and tanh inflict on large |z|. This is what we actually train with.
    """
    return np.maximum(0.0, z)


def d_relu(z):
    """1 where z > 0, else 0. (Kinked at exactly 0; we take the 0 side.)"""
    return (z > 0).astype(z.dtype)


ACTIVATIONS = {
    "sigmoid": (sigmoid, d_sigmoid),
    "tanh": (tanh, d_tanh),
    "relu": (relu, d_relu),
}


# ---------------------------------------------------------------------------
# 2. OUTPUT LAYER + LOSS       (SGD Part 2: "Multi-Class & Softmax")
# ---------------------------------------------------------------------------
# The last layer produces 10 raw scores (logits) -- one per digit. Raw scores
# are not probabilities: they can be negative and they don't sum to 1. Softmax
# exponentiates them (everything becomes positive) and divides by the sum, so
# the result is a genuine probability distribution over the 10 digits.

def softmax(z):
    """Row-wise softmax over a batch of logits, shape (batch, n_classes)."""
    # Subtracting the row max before exponentiating is the standard numerical
    # -stability trick: softmax(z) == softmax(z - c) for any constant c, and
    # this keeps e^(...) from overflowing.
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y_onehot, eps=1e-12):
    """Average cross-entropy loss = mean negative log probability of the truth.

    Only the true class contributes, so this is just -log(p[correct digit]),
    averaged over the batch.
    """
    return float(-np.sum(y_onehot * np.log(probs + eps)) / probs.shape[0])


# ---------------------------------------------------------------------------
# 3. THE NETWORK                    (Part 1: "Feed-Forward Neural Networks")
# ---------------------------------------------------------------------------

class MLP:
    """A fully connected feed-forward net.

    `sizes` lists the width of every layer, input first. For this project:
        [784, 32, 16, 10]
          784 = Layer 0, the input layer: one unit per pixel of the 28x28 image
           32 = Layer 1, hidden      (hidden only because we never observe it)
           16 = Layer 2, hidden
           10 = output layer: one unit per digit 0-9

    Two hidden layers, so this net is only barely "deep" -- and that is the
    whole content of the word, per lecture: deep == literally more stacked
    hidden layers, nothing more profound than that.

    "Fully connected" means every unit in a layer is wired to every unit in
    the layer before it. That's exactly what a dense weight matrix is: W[l]
    has shape (units in layer l-1, units in layer l), one weight per wire.
    """

    def __init__(self, sizes, activation="relu", seed=0):
        self.sizes = list(sizes)
        self.activation_name = activation
        self.f, self.df = ACTIVATIONS[activation]

        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            # He initialization: scale ~ sqrt(2 / n_in), the right variance for
            # ReLU. Starting from a *random* point is fine and expected -- but
            # note SGD Part 1's warning: a neural net's loss surface is
            # non-convex, so unlike logistic regression there is no guarantee
            # we land in the global minimum. Where we start matters.
            self.W.append(rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    # -- forward pass -------------------------------------------------------
    # (Part 2, sections 2-3: the computation graph, evaluated left to right.)
    def forward(self, X):
        """Run inputs through the net and keep every intermediate value.

        X: (batch, 784). Returns (probs, cache).

        We save the z's and a's because the backward pass needs them -- that's
        what makes a computation graph useful: each node remembers its own
        inputs and output, so each node can later compute its own local
        derivative.
        """
        a = X
        cache = {"a": [X], "z": []}

        # Hidden layers: linear score, then the non-linear activation.
        for l in range(len(self.W) - 1):
            z = a @ self.W[l] + self.b[l]   # z = w . x + b, for every unit at once
            a = self.f(z)                   # a = f(z)
            cache["z"].append(z)
            cache["a"].append(a)

        # Output layer: linear scores, then softmax -> probability distribution.
        z_out = a @ self.W[-1] + self.b[-1]
        probs = softmax(z_out)
        cache["z"].append(z_out)
        cache["a"].append(probs)
        cache["logits"] = z_out
        return probs, cache

    # -- backward pass ------------------------------------------------------
    # (Part 2, sections 4-5: backward differentiation / error backpropagation.)
    def backward(self, cache, y_onehot, l2=0.0):
        """Compute dLoss/dW and dLoss/db for every layer, right to left.

        The question lecture poses: how much did a weight buried in the first
        hidden layer contribute to the error at the very end? Answer: multiply
        the local derivatives along every path from that weight to the loss --
        the chain rule. We do it layer by layer so each product is reused
        instead of recomputed.
        """
        B = y_onehot.shape[0]
        n = len(self.W)
        dW = [None] * n
        db = [None] * n

        # ---- Step 1: the gradient at the output.
        # Softmax + cross-entropy together collapse to the elegant result from
        # SGD Part 2:   dLoss/dz = (prediction - truth)
        # Read it the way lecture does:
        #   predicted 1.0, truth 1  -> 0.0  -> no update, we were right
        #   predicted 0.9, truth 1  -> -0.1 -> nudge gently
        #   predicted 0.0, truth 1  -> -1.0 -> update hard
        # Dividing by B averages the gradient over the mini-batch.
        dz = (cache["a"][-1] - y_onehot) / B

        # ---- Step 2: walk backwards through the layers.
        for l in reversed(range(n)):
            a_prev = cache["a"][l]          # what fed INTO this layer

            # Gradient w.r.t. this layer's weights. This is the matrix form of
            # lecture's  (y_hat - y) * x_i : the incoming activation times the
            # error signal arriving from above.
            dW[l] = a_prev.T @ dz
            db[l] = dz.sum(axis=0)

            # L2 regularization (SGD Part 2, section 3): penalize large weights
            # so the net generalizes instead of memorizing noisy pixels. The
            # derivative of (l2/2) * ||W||^2 is just l2 * W. Biases are left
            # alone by convention -- they don't cause overfitting.
            if l2:
                dW[l] += l2 * self.W[l]

            if l > 0:
                # Push the error signal through the wires into the previous
                # layer: each unit below gets blamed in proportion to the
                # weight connecting it upward.
                da_prev = dz @ self.W[l].T
                # Then through that layer's activation function -- the local
                # derivative f'(z). THIS is the multiplication that dies if f
                # is a step function, and the reason we need smooth
                # activations at all.
                dz = da_prev * self.df(cache["z"][l - 1])

        return dW, db

    # -- the update rule ----------------------------------------------------
    # (SGD Part 1, section 5:  w_{t+1} = w_t - eta * dLoss/dw)
    def sgd_step(self, dW, db, lr):
        """Step every parameter downhill by learning rate * its own gradient.

        The skiing analogy from lecture, in two lines: the gradient points
        uphill, so we subtract it to go down, and `lr` (eta) is how big a step
        we take. Too big and we overshoot the minimum and bounce; too small
        and we crawl.
        """
        for l in range(len(self.W)):
            self.W[l] -= lr * dW[l]
            self.b[l] -= lr * db[l]

    # -- convenience --------------------------------------------------------
    def predict(self, X):
        probs, _ = self.forward(X)
        return probs.argmax(axis=1)

    def accuracy(self, X, y):
        return float((self.predict(X) == y).mean())
