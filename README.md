# Digit Scope

A draw-a-digit recognizer built from the CIS 5240 Week 12 notes, with a nine-step
guided walkthrough of the forward pass. No ML framework: the network, backprop
and the training loop are written out in numpy.

The walkthrough narrates one operation per step -- a single wire, then one
neuron's weighted sum, then the activation, then the whole layer, then softmax
-- and fills in the arithmetic from whatever digit is currently drawn, so the
numbers on screen are that drawing's own.

The last step explains a mistake. `train/make_misses.py` exports six digits the
network actually gets wrong out of the 10,000 test images, and loading one shows
the cross-entropy loss on that example and the `(p - y)` gradient it would send
backwards. Two of the six are failures the network is 100% confident about,
which is the point: a softmax probability says how strongly these weights vote,
not how often they are right.

**Live page:** https://ayushmp10.github.io/digit-recognizer/

    784 -> 32 (relu) -> 16 (relu) -> 10 (softmax)
    25,818 parameters    97.81% accuracy on the 10,000 MNIST test digits

## Where each idea from the notes lives

| Lecture concept | Code |
| --- | --- |
| Layers, "deep" = stacked hidden layers | `train/network.py` &rarr; `MLP.__init__` |
| Activation functions + their derivatives | `train/network.py` &rarr; `sigmoid`, `tanh`, `relu` |
| Why a step function can't learn | `train/network.py` &rarr; `backward`, the `dz * df(z)` line |
| Forward pass / computation graph | `train/network.py` &rarr; `MLP.forward` |
| Chain rule, error backpropagation | `train/network.py` &rarr; `MLP.backward` |
| Softmax over raw scores | `train/network.py` &rarr; `softmax` |
| Cross-entropy loss | `train/network.py` &rarr; `cross_entropy` |
| `(y_hat - y) * x_i` gradient | `train/network.py` &rarr; first line of `backward` |
| Weight update `w - eta * grad` | `train/network.py` &rarr; `MLP.sgd_step` |
| Learning rate, decaying eta | `train/train.py` &rarr; `LR0`, `LR_DECAY` |
| Mini-batch training | `train/train.py` &rarr; `BATCH` |
| L2 regularization | `train/network.py` &rarr; `backward(l2=...)` |
| Early stopping on a dev set | `train/train.py` &rarr; the `best` dict |
| Cross-entropy and `(p - y)` on a real error | the page's last walkthrough step |

The site is the `docs/` folder, served by GitHub Pages: `index.html` plus the exported
weights and sample digits. It is static, so the network runs entirely in your browser.

## Design

The page was redesigned with [Hallmark](https://github.com/nutlope/hallmark) at full depth,
in app-screen scope, on the Console theme (dealt 1 of 3). The content and every handler were
kept; the system and the composition changed. The CSS carries the stamp, and the build is
logged in `.hallmark/log.json`.

    node ~/.claude/skills/hallmark/scripts/sloplint.mjs docs/index.html \
      --genre modern-minimal --scope app --kept --url http://localhost:8731/

The scaffold's `tokens.css` and `base.css` are not kept: an Artifact is a single HTML file,
so the page inlines them and nothing imported them. Regenerate them with:

    node ~/.claude/skills/hallmark/scripts/theme.mjs scaffold console --redesign full --scope app

## Running it

    # 1. fetch MNIST into its own directory
    mkdir -p data/mnist && cd data/mnist
    for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
             t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
      curl -O "https://ossci-datasets.s3.amazonaws.com/mnist/$f"
    done

    # 2. check the hand-derived gradients against finite differences
    cd ../../train && python3 gradcheck.py

    # 3. train (~18s) and export weights.json for the page
    python3 train.py ../data/mnist ../docs

    # 4. export the sample digits and the real misclassifications
    python3 make_samples.py ../data/mnist ../docs
    python3 make_misses.py ../data/mnist ../docs

    # 5. serve the page
    cd ../docs && python3 -m http.server 8731

## Notes on the two things that are easy to get wrong

**Input normalisation.** MNIST digits are cropped, scaled so the longest side
is 20px, and centred on their centre of mass in a 28x28 field. A digit drawn
with a mouse and naively downsampled lands outside the distribution the net
trained on, and accuracy collapses. `preprocess()` in `docs/index.html` repeats
that normalisation exactly; the "what layer 0 sees" panel shows the result.

**Spurious numpy warnings.** numpy 2.0 on Apple's Accelerate BLAS leaks bogus
floating-point exception flags out of `matmul` — `np.ones((64,784)) @ 0.01`
raises "overflow encountered in matmul" with every value finite. `train.py`
mutes that one message and asserts finiteness itself instead.
