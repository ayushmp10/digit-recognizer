"""Export real test digits this network gets WRONG.

The walkthrough's last step explains a mistake, so it should explain an actual
one: these are drawn from the 10,000 MNIST test images, classified with the
exact weights the page loads.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_mnist

w = json.load(open(sys.argv[2] + '/weights.json'))
sizes = w['sizes']
W = [np.array(a, dtype=np.float64).reshape(i, o) for a, i, o in zip(w['W'], sizes[:-1], sizes[1:])]
b = [np.array(a, dtype=np.float64) for a in w['b']]

def forward(X):
    a = X
    for l in range(len(W) - 1):
        a = np.maximum(0.0, a @ W[l] + b[l])
    z = a @ W[-1] + b[-1]
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return z, e / e.sum(axis=1, keepdims=True)

_, _, Xte, yte = load_mnist(sys.argv[1])
logits, probs = forward(Xte)
pred = probs.argmax(1)
wrong = np.flatnonzero(pred != yte)
conf = probs[wrong, pred[wrong]]
print(f"{len(wrong)} wrong out of {len(yte)}  ({100*(1-len(wrong)/len(yte)):.2f}% correct)")

# Pick a spread: a couple the net was very sure about (the interesting
# failures), a couple where it was torn, and keep the true labels distinct.
order = np.argsort(-conf)
picked, used_pairs = [], set()
for group in (order[:40], order[len(order)//2:len(order)//2+40], order[-40:]):
    for k in group:
        i = wrong[k]
        pair = (int(yte[i]), int(pred[i]))
        if pair in used_pairs:
            continue
        used_pairs.add(pair)
        picked.append(int(i))
        if len(picked) % 2 == 0:
            break
picked = picked[:6]

out = []
for i in picked:
    out.append({
        "truth": int(yte[i]),
        "pred": int(pred[i]),
        "conf": round(float(probs[i, pred[i]]), 4),
        "px": [int(round(v * 255)) for v in Xte[i]],
    })
    print(f"  index {i}: a {yte[i]} read as {pred[i]} at {100*probs[i,pred[i]]:.1f}% "
          f"(it gave the right answer {100*probs[i,yte[i]]:.2f}%)")

path = os.path.join(sys.argv[2], 'misses.json')
with open(path, 'w') as fh:
    json.dump(out, fh, separators=(',', ':'))
print(f"wrote {path} ({os.path.getsize(path)/1024:.0f} KB)")
