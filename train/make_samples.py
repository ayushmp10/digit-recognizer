"""Pick one clean test digit per class and export them for the web app."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_mnist

Xtr, ytr, Xte, yte = load_mnist(sys.argv[1])
rng = np.random.default_rng(7)
out = []
for d in range(10):
    idx = np.where(yte == d)[0]
    # Prefer a middling-ink example: not the faintest, not the heaviest.
    ink = Xte[idx].sum(axis=1)
    pick = idx[np.argsort(ink)[len(idx) // 2]]
    out.append({"label": int(d),
                "px": [int(round(v * 255)) for v in Xte[pick]]})
path = os.path.join(sys.argv[2], "samples.json")
with open(path, "w") as fh:
    json.dump(out, fh, separators=(",", ":"))
print(f"wrote {path} ({os.path.getsize(path)/1024:.0f} KB)")
