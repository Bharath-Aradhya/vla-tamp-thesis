"""Inspect one recorded demonstration: shapes, value ranges, segments, and a picture strip.

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\inspect_demo.py demos\\ep_p01_7000_ok.npz
Prints what is inside and saves inspect_<name>.png: 8 frames spread over the episode, top view above, side view below.
"""
import json
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1])
d = np.load(path, allow_pickle=False)
print("meta    :", json.loads(str(d["meta"])))
segs = json.loads(str(d["segments"]))
print("segments:", segs)
for k in ("top", "side", "state", "action", "bars", "seg"):
    if k in d:
        a = d[k]
        print(f"{k:7s} : shape {a.shape}, dtype {a.dtype}" + (f", min {a.min():.2f}, max {a.max():.2f}" if a.dtype != np.uint8 else ""))
seg = d["seg"]
for i, s in enumerate(segs):
    print(f"  segment {i} '{s}': frames {int((seg == i).sum())}")
act, st = d["action"], d["state"]
print("action - state mean abs difference per dimension (how far ahead the target leads):")
print("  ", np.round(np.abs(act - st).mean(axis=0), 3))
import imageio.v2 as imageio
T = len(seg)
idx = np.linspace(0, T - 1, 8).astype(int)
strip = np.concatenate([np.concatenate([d[v][i] for i in idx], axis=1) for v in ("top", "side") if v in d], axis=0)
out = Path(f"inspect_{path.stem}.png")
imageio.imwrite(out, strip)
print("saved", out)
