"""Approach-from-the-left test: can the hand come in from the free side and leave the bin-row bar alone?

approach_direction_test.py showed that coming in from the right or straight on drags or topples the bar on the bin row
(dense_chain_test: 14 of 20 neighbours were knocked down). The free side is the LEFT (+y, where no bars stand).
This tries pre-grasp offsets (back along the fingers, sideways in the hand frame; height stays 5 cm):
  L1 (-6, +10) cm  mirror of today's approach    L2 (-8, +4) cm  mostly straight, slightly from the left
  L3 (-10, +10) cm further back and left          L4 (-2, +10) cm  from the left, close in
  L5 (-6, +6) cm   between L1 and L2
on the same cases as before (pick at t1/t2/t3 with a bar at bin1 or bin2, and with no other bar), 3 scenes each = 120 trials.
OK = bar lifted and upright and the bin-row bar moved less than 2 cm. Saves approach_left_HHMMSS.csv.

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\approach_left_test.py
Needs scripts\\world.py, place_validation.py, stack_unstack_validation.py. About 7 minutes. CPU only.
NOTE: written without being able to run MuJoCo in the authoring environment.
"""
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import world as W

PRES = {"L1": (-0.06, 0.10), "L2": (-0.08, 0.04), "L3": (-0.10, 0.10), "L4": (-0.02, 0.10), "L5": (-0.06, 0.06)}
CASES = [("t1", "bin1"), ("t2", "bin1"), ("t2", "bin2"), ("t3", "bin2"), ("t3", "bin1"), ("t1", None), ("t2", None), ("t3", None)]
REPS = 3
BASE_PRE = np.array([-0.06, -0.10, 0.05])


def main():
    t0 = time.time()
    w = W.World()
    pv = W.pv
    rows, res, n = [], {}, 0
    total = len(PRES) * len(CASES) * REPS
    for h, (px, py) in PRES.items():
        pv.PRE = np.array([px, py, 0.05])
        for (p, q) in CASES:
            oks = dist = 0
            for rep in range(REPS):
                placement = {"red": ("slot", p)}
                if q:
                    placement["green"] = ("slot", q)
                w.reset(placement, np.random.default_rng(4000 + rep))
                g0 = w.xy("green") if q else None
                ctx = w.pick("red")
                held = ctx is not None and w.tilt("red") <= 20.0
                moved = float(np.linalg.norm(w.xy("green") - g0)) if q else 0.0
                d = bool(q) and (moved > 0.02 or w.tilt("green") > 8.0)
                ok = held and not d
                oks += ok
                dist += d
                n += 1
                print(f"[{n:2d}/{total}] approach {h} {PRES[h]}, pick at {p}, bar at {q or 'none':4s}: "
                      f"{'ok' if ok else ('FAIL not lifted' if not held else 'bar held')}"
                      f"{', bin-row bar moved ' + format(moved * 100, '.1f') + ' cm' if moved > 0.01 else ''}"
                      f"   ({time.time() - t0:.0f}s)", flush=True)
                rows.append([h, p, q or "none", rep, int(held), int(d), round(moved, 4)])
            res[(h, p, q)] = (oks, dist)
    pv.PRE = BASE_PRE.copy()
    print(f"\nPick OK (held and neighbour undisturbed) out of {REPS}; rows = approach, columns = case")
    print("approach  " + "".join(f"{p}+{q or 'none'}".rjust(11) for (p, q) in CASES))
    for h in PRES:
        print(f"{h:>8s}  " + "".join(f"{res[(h, p, q)][0]}/{REPS}".rjust(11) for (p, q) in CASES))
    out = W.ROOT / f"approach_left_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["approach", "pick_slot", "other_at", "rep", "held", "neighbour_disturbed", "neighbour_moved_m"])
        wr.writerows(rows)
    print(f"\nSaved {out.name} in {W.ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
