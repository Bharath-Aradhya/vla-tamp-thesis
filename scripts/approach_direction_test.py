"""Approach direction test: which way should the hand come in so it does not sweep the bar on the bin row?

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\approach_direction_test.py

approach_debug.py showed the cause: the hand starts its final approach 6 cm closer and 10 cm to the RIGHT of the bar at
grasp height, i.e. right on top of the bar standing on the bin row, and then moves LEFT toward the target bar,
dragging the bin-row bar along (it moved 6-9 cm by the half-way point and ended 11-19 cm off). A higher approach
only helped partly and broke the grasp at 20 cm. A different approach direction should avoid it. This tries the
pre-grasp offset (back along the fingers, to the right across the palm; height stays 5 cm) of:
  A (-6, -10) cm  today's approach        B (-10, 0) cm  straight in from the robot side
  C (-8, -4) cm   mostly straight         D (-10, -4) cm  straight, slightly from the right
  E (-12, 0) cm   straight, further back
on the cases: pick at t1 / t2 / t3 with no other bar (does the grasp still work?), and pick at t1 with a bar at bin1;
t2 with bin1; t2 with bin2; t3 with bin2; t3 with bin1. 3 scenes each = 120 trials.
OK = bar lifted and upright, and the bin-row bar moved less than 2 cm. Saves approach_direction_HHMMSS.csv.
Needs scripts\\world.py, place_validation.py, stack_unstack_validation.py. Expect about 6 minutes. CPU only.
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

PRES = {"A": (-0.06, -0.10), "B": (-0.10, 0.0), "C": (-0.08, -0.04), "D": (-0.10, -0.04), "E": (-0.12, 0.0)}
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
    out = W.ROOT / f"approach_direction_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["approach", "pick_slot", "other_at", "rep", "held", "neighbour_disturbed", "neighbour_moved_m"])
        wr.writerows(rows)
    print(f"\nSaved {out.name} in {W.ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
