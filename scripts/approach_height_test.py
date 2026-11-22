"""Approach height test: can a higher approach stop the hand from hitting bars on the bin row when picking at the table row?

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\approach_height_test.py

The pick matrix showed two kinds of interference. (1) Sliding: closing the hand pulls the bar 8 cm toward the robot,
so the bar in that slot gets pushed (t2 -> t1, t3 -> t2, bin2 -> bin1). That is a fixed feature of the grasp and
becomes a planner precondition. (2) Approach: the hand comes in from 6 cm closer and 10 cm to the right, 5 cm above
the grasp point, and hits whatever stands on the bin row there (picking at t1 with a bar at bin1: failed 3/3, bin1
bar moved 17-18 cm; t2 with bin1 or bin2: failed or disturbed; t3 with bin2: failed 3/3). A higher approach should
clear those. This tries the approach height (the pre-grasp point above the grasp point) 5, 10, 15 and 20 cm
on these cases: pick at t1 with bin1 occupied; t2 with bin1; t2 with bin2; t3 with bin2; t3 with bin1; and
t1, t2, t3 with no other bar (to check that the grasp itself still works). 3 scenes each, 96 trials.
OK = bar lifted and upright, and the bar on the bin row moved less than 2 cm. Saves approach_height_HHMMSS.csv.
Needs scripts\\world.py, place_validation.py, stack_unstack_validation.py. Expect about 5 minutes. CPU only.
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

HEIGHTS = [0.05, 0.10, 0.15, 0.20]
CASES = [("t1", "bin1"), ("t2", "bin1"), ("t2", "bin2"), ("t3", "bin2"), ("t3", "bin1"), ("t1", None), ("t2", None), ("t3", None)]
REPS = 3
BASE_PRE = np.array([-0.06, -0.10, 0.05])


def main():
    t0 = time.time()
    w = W.World()
    pv = W.pv
    rows, res, n = [], {}, 0
    total = len(HEIGHTS) * len(CASES) * REPS
    for h in HEIGHTS:
        pv.PRE = np.array([BASE_PRE[0], BASE_PRE[1], h])
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
                print(f"[{n:2d}/{total}] approach {h * 100:.0f} cm, pick at {p}, bar at {q or 'none':4s}: "
                      f"{'ok' if ok else ('FAIL not lifted' if not held else 'bar held')}"
                      f"{', bin-row bar moved ' + format(moved * 100, '.1f') + ' cm' if moved > 0.01 else ''}"
                      f"   ({time.time() - t0:.0f}s)", flush=True)
                rows.append([h, p, q or "none", rep, int(held), int(d), round(moved, 4)])
            res[(h, p, q)] = (oks, dist)
    pv.PRE = BASE_PRE.copy()
    print(f"\nPick OK (held and neighbour undisturbed) out of {REPS}; rows = approach height, columns = case")
    print("height   " + "".join(f"{p}+{q or 'none'}".rjust(11) for (p, q) in CASES))
    for h in HEIGHTS:
        print(f"{h * 100:4.0f} cm  " + "".join(f"{res[(h, p, q)][0]}/{REPS}".rjust(11) for (p, q) in CASES))
    out = W.ROOT / f"approach_height_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["height_m", "pick_slot", "other_at", "rep", "held", "neighbour_disturbed", "neighbour_moved_m"])
        wr.writerows(rows)
    print(f"\nSaved {out.name} in {W.ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
