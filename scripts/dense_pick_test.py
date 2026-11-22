"""Dense pick test: does picking one bar disturb its neighbours? (3 bars standing at t1, t2, t3, 8 cm apart)

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\dense_pick_test.py

Why: closing the hand slides the picked bar about 8 cm toward the robot, and the bar is carried over its
neighbours. In every validation so far the bar stood alone. The PDDL tasks put 2-3 bars next to each other.
For 6 random scenes (red t1, blue t2, green t3, random rotations outside 38-54 deg) it picks, in turn, the bar at
t2, at t3 and at t1 and puts it into bin1 (18 trials). OK = the bar ends at bin1 (within 2 cm, standing) AND
both other bars stay within 2 cm of where they were and standing. Prints per pick position what happened and
saves dense_pick_HHMMSS.csv. Needs scripts\\world.py, place_validation.py, stack_unstack_validation.py.
Expect about 4 minutes. CPU only.
NOTE: written without being able to run MuJoCo in the authoring environment.
"""
import csv
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import world as W

REPS = 6
LAYOUT = {"red": ("slot", "t1"), "blue": ("slot", "t2"), "green": ("slot", "t3")}
AT = {"t1": "red", "t2": "blue", "t3": "green"}


def main():
    t0 = time.time()
    w = W.World()
    rows, summ = [], {s: Counter() for s in ("t2", "t3", "t1")}
    n = 0
    for rep in range(REPS):
        for slot in ("t2", "t3", "t1"):
            w.reset(LAYOUT, np.random.default_rng(1000 + rep))
            before = {i: w.xy(i) for i in W.ITEMS}
            item = AT[slot]
            ctx = w.pick(item)
            why = "pick failed (bar not lifted)" if ctx is None else w.place_table(item, "bin1", ctx)
            moved = {i: float(np.linalg.norm(w.xy(i) - before[i])) for i in W.ITEMS if i != item}
            tilts = {i: w.tilt(i) for i in W.ITEMS if i != item}
            err = float(np.linalg.norm(w.xy(item) - np.array(W.SLOTS["bin1"])))
            disturbed = [i for i in moved if moved[i] > 0.02 or tilts[i] > 8.0 or w.z(i) < 0.5]
            if why:
                stage = why
            elif w.z(item) < 0.5 or w.tilt(item) > 8.0 or err > 0.02:
                stage = "landing failed"
            else:
                stage = "ok"
            result = stage if not disturbed else (stage + " + disturbed " + "/".join(disturbed))
            summ[slot][("ok" if stage == "ok" and not disturbed else
                        "bar ok but neighbour disturbed" if stage == "ok" else stage)] += 1
            n += 1
            print(f"[{n:2d}/{REPS * 3}] pick {item:5s} from {slot}: {result}"
                  + "".join(f"  ({i} moved {moved[i] * 100:.1f} cm)" for i in moved if moved[i] > 0.005)
                  + f"   ({time.time() - t0:.0f}s)", flush=True)
            rows.append([rep, slot, item, stage, round(err, 4)] + [round(moved.get(i, 0.0), 4) for i in W.ITEMS]
                        + ["/".join(disturbed)])
    print("\nPick position -> outcomes (6 scenes each)")
    for slot in ("t1", "t2", "t3"):
        print(f"  {slot}: " + ", ".join(f"{v} x {k}" for k, v in summ[slot].most_common()))
    out = W.ROOT / f"dense_pick_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["rep", "pick_slot", "item", "stage", "err_bin1_m", "moved_red_m", "moved_blue_m", "moved_green_m", "disturbed"])
        wr.writerows(rows)
    print(f"\nSaved {out.name} in {W.ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
