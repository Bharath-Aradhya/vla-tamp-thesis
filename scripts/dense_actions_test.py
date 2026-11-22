"""Dense actions test: do PLACE, STACK and UNSTACK disturb the bars standing next to them?

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\dense_actions_test.py

dense_pick_test.py showed the picked bar always arrives, but picking slides it toward the robot into its neighbour:
picking at t2 pushed the bar at t1 by 8-26 cm in 6 of 6 scenes, picking at t3 pushed t2 in 6 of 6; picking at t1
disturbed its neighbour in 2 of 6 (1-3 cm). This tests the other actions with bars standing next to the target:
  place  blue (from bin1) at t2 with red at t1 and green at t3 | at t1 with green at t2 | at t3 with green at t2
  place  blue (from t2) at bin2 with red at bin1              | place blue (from t1) at bin1 with red at bin2
  stack  green (from bin1) onto blue at t2 with red at t1 | onto blue at t1 with red at t2 | onto blue at t3 with red at t2
  unstack blue off red at t2 (green at t1) | off red at t1 (green at t2) | off red at t3 (green at t2), into bin1
4 random scenes each = 44 trials. OK = the moved bar ends where it should (within 2 cm, standing, right height) and
every other bar stays within 2 cm of where it was and standing. Saves dense_actions_HHMMSS.csv.
Needs scripts\\world.py (the new one), place_validation.py, stack_unstack_validation.py. Expect about 6 minutes.
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

REPS = 4
S = lambda slot: ("slot", slot)
CASES = [
    ("place at t2 (red t1, green t3)", {"red": S("t1"), "green": S("t3"), "blue": S("bin1")}, ("place", "blue", "t2")),
    ("place at t1 (green t2)", {"green": S("t2"), "blue": S("bin1")}, ("place", "blue", "t1")),
    ("place at t3 (green t2)", {"green": S("t2"), "blue": S("bin1")}, ("place", "blue", "t3")),
    ("place at bin2 (red bin1)", {"red": S("bin1"), "blue": S("t2")}, ("place", "blue", "bin2")),
    ("place at bin1 (red bin2)", {"red": S("bin2"), "blue": S("t1")}, ("place", "blue", "bin1")),
    ("stack on t2 (red t1)", {"red": S("t1"), "blue": S("t2"), "green": S("bin1")}, ("stack", "green", "blue")),
    ("stack on t1 (red t2)", {"red": S("t2"), "blue": S("t1"), "green": S("bin1")}, ("stack", "green", "blue")),
    ("stack on t3 (red t2)", {"red": S("t2"), "blue": S("t3"), "green": S("bin1")}, ("stack", "green", "blue")),
    ("unstack at t2 (green t1)", {"red": S("t2"), "blue": ("on", "red"), "green": S("t1")}, ("unstack", "blue", "bin1")),
    ("unstack at t1 (green t2)", {"red": S("t1"), "blue": ("on", "red"), "green": S("t2")}, ("unstack", "blue", "bin1")),
    ("unstack at t3 (green t2)", {"red": S("t3"), "blue": ("on", "red"), "green": S("t2")}, ("unstack", "blue", "bin1")),
]


def run(w, placement, action, seed):
    w.reset(placement, np.random.default_rng(seed))
    before = {i: w.xy(i) for i in placement}
    kind, item, target = action
    ctx = w.pick(item, W.su.UNSTACK_LIFT if kind == "unstack" else W.LIFT_BY)
    if ctx is None:
        why = "pick failed (bar not lifted)"
    elif kind == "stack":
        why = w.place_on(item, target, ctx)
    else:
        why = w.place_table(item, target, ctx)
    # where the moved bar should be
    if kind == "stack":
        base = w.xy(target)
        err = float(np.linalg.norm(w.xy(item) - base))
        z_ok = abs(w.z(item) - (w.z(target) + W.BAR_H)) <= 0.012
    else:
        err = float(np.linalg.norm(w.xy(item) - np.array(W.SLOTS[target])))
        z_ok = abs(w.z(item) - w.table_z) <= 0.012
    others = [i for i in placement if i != item]
    moved = {i: float(np.linalg.norm(w.xy(i) - before[i])) for i in others}
    disturbed = [i for i in others if moved[i] > 0.02 or w.tilt(i) > 8.0 or w.z(i) < 0.5]
    if why:
        stage = why
    elif w.z(item) < 0.5 or w.tilt(item) > 8.0 or err > 0.02 or not z_ok:
        stage = "landing failed"
    else:
        stage = "ok"
    return stage, err, disturbed, moved


def main():
    t0 = time.time()
    w = W.World()
    rows, summ, n = [], {}, 0
    total = len(CASES) * REPS
    for name, placement, action in CASES:
        summ[name] = Counter()
        for rep in range(REPS):
            stage, err, disturbed, moved = run(w, placement, action, 2000 + rep)
            label = ("ok" if stage == "ok" and not disturbed else
                     "bar ok but neighbour disturbed" if stage == "ok" else stage)
            summ[name][label] += 1
            n += 1
            print(f"[{n:2d}/{total}] {name}: {stage}" + (f" + disturbed {'/'.join(disturbed)}" if disturbed else "")
                  + "".join(f" ({i} moved {moved[i] * 100:.1f} cm)" for i in moved if moved[i] > 0.01)
                  + f"   ({time.time() - t0:.0f}s)", flush=True)
            rows.append([name, rep, stage, round(err, 4), "/".join(disturbed)] + [round(v, 4) for v in moved.values()])
    print("\nCase -> outcomes (4 scenes each)")
    for name, c in summ.items():
        print(f"  {name:34s} " + ", ".join(f"{v} x {k}" for k, v in c.most_common()))
    out = W.ROOT / f"dense_actions_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["case", "rep", "stage", "err_m", "disturbed", "moved_other_1_m", "moved_other_2_m"])
        wr.writerows(rows)
    print(f"\nSaved {out.name} in {W.ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
