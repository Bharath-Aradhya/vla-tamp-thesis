"""Unstack grasp height sweep: where on the top bar should the hand grip so it does not catch the base bar?

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\stack_unstack_dz.py

The debug pictures show that when the top bar is lifted off, the hand also catches the base bar (the lower finger
sits right at the joint between the two bars). This tries gripping the top bar higher, with the grasp point
0, 0.8, 1.2 and 1.6 cm above its centre, on 6 stacks each (3 bases x 2 sources, same positions and rotations for
every height). The stack itself is unchanged; only the unstack grasp moves. Prints per height how many
stack+unstack cycles worked, and what went wrong otherwise. Saves stack_unstack_dz_HHMMSS.csv.
Needs scripts\\place_validation.py and the NEW scripts\\stack_unstack_validation.py. Expect about 3 to 4 minutes.
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
import place_validation as pv
import stack_unstack_validation as su

DZS = [0.0, 0.008, 0.012, 0.016]


def main():
    t0 = time.time()
    su.build_scene()
    cases = []
    rng = np.random.default_rng(50)
    for bname, bxy in su.BASES.items():
        for sname, sxy in su.SOURCES.items():
            cases.append((bname, sname,
                          (sxy[0] + rng.uniform(-0.01, 0.01), sxy[1] + rng.uniform(-0.01, 0.01)),
                          (bxy[0] + rng.uniform(-0.01, 0.01), bxy[1] + rng.uniform(-0.01, 0.01)),
                          int(rng.integers(1 << 30))))
    rows, summary = [], {}
    for dz in DZS:
        stack_ok, cycle_ok, why = 0, 0, Counter()
        for (bname, sname, s, b, seed) in cases:
            r = su.trial(s, b, np.random.default_rng(seed), unstack_dz=dz)
            stack_ok += r["stack"] == "ok"
            cycle_ok += r["unstack"] == "ok"
            if r["stack"] != "ok":
                why["stack: " + r["stack"]] += 1
            elif r["unstack"] != "ok":
                why["unstack: " + r["unstack"]] += 1
            rows.append([dz, sname, bname, r["stack"], r["unstack"], round(r["unstack_err"], 4), round(r["base_shift"], 4)])
            print(f"dz {dz * 100:.1f} cm  {sname} -> {bname}: stack {r['stack']}; unstack {r['unstack']}"
                  f"   ({time.time() - t0:.0f}s)", flush=True)
        summary[dz] = (stack_ok, cycle_ok, why)
    n = len(cases)
    print(f"\nGrip height above the top bar's centre -> full cycles OK (stack then unstack), out of {n}")
    for dz, (so, co, why) in summary.items():
        print(f"  {dz * 100:4.1f} cm: {co}/{n}   (stacks ok {so}/{n})")
        for k, v in why.most_common():
            print(f"        {v} x {k}")
    out = pv.ROOT / f"stack_unstack_dz_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unstack_dz_m", "src", "base", "stack", "unstack", "unstack_err_m", "base_shift_m"])
        w.writerows(rows)
    print(f"\nSaved {out.name} in {pv.ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
