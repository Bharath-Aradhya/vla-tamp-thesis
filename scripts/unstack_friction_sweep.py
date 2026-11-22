"""Unstack friction sweep: how low must the bar friction be for stack + unstack to work, and is the grasp still fine?

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\unstack_friction_sweep.py

The probe showed that with bar friction 0.6 the base bar tilts far less when the top bar is lifted off (unstack
worked in 3 of 5 trials, 0 of 5 at friction 1.0). MuJoCo uses the HIGHER friction of the two touching geoms, so
lowering the bars' friction should mostly lower bar-on-bar friction while fingers and table keep theirs.
This runs stack + unstack at bar friction 0.6 and 0.4: base bars at t1, t2, t3, source bin1,
4 rotations each = 12 trials per value. Reports full cycles OK, what failed, and the base bar's largest
movement. Saves unstack_friction_HHMMSS.csv.
Second version: the unstack now lifts the top bar only 4 cm and carries it away without the extra 6 cm lift
(the first run failed at t1 with "arm could not reach", probably from lifting to 1.06 m close to the body).
Same positions and seeds as the first run, so results are directly comparable.
Needs scripts\\place_validation.py and the new scripts\\stack_unstack_validation.py. Expect about 3 to 4 minutes.
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

FRICTIONS = [0.6, 0.4]
ROTS = 4


def main():
    t0 = time.time()
    su.build_scene()
    src = su.SOURCES["bin1"]
    cases = []
    rng = np.random.default_rng(70)
    for bname, bxy in su.BASES.items():
        for _ in range(ROTS):
            cases.append((bname,
                          (src[0] + rng.uniform(-0.01, 0.01), src[1] + rng.uniform(-0.01, 0.01)),
                          (bxy[0] + rng.uniform(-0.01, 0.01), bxy[1] + rng.uniform(-0.01, 0.01)),
                          int(rng.integers(1 << 30))))
    rows, summary = [], {}
    for mu in FRICTIONS:
        for geom in ("cube_geom", "base_geom"):
            pv.model.geom_friction[pv.model.geom(geom).id, 0] = mu
        why, ok_by_base = Counter(), Counter()
        stacks, cycles = 0, 0
        for (bname, s, b, seed) in cases:
            r = su.trial(s, b, np.random.default_rng(seed))
            stacks += r["stack"] == "ok"
            cycles += r["unstack"] == "ok"
            ok_by_base[bname] += r["unstack"] == "ok"
            if r["stack"] != "ok":
                why["stack: " + r["stack"]] += 1
            elif r["unstack"] != "ok":
                why["unstack: " + r["unstack"]] += 1
            rows.append([mu, bname, r["stack"], r["unstack"], round(r["unstack_err"], 4), round(r["base_shift"], 4)])
            print(f"friction {mu:.2f} base {bname}: stack {r['stack']}; unstack {r['unstack']}"
                  f"   ({time.time() - t0:.0f}s)", flush=True)
        summary[mu] = (stacks, cycles, why, ok_by_base)
    n = len(cases)
    print(f"\nBar friction -> stack OK / full stack+unstack cycles OK, out of {n}")
    for mu, (st, cy, why, byb) in summary.items():
        print(f"  {mu:.2f}: stack {st}/{n}, full cycle {cy}/{n}   (cycles per base: "
              + ", ".join(f"{k} {byb[k]}/{ROTS}" for k in su.BASES) + ")")
        for k, v in why.most_common():
            print(f"        {v} x {k}")
    out = pv.ROOT / f"unstack_friction_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["friction", "base", "stack", "unstack", "unstack_err_m", "base_shift_m"])
        w.writerows(rows)
    print(f"\nSaved {out.name} in {pv.ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
