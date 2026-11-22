"""Corner validation: is the far-right corner (0.48, -0.25) good enough as the sixth slot (PDDL region r4)?

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\corner_validation.py

Uses the frozen pick-and-place (scripts\\place_validation.py) with bar friction 0.6. Every ordered pair that involves
the corner (corner -> each of the 5 validated slots, and each of them -> corner) = 10 pairs x 4 random rotations
= 40 trials. OK = ends within 2 cm of the target, standing, on the table. Prints success per direction,
failure stages, landing error. Saves corner_validation_HHMMSS.csv. Expect about 5 to 6 minutes. CPU only.
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

FRICTION = 0.6
CORNER = ("r4", (0.48, -0.25))
OTHERS = list(zip(pv.NAMES, pv.PLACE_LOCATIONS))
TRIALS = 4


def main():
    t0 = time.time()
    pv.load_scene(pv.SHAPE)
    pv.model.geom_friction[pv.model.geom("cube_geom").id, 0] = FRICTION
    rng = np.random.default_rng(80)
    pairs = [(CORNER, o) for o in OTHERS] + [(o, CORNER) for o in OTHERS]
    total = len(pairs) * TRIALS
    print(f"bar friction {FRICTION}; {len(pairs)} pairs x {TRIALS} rotations = {total} trials")
    rows, stages, errs, ok_by = [], Counter(), [], Counter()
    done = 0
    for (sn, sxy), (dn, dxy) in pairs:
        for _ in range(TRIALS):
            s = (sxy[0] + rng.uniform(-pv.JITTER, pv.JITTER), sxy[1] + rng.uniform(-pv.JITTER, pv.JITTER))
            yaw = float(rng.uniform(0, 90))
            r = pv.place_once(s, dxy, yaw)
            stages[r["stage"]] += 1
            ok_by[(sn, dn)] += r["stage"] == "ok"
            if r["stage"] == "ok":
                errs.append(r["err_xy"])
            done += 1
            print(f"[{done:2d}/{total}] {sn:4s} -> {dn:4s} rot {yaw:4.1f}: {r['stage']}"
                  + (f", {r['err_xy'] * 100:.1f} cm" if r["err_xy"] == r["err_xy"] else "")
                  + f"   ({time.time() - t0:.0f}s)", flush=True)
            rows.append([sn, dn, round(s[0], 3), round(s[1], 3), round(yaw, 1), r["stage"], round(r["err_xy"], 4),
                         round(r["z"], 3), round(r["tilt_final"], 1), round(r["pick_lift"], 3), round(r["ik_carry"], 3)])
    print(f"\nOK: {stages['ok']} of {total} ({100 * stages['ok'] / total:.0f}%)")
    for k, v in stages.most_common():
        if k != "ok":
            print(f"   {v:3d}  {k}")
    if errs:
        e = np.array(errs) * 100
        print(f"Landing error of the OK trials: mean {e.mean():.1f} cm, median {np.median(e):.1f} cm, worst {e.max():.1f} cm")
    print(f"\nOK out of {TRIALS}:  corner -> slot   |   slot -> corner")
    for n, _ in OTHERS:
        print(f"  {n:5s}  {ok_by[('r4', n)]}/{TRIALS}   |   {ok_by[(n, 'r4')]}/{TRIALS}")
    out = pv.ROOT / f"corner_validation_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "start_x", "start_y", "cube_yaw_deg", "stage", "err_xy_m", "final_z_m",
                    "tilt_final_deg", "pick_lift_m", "ik_carry_m"])
        w.writerows(rows)
    print(f"\nSaved {out.name} in {pv.ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
