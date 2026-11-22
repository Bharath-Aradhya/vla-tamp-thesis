"""Candidate sixth slots: which spot besides the far corner can take a bar reliably (both pick from and place into)?

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\candidate_slots.py

The far corner (0.48, -0.25) is out: the arm undershoots by about 3 cm there when lowering to the table (the
corrected aim still failed 13 of 20), i.e. it is at the edge of what the fixed stance can reach. This tests five other
spots, each 10 cm from its neighbours:
  c1 (0.32, -0.05)   c2 (0.40, -0.05)   c3 (0.48, -0.05)     (a row to the left of t1-t3)
  c4 (0.32, -0.35)   c5 (0.40, -0.35)                         (a row to the right of bin1, bin2)
For each: from t2 and from bin1 INTO the candidate, and from the candidate to t2 and to bin1, 3 random rotations each
= 12 trials per candidate, 60 in total. Bar friction 0.6, standard aim correction. Prints OK per candidate and
direction, the mean landing error in the hand frame (to see if a calibration would help) and failure stages.
Saves candidate_slots_HHMMSS.csv. Expect about 4 to 5 minutes. CPU only.
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
CANDIDATES = {"c1": (0.32, -0.05), "c2": (0.40, -0.05), "c3": (0.48, -0.05), "c4": (0.32, -0.35), "c5": (0.40, -0.35)}
PARTNERS = {"t2": (0.40, -0.15), "bin1": (0.32, -0.25)}
TRIALS = 3


def main():
    t0 = time.time()
    pv.load_scene(pv.SHAPE)
    pv.model.geom_friction[pv.model.geom("cube_geom").id, 0] = FRICTION
    rng = np.random.default_rng(100)
    rows, summ = [], {}
    total = len(CANDIDATES) * len(PARTNERS) * 2 * TRIALS
    done = 0
    for cname, cxy in CANDIDATES.items():
        into_ok = from_ok = 0
        stages = Counter()
        ex, ey = [], []
        for pname, pxy in PARTNERS.items():
            for direction in ("into", "from"):
                src, dst = ((pxy, cxy) if direction == "into" else (cxy, pxy))
                sn, dn = ((pname, cname) if direction == "into" else (cname, pname))
                for _ in range(TRIALS):
                    s = (src[0] + rng.uniform(-pv.JITTER, pv.JITTER), src[1] + rng.uniform(-pv.JITTER, pv.JITTER))
                    yaw = float(rng.uniform(0, 90))
                    r = pv.place_once(s, dst, yaw)
                    ok = r["stage"] == "ok"
                    stages[r["stage"]] += 1
                    if direction == "into":
                        into_ok += ok
                        if r["ex_hand"] == r["ex_hand"] and r["tilt_final"] <= 8.0 and r["err_xy"] < 0.08:
                            ex.append(r["ex_hand"])
                            ey.append(r["ey_hand"])
                    else:
                        from_ok += ok
                    done += 1
                    print(f"[{done:2d}/{total}] {sn:4s} -> {dn:4s} rot {yaw:4.1f}: {r['stage']}"
                          + (f", {r['err_xy'] * 100:.1f} cm" if r["err_xy"] == r["err_xy"] else "")
                          + f"   ({time.time() - t0:.0f}s)", flush=True)
                    rows.append([sn, dn, round(s[0], 3), round(s[1], 3), round(yaw, 1), r["stage"], round(r["err_xy"], 4),
                                 round(r["ex_hand"], 4), round(r["ey_hand"], 4), round(r["tilt_final"], 1)])
        summ[cname] = (into_ok, from_ok, stages, ex, ey)
    per = TRIALS * len(PARTNERS)
    print(f"\nCandidate slot -> OK placing INTO it / OK picking FROM it (out of {per} each), mean hand-frame landing error")
    for cname, (i, f, st, ex, ey) in summ.items():
        err = (f"({np.mean(ex) * 100:+.1f}, {np.mean(ey) * 100:+.1f}) cm" if ex else "n/a")
        print(f"  {cname} {CANDIDATES[cname]}:  into {i}/{per}   from {f}/{per}   landing error {err}")
        for k, v in st.most_common():
            if k != "ok":
                print(f"        {v} x {k}")
    out = pv.ROOT / f"candidate_slots_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["src", "dst", "start_x", "start_y", "yaw_deg", "stage", "err_xy_m", "ex_hand_m", "ey_hand_m", "tilt_final_deg"])
        w.writerows(rows)
    print(f"\nSaved {out.name} in {pv.ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
