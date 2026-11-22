"""Dense-scene chain test: pick the target next to another bar, move it away, then fix up the neighbour it disturbed.

Run from the project root (D:\\vla-tamp-thesis):
    python scripts\\dense_chain_test.py

approach_direction_test.py showed no pre-grasp direction leaves the neighbour untouched in the dense cases, but approach C
(-8,-4 cm) lifts the target in 13/15 dense picks (today's A: 5/15) and keeps 8/9 alone. So the rule tried here is:
  use C when another bar stands within 13 cm of the target, otherwise today's A.
Then, because a closed-loop executor re-observes the scene anyway, this checks whether the disturbance is recoverable:
  1. pick the target (red) at its slot with that rule, put it down at the spare slot c3 (0.48,-0.05)
  2. look at the neighbour (green): still in its slot (< 2 cm), moved but upright on the table, or fallen / off the table
  3. if it is upright on the table: pick it where it ACTUALLY is now and put it back on its own slot
Cases: t1+bin1, t2+bin1, t2+bin2, t3+bin2, t3+bin1, 4 scenes each = 20 chains. Saves dense_chain_HHMMSS.csv.
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

W.SLOTS["c3"] = (0.48, -0.05)
PRE_A = np.array([-0.06, -0.10, 0.05])
PRE_C = np.array([-0.08, -0.04, 0.05])
NEAR = 0.13
CASES = [("t1", "bin1"), ("t2", "bin1"), ("t2", "bin2"), ("t3", "bin2"), ("t3", "bin1")]
REPS = 4


def state_of(w, item, slot):
    off = float(np.linalg.norm(w.xy(item) - np.array(W.SLOTS[slot])))
    on_table = abs(w.z(item) - w.table_z) < 0.015
    up = w.tilt(item) <= 8.0
    if on_table and up and off < 0.02:
        return "in_slot", off
    if on_table and up:
        return "displaced", off
    if on_table:
        return "tipped", off
    return "fallen", off


def main():
    t0 = time.time()
    w = W.World()
    pv = W.pv
    rows, n = [], 0
    total = len(CASES) * REPS
    for (p, q) in CASES:
        for rep in range(REPS):
            n += 1
            w.reset({"red": ("slot", p), "green": ("slot", q)}, np.random.default_rng(5000 + rep))
            near = float(np.linalg.norm(w.xy("red") - w.xy("green"))) < NEAR
            pv.PRE = (PRE_C if near else PRE_A).copy()
            ctx = w.pick("red")
            held = ctx is not None and w.tilt("red") <= 20.0
            place1 = None
            if held:
                place1 = w.place_table("red", "c3", ctx) or "ok"
            red_ok = held and place1 == "ok" and float(np.linalg.norm(w.xy("red") - np.array(W.SLOTS["c3"]))) < 0.02
            st, off = state_of(w, "green", q)
            recovered = ""
            err2 = ""
            if st in ("in_slot", "displaced") and red_ok:
                pv.PRE = PRE_A.copy()
                ctx2 = w.pick("green")
                if ctx2 is None:
                    recovered = "pick_failed"
                else:
                    r2 = w.place_table("green", q, ctx2)
                    e = float(np.linalg.norm(w.xy("green") - np.array(W.SLOTS[q])))
                    recovered = "ok" if (r2 is None and e < 0.02 and w.tilt("green") <= 8.0) else "place_failed"
                    err2 = round(e, 4)
            print(f"[{n:2d}/{total}] {p}+{q} rule={'C' if near else 'A'}: target "
                  f"{'moved to c3' if red_ok else ('FAIL ' + (place1 or 'not lifted'))}; neighbour {st} ({off * 100:.1f} cm off)"
                  f"{'; recovery ' + recovered if recovered else ''}   ({time.time() - t0:.0f}s)", flush=True)
            rows.append([p, q, rep, int(red_ok), st, round(off, 4), recovered, err2])
    print("\ncase        target ok   neighbour in_slot/displaced/tipped/fallen   recovered")
    for (p, q) in CASES:
        r = [x for x in rows if x[0] == p and x[1] == q]
        cnt = lambda s: sum(1 for x in r if x[4] == s)
        print(f"{p}+{q:5s}   {sum(x[3] for x in r)}/{len(r)}        "
              f"{cnt('in_slot')}/{cnt('displaced')}/{cnt('tipped')}/{cnt('fallen')}                       "
              f"{sum(1 for x in r if x[6] == 'ok')}/{len(r)}")
    ok_all = sum(1 for x in rows if x[3] and (x[4] == 'in_slot' or x[6] == 'ok'))
    print(f"\nWhole chain fine (target moved AND neighbour intact or recovered): {ok_all}/{len(rows)}")
    out = W.ROOT / f"dense_chain_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["pick_slot", "other_slot", "rep", "target_ok", "neighbour_state", "neighbour_off_m", "recovery", "recovery_err_m"])
        wr.writerows(rows)
    print(f"\nSaved {out.name} in {W.ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
