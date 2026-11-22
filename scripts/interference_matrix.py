"""Interference matrix: for each action at each slot, which neighbouring slots must be EMPTY for it to work?

Run from the project root (D:\\vla-tamp-thesis), one action at a time:
    python scripts\\interference_matrix.py place          (all 6 slots, incl. the spare slot c3)
    python scripts\\interference_matrix.py unstack
    python scripts\\interference_matrix.py pick c3        (pick: only the trials that involve c3, the rest is done)
c3 = (0.48, -0.05), 10 cm to the left of t3: a spare buffer slot so plans can park a bar (p03, p07 need one).

The dense tests showed that the scripted skills push bars standing next to them: picking slides the bar toward the
robot (into the slot 8 cm closer), the hand approaches from the right (over the bin row when picking at the table
row), and the release backs the hand up toward the robot. This measures it systematically. For every slot p
(t1, t2, t3, bin1, bin2, c3) the action is done at p with one other bar standing at each of the other slots q in turn,
and once with no other bar (baseline). 3 random scenes each = 15 trials per slot, 75 per action.
  pick:    a bar at p is grasped and lifted 12 cm (success = lifted, held upright).
  place:   a held bar (picked from a safe slot) is put down at p.
  unstack: a bar on top of a bar at p is lifted off and put down at a free slot.
Output: per slot p, for each q: successes / 3 and how often the bar at q was disturbed (moved > 2 cm or tilted).
From the "disturbed" counts the planner's preconditions can be generated. Saves interference_<action>_HHMMSS.csv.
Needs scripts\\world.py, place_validation.py, stack_unstack_validation.py.
Expect about 3 min (pick c3), 10-12 min (place, unstack). CPU only.
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

W.SLOTS["c3"] = (0.48, -0.05)       # spare buffer slot (validated: 12/12 pick-and-place)
SLOTS = ["t1", "t2", "t3", "bin1", "bin2", "c3"]
C3_ONLY = len(sys.argv) > 2 and sys.argv[2] == "c3"      # only the trials that involve c3
REPS = 3
SOURCES = ["bin1", "t1", "bin2", "t2", "t3", "c3"]      # preferred sources for the place test, safest first


def free_slot(avoid):
    return next(s for s in ["bin1", "t1", "bin2", "t2", "t3", "c3"] if s not in avoid)


def trial(w, action, p, q, seed):
    S = lambda s: ("slot", s)
    placement, moving, target = {}, None, None
    if action == "pick":
        placement["red"] = S(p)
        moving = "red"
    elif action == "place":
        src = next(s for s in SOURCES if s not in (p, q))
        placement["blue"] = S(src)
        moving, target = "blue", p
    else:
        placement["red"] = S(p)
        placement["blue"] = ("on", "red")
        moving, target = "blue", free_slot((p, q))
    if q is not None:
        placement["green"] = S(q)
    w.reset(placement, np.random.default_rng(seed))
    before = {i: w.xy(i) for i in placement}
    ok, why = True, ""
    if action == "pick":
        ctx = w.pick("red")
        ok = ctx is not None and w.tilt("red") <= 20.0
        why = "" if ok else "not lifted"
    elif action == "place":
        ctx = w.pick("blue")
        if ctx is None:
            ok, why = False, "source pick failed"
        else:
            r = w.place_table("blue", p, ctx)
            err = float(np.linalg.norm(w.xy("blue") - np.array(W.SLOTS[p])))
            ok = r is None and err <= 0.02 and w.tilt("blue") <= 8.0 and abs(w.z("blue") - w.table_z) <= 0.012
            why = r or ("" if ok else "landing failed")
    else:
        ctx = w.pick("blue", W.su.UNSTACK_LIFT)
        if ctx is None:
            ok, why = False, "not lifted"
        else:
            r = w.place_table("blue", target, ctx)
            err = float(np.linalg.norm(w.xy("blue") - np.array(W.SLOTS[target])))
            ok = r is None and err <= 0.02 and w.tilt("blue") <= 8.0 and abs(w.z("blue") - w.table_z) <= 0.012
            why = r or ("" if ok else "landing failed")
    others = [i for i in placement if i != moving]
    moved = {i: float(np.linalg.norm(w.xy(i) - before[i])) for i in others}
    disturbed = {i: (moved[i] > 0.02 or w.tilt(i) > 8.0 or w.z(i) < 0.5) for i in others}
    q_dist = bool(disturbed.get("green", False))
    base_dist = bool(disturbed.get("red", False)) if action != "pick" and "red" in others else False
    return ok, why, q_dist, base_dist, moved.get("green", 0.0)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("pick", "place", "unstack"):
        sys.exit("usage: python scripts\\interference_matrix.py pick|place|unstack")
    action = sys.argv[1]
    t0 = time.time()
    w = W.World()
    rows, res = [], {}
    total = sum(1 for p in SLOTS for q in [None] + [x for x in SLOTS if x != p] if (not C3_ONLY or p == "c3" or q == "c3")) * REPS
    n = 0
    for p in SLOTS:
        for q in [None] + [s for s in SLOTS if s != p]:
            if C3_ONLY and p != "c3" and q != "c3":
                continue
            oks = dq = db = 0
            for rep in range(REPS):
                ok, why, q_dist, base_dist, qm = trial(w, action, p, q, 3000 + rep)
                oks += ok
                dq += q_dist
                db += base_dist
                n += 1
                rows.append([action, p, q or "none", rep, int(ok), why, int(q_dist), int(base_dist), round(qm, 4)])
                print(f"[{n:2d}/{total}] {action} at {p:4s}, other bar at {q or 'none':4s}: "
                      f"{'ok' if ok else 'FAIL ' + why}{', neighbour disturbed' if q_dist else ''}"
                      f"{', base disturbed' if base_dist else ''}   ({time.time() - t0:.0f}s)", flush=True)
            res[(p, q)] = (oks, dq, db)
    print(f"\n{action.upper()} at slot p, with another bar at slot q.  Cell = successes/{REPS}  [neighbour disturbed/{REPS}]")
    print("p \\ q      " + "".join(f"{s:>12s}" for s in ["none"] + SLOTS))
    for p in SLOTS:
        line = f"{p:8s}   "
        for q in [None] + SLOTS:
            if q == p or (p, q) not in res:
                line += f"{'-':>12s}"
            else:
                o, d, _ = res[(p, q)]
                line += f"{o}/{REPS} [{d}/{REPS}]".rjust(12) if q else f"{o}/{REPS}".rjust(12)
        print(line)
    out = W.ROOT / f"interference_{action}_{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["action", "slot", "other_at", "rep", "ok", "why", "neighbour_disturbed", "base_disturbed", "neighbour_moved_m"])
        wr.writerows(rows)
    print(f"\nSaved {out.name} in {W.ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
