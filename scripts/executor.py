"""Closed-loop plan executor for condition (b): PDDL planner + scripted skills + observation + replanning.

Usage (from the project root):
    python scripts\\executor.py p01 p02 p05 --reps 5          # in MuJoCo with the scripted skills
    python scripts\\executor.py p01 p05 --reps 3 --dry-run    # no physics: abstract world with random skill failures,
                                                            # only to test the control loop
    options: --domain pddl\\domain_skills.pddl (default)   --bfs (use the built-in search instead of Fast Downward)

Loop (every task, every repetition):
  1. reset the scene from the task's initial state (items at slots / on other items)
  2. observe  : read the bar poses from the simulator and turn them into PDDL facts (at / on / holding / clear / free)
  3. plan     : write a PDDL problem for the OBSERVED state + the task goal, solve it (Fast Downward, or --bfs)
  4. act      : run the plan step by step with the scripted skill; after EVERY step observe again and compare the facts with
                what the plan predicted; if they differ (a pick missed, a neighbour was knocked over, ...) -> go to 2 (replan)
  5. stop     : when the goal holds in the OBSERVED state (success), when no plan exists, or at the replan/step limits
Success is always judged from the observed physics state, never from the plan.
Bars that fell off the table or lie tilted are "lost": they are left out of the problem; if the goal needs one, no plan exists.
Saves executor_HHMMSS.csv (one row per run) in the project root; prints a per-task summary.
NOTE: written without being able to run MuJoCo in the authoring environment; --dry-run was tested.
"""
import argparse
import csv
import random
import sys
import tempfile
import time
from collections import deque
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from symbolic import Problem, apply_action, goal_reached, parse_domain, parse_problem  # noqa: E402

SLOT_XY = {"t1": (0.32, -0.15), "t2": (0.40, -0.15), "t3": (0.48, -0.15), "bin1": (0.32, -0.25), "bin2": (0.40, -0.25)}
SLOT_TOL = 0.06            # an upright bar on the table within 6 cm of a slot centre counts as being at that slot
                           # (it may have been pushed a few cm; the skills pick it where it actually is)
MAX_REPLANS = 4
MAX_SKILLS = 24
PRE_A, PRE_C = (-0.06, -0.10, 0.05), (-0.08, -0.04, 0.05)


# ------------------------------------------------------------------------------------------ facts
def facts_from_obs(obs, held, items):
    """obs: {item: ('slot', name) | ('on', base) | ('lost', why)}; held: item or None -> set of PDDL atoms (tuples)."""
    atoms = set()
    covered = set()
    for i in items:
        kind = obs[i][0]
        if i == held:
            atoms.add(("holding", i))
        elif kind == "slot":
            atoms.add(("at", i, obs[i][1]))
            covered.add(obs[i][1])
        elif kind == "on":
            atoms.add(("on", i, obs[i][1]))
    atoms |= {("free", s) for s in SLOT_XY if s not in covered}
    tops = {obs[i][1] for i in items if i != held and obs[i][0] == "on"}
    atoms |= {("clear", i) for i in items if i != held and obs[i][0] != "lost" and i not in tops}
    if held is None:
        atoms.add(("hand-empty",))
    return atoms


def problem_text(items, atoms, goal):
    fmt = lambda a: "(" + " ".join(a) + ")"
    return ("(define (problem observed)\n  (:domain tabletop-skills)\n"
            f"  (:objects {' '.join(items)} - item)\n  (:init {' '.join(fmt(a) for a in sorted(atoms))})\n"
            f"  (:goal (and {' '.join(fmt(g) for g in goal)})))\n")


def tracked(atoms):
    return {a for a in atoms if a[0] in ("at", "on", "holding")}


# ------------------------------------------------------------------------------------------ planners
def plan_bfs(domain, items, atoms, goal, max_depth=14):
    objs = {i: "item" for i in items}
    prob = Problem("obs", objs, frozenset(atoms), list(goal))
    grounded = []
    for name, act in domain.actions.items():
        n = len(act.params)
        if n == 1:
            grounded += [(name, [i]) for i in items]
        else:
            grounded += [(name, [i, j]) for i in items for j in items if i != j]
    q, seen = deque([(prob.init, [])]), {prob.init}
    while q:
        s, pl = q.popleft()
        if goal_reached(prob, s):
            return pl
        if len(pl) >= max_depth:
            continue
        for name, args in grounded:
            try:
                n_s = apply_action(domain, prob, s, name, args)
            except ValueError:
                continue
            if n_s not in seen:
                seen.add(n_s)
                q.append((n_s, pl + [(name, args)]))
    return None


def make_planner(domain_path, use_bfs):
    domain = parse_domain(domain_path)
    fd_ok = False
    if not use_bfs:
        try:
            import plan as planmod
            fd_ok = True
        except Exception:
            fd_ok = False

    def planner(items, atoms, goal):
        if any(g[1] not in items or (g[0] == "on" and g[2] not in items) for g in goal):
            return None                                   # the goal needs a lost bar
        if fd_ok:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    p = Path(tmp) / "obs.pddl"
                    p.write_text(problem_text(items, atoms, goal), encoding="utf-8")
                    return planmod.solve(p, domain_path)
            except SystemExit:
                pass
        return plan_bfs(domain, items, atoms, goal)
    return planner, domain


# ------------------------------------------------------------------------------------------ backends
class SimBackend:
    """Abstract world for --dry-run: skills succeed with probability P_OK, otherwise nothing happens;
    sometimes a failed pick knocks a neighbour over."""
    P_OK, P_KNOCK = 0.8, 0.15

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def reset(self, placement, items, seed):
        self.loc = dict(placement)
        self.items = items
        self.held = None

    def observe(self):
        return {i: ("held", "") if i == self.held else self.loc.get(i, ("lost", "off table")) for i in self.items}

    def pick(self, item):
        if self.rng.random() < self.P_OK:
            self.loc.pop(item)
            self.held = item
            return True
        others = [j for j in self.items if j != item and self.loc.get(j, ("",))[0] != "lost"]
        if others and self.rng.random() < self.P_KNOCK:
            self.loc[self.rng.choice(others)] = ("lost", "knocked off")
        return False

    def place(self, item, slot):
        ok = self.rng.random() < self.P_OK
        self.loc[item] = ("slot", slot) if ok else ("lost", "landing failed")
        self.held = None
        return ok

    def stack(self, item, base):
        ok = self.rng.random() < self.P_OK
        self.loc[item] = ("on", base) if ok else ("lost", "stack failed")
        self.held = None
        return ok


class MujocoBackend:
    def __init__(self, seed):
        sys.path.insert(0, str(HERE))
        import numpy as np
        import world as W
        self.np, self.W = np, W
        self.w = W.World()
        self.ctx = None
        self.held = None

    def reset(self, placement, items, seed):
        self.items = items
        self.held, self.ctx = None, None
        pl = {i: (k, v) for i, (k, v) in placement.items()}
        self.w.reset(pl, self.np.random.default_rng(seed))

    def observe(self):
        np, w, W = self.np, self.w, self.W
        obs = {}
        self._d = {}
        for i in self.items:
            xy, z, tilt = w.xy(i), w.z(i), w.tilt(i)
            if i == self.held:
                obs[i] = ("held", "")
                continue
            if abs(z - w.table_z) < 0.015 and tilt <= 8.0:
                name, d = min(((s, float(np.linalg.norm(xy - np.array(p)))) for s, p in SLOT_XY.items()), key=lambda t: t[1])
                obs[i] = ("slot", name) if d <= SLOT_TOL else ("lost", f"off-slot {d * 100:.0f} cm")
                self._d[i] = d
                continue
            base = None
            for j in self.items:
                if j != i and tilt <= 12.0 and float(np.linalg.norm(xy - w.xy(j))) < 0.03 and abs(z - w.z(j) - W.BAR_H) < 0.02:
                    base = j
            obs[i] = ("on", base) if base else ("lost", "fallen or tilted")
        for i in self.items:                                  # two bars mapped to the same slot: the farther one is lost
            if obs[i][0] == "slot":
                rivals = [j for j in self.items if j != i and obs[j] == obs[i]]
                if any(self._d[j] < self._d[i] for j in rivals):
                    obs[i] = ("lost", f"slot conflict at {obs[i][1]}")
        return obs

    def pick(self, item):
        np, w = self.np, self.w
        self.W.pv.PRE = np.array(PRE_A)       # approach C was only measured against bin-row bars, which the planner keeps empty
        ctx = w.pick(item)
        ok = ctx is not None and w.tilt(item) <= 20.0
        self.ctx, self.held = (ctx, item) if ok else (None, None)
        return ok

    def place(self, item, slot):
        r = self.w.place_table(item, slot, self.ctx)
        self.ctx, self.held = None, None
        return r is None

    def stack(self, item, base):
        r = self.w.place_on(item, base, self.ctx)
        self.ctx, self.held = None, None
        return r is None


# ------------------------------------------------------------------------------------------ run
def initial_placement(problem):
    pl = {}
    for a in problem.init:
        if a[0] == "at" and a[2] in SLOT_XY:
            pl[a[1]] = ("slot", a[2])
        elif a[0] == "on":
            pl[a[1]] = ("on", a[2])
    return pl


def run_task(backend, planner, domain, problem, seed, verbose=True):
    items = [o for o, t in problem.objects.items() if t == "item"]
    backend.reset(initial_placement(problem), items, seed)
    goal = list(problem.goal)
    replans, skills, first_len, reason, first_dev = -1, 0, None, "", ""
    log = lambda *a: verbose and print("     ", *a, flush=True)
    while True:
        obs = backend.observe()
        held = backend.held
        lost = [i for i in items if obs[i][0] == "lost"]
        atoms = facts_from_obs(obs, held, items)
        present = [i for i in items if obs[i][0] != "lost"]
        if all(g in atoms for g in goal):
            return True, replans, skills, first_len, "goal reached", first_dev
        replans += 1
        if replans > MAX_REPLANS:
            return False, replans, skills, first_len, "replan limit", first_dev
        plan = planner(present, atoms, goal)
        if plan is None:
            return False, replans, skills, first_len, (("no plan (lost: " + ", ".join(f"{i} [{obs[i][1]}]" for i in lost) + ")") if lost else "no plan under the skill rules"), first_dev
        first_len = len(plan) if first_len is None else first_len
        log(f"plan {replans + 1}: " + "; ".join(f"{n} {' '.join(a)}" for n, a in plan))
        state = frozenset(atoms)
        pobj = Problem("p", {i: "item" for i in present}, state, goal)
        deviated = False
        for name, args in plan:
            if skills >= MAX_SKILLS:
                return False, replans, skills, first_len, "skill limit", first_dev
            skills += 1
            kind, i = name.split("-")[0], args[0]
            t0 = time.time()
            if kind == "pick":
                ok = backend.pick(i)
            elif kind == "place":
                ok = backend.place(i, name.split("-")[1])
            else:
                ok = backend.stack(i, args[1])
            state = apply_action(domain, pobj, state, name, args)
            obs2 = backend.observe()
            seen = facts_from_obs(obs2, backend.held, items)
            if tracked(seen) != tracked(state) or not ok:
                log(f"after ({name} {' '.join(args)}) [{time.time() - t0:.0f}s]: observed state differs from the plan -> replan")
                deviated = True
                first_dev = first_dev or f"{name} {' '.join(args)}"
                break
        if not deviated:
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="+", help="task ids such as p01 p02 p05")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bfs", action="store_true")
    ap.add_argument("--domain", default=str(ROOT / "pddl" / "domain_skills.pddl"))
    args = ap.parse_args()
    files = []
    for t in args.tasks:
        f = sorted((ROOT / "pddl" / "problems").glob(f"{t}*.pddl"))
        if not f:
            sys.exit(f"no problem file starting with {t} in pddl\\problems")
        files.append(f[0])
    planner, domain = make_planner(args.domain, args.bfs or args.dry_run)
    backend = SimBackend(7) if args.dry_run else MujocoBackend(0)
    t0, rows = time.time(), []
    for f in files:
        problem = parse_problem(f)
        for rep in range(args.reps):
            seed = 6000 + rep
            if args.dry_run:
                backend.rng = random.Random(seed)
            print(f"{f.name[:3]} rep {rep}:", flush=True)
            ok, replans, skills, first, why, dev = run_task(backend, planner, domain, problem, seed)
            print(f"  -> {'SUCCESS' if ok else 'FAIL'} ({why}){' first deviation after ' + dev if dev else ''}; skills {skills}, replans {max(replans, 0)}   ({time.time() - t0:.0f}s)", flush=True)
            rows.append([f.name[:3], rep, seed, int(ok), skills, max(replans, 0), first or 0, dev, why])
    print("\ntask   success   mean skills   mean replans")
    for f in files:
        r = [x for x in rows if x[0] == f.name[:3]]
        print(f"{f.name[:3]}    {sum(x[3] for x in r)}/{len(r)}      {sum(x[4] for x in r) / len(r):.1f}           {sum(x[5] for x in r) / len(r):.1f}")
    out = ROOT / f"executor_{'dry_' if args.dry_run else ''}{datetime.now():%H%M%S}.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["task", "rep", "seed", "success", "skills", "replans", "first_plan_len", "first_deviation_after", "reason"])
        wr.writerows(rows)
    print(f"\nSaved {out.name} in {ROOT}   (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
