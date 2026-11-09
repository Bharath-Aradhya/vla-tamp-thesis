"""Run Fast Downward on the thesis PDDL domain and check the plans.

Usage (from the project root):
    python scripts/plan.py pddl/problems/p03_three_tower.pddl     # one problem
    python scripts/plan.py --all                                    # all problems + self-check

Two ways to get Fast Downward (the script tries them in this order):
  1. A source build: set the environment variable FAST_DOWNWARD to the folder that contains
     fast-downward.py (or put it in ./tools/downward or ./downward). Tested on Linux.
  2. The pip package (prebuilt wheels, no compiler needed, also for Windows):
         pip install unified-planning up-fast-downward
     This route is NOT yet tested by the author; please report any error.
Plans are optimal (A* with the LM-cut heuristic) where the engine allows it, which is fast
for a domain this small.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from symbolic import parse_domain, parse_plan, parse_problem, validate_plan

ROOT = Path(__file__).resolve().parent.parent
DOMAIN = ROOT / "pddl" / "domain.pddl"
PROBLEMS = ROOT / "pddl" / "problems"

# plan length expected for each task, as a self-check (shortest possible plan)
EXPECTED = {"p01": 2, "p02": 2, "p03": 4, "p04": 2, "p05": 3, "p06": 4, "p07": 4}


def find_fast_downward():
    """Path to fast-downward.py of a source build, or None."""
    candidates = []
    if os.environ.get("FAST_DOWNWARD"):
        candidates.append(Path(os.environ["FAST_DOWNWARD"]))
    candidates += [ROOT / "tools" / "downward", ROOT / "downward"]
    for c in candidates:
        if (c / "fast-downward.py").exists():
            return c / "fast-downward.py"
    return None


def _solve_with_unified_planning(problem_path, domain_path):
    try:
        from unified_planning.io import PDDLReader
        from unified_planning.shortcuts import OneshotPlanner
    except ImportError:
        sys.exit("Fast Downward not found. Either build it and set FAST_DOWNWARD, or run:\n"
                 "    pip install unified-planning up-fast-downward\n(see README.md)")
    up_problem = PDDLReader().parse_problem(str(domain_path), str(problem_path))
    try:
        planner = OneshotPlanner(name="fast-downward-opt")      # optimal engine, if available
    except Exception:
        planner = OneshotPlanner(name="fast-downward")
    with planner:
        result = planner.solve(up_problem)
    if result.plan is None:
        return None
    return [(a.action.name.lower(), [str(p).lower() for p in a.actual_parameters])
            for a in result.plan.actions]


def solve(problem_path, domain_path=DOMAIN, timeout=120):
    """Return the plan as [(action, [args])], or None if the problem has no plan."""
    fd = find_fast_downward()
    if fd is None:
        return _solve_with_unified_planning(problem_path, domain_path)
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, str(fd), str(domain_path), str(problem_path),
               "--search", "astar(lmcut())"]
        subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout)
        plan_file = Path(tmp) / "sas_plan"
        if not plan_file.exists():
            return None
        return parse_plan(plan_file.read_text())


def show(plan):
    return "  " + "\n  ".join(f"{i}. ({n} {' '.join(a)})" for i, (n, a) in enumerate(plan, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem", nargs="?", help="path to a problem .pddl file")
    ap.add_argument("--all", action="store_true", help="solve and check every problem")
    args = ap.parse_args()
    files = sorted(PROBLEMS.glob("*.pddl")) if args.all else [Path(args.problem)]
    if not args.all and not args.problem:
        ap.error("give a problem file or --all")

    domain = parse_domain(DOMAIN)
    bad = 0
    for f in files:
        problem = parse_problem(f)
        plan = solve(f)
        if plan is None:
            print(f"{f.name}: NO PLAN FOUND")
            bad += 1
            continue
        ok, msg, _ = validate_plan(domain, problem, plan)
        want = EXPECTED.get(f.name[:3])
        length_note = "" if want is None or want == len(plan) else f"  (note: shortest known plan has {want} steps)"
        print(f"{f.name}: {len(plan)} steps, independent check: {msg}{length_note}")
        print(show(plan))
        bad += not ok
    print("\nAll good." if not bad else f"\n{bad} problem(s) need attention.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
