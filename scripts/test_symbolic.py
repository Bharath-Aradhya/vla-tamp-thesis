"""Checks of the PDDL domain with the state simulator (no planner needed).

Run from the project root:   python scripts/test_symbolic.py
Valid hand-written plans must be accepted; invalid ones must be rejected.
"""
from pathlib import Path

from symbolic import parse_domain, parse_problem, validate_plan

ROOT = Path(__file__).resolve().parent.parent
domain = parse_domain(ROOT / "pddl" / "domain.pddl")


def problem(prefix):
    return parse_problem(next((ROOT / "pddl" / "problems").glob(prefix + "*.pddl")))


VALID = {
    "p01": [("pick", ["red", "t1"]), ("place", ["red", "bin1"])],
    "p02": [("pick", ["red", "t1"]), ("stack", ["red", "blue"])],
    "p03": [("pick", ["blue", "t2"]), ("stack", ["blue", "green"]), ("pick", ["red", "t1"]), ("stack", ["red", "blue"])],
    "p04": [("unstack", ["red", "blue"]), ("place", ["red", "bin1"])],
    "p05": [("push", ["blue", "t2", "t3"]), ("push", ["red", "t1", "t2"]), ("push", ["blue", "t3", "t1"])],
    "p06": [("unstack", ["blue", "red"]), ("place", ["blue", "t2"]), ("pick", ["red", "t1"]), ("place", ["red", "bin1"])],
    "p07": [("pick", ["red", "t1"]), ("place", ["red", "bin1"]), ("pick", ["blue", "t2"]), ("place", ["blue", "bin2"])],
}
INVALID = {
    "pick the same block twice": ("p01", [("pick", ["red", "t1"]), ("pick", ["red", "t1"])]),
    "stack without holding": ("p02", [("stack", ["red", "blue"])]),
    "pick a block that is covered": ("p06", [("pick", ["red", "t1"])]),
    "push onto an occupied spot": ("p05", [("push", ["red", "t1", "t2"])]),
    "push a block into the bin": ("p01", [("push", ["red", "t1", "bin1"])]),
    "goal not reached": ("p01", [("pick", ["red", "t1"]), ("place", ["red", "t2"])]),
    "wrong argument type": ("p01", [("pick", ["t1", "red"])]),
}

failed = 0
for key, plan in VALID.items():
    ok, msg, _ = validate_plan(domain, problem(key), plan)
    print(f"valid plan {key}: {'ok' if ok else 'FAILED - ' + msg}")
    failed += not ok
for label, (key, plan) in INVALID.items():
    ok, msg, _ = validate_plan(domain, problem(key), plan)
    print(f"rejects '{label}': {'ok' if not ok else 'FAILED - it was accepted'}")
    failed += ok
print("\nAll tests passed." if not failed else f"\n{failed} test(s) FAILED.")
raise SystemExit(1 if failed else 0)
