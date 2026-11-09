"""Minimal STRIPS reader and state simulator for the thesis PDDL domain.

Why it exists: (1) it checks every plan from Fast Downward independently, and (2) the
planner-VLA loop needs to track the symbolic world state, apply an action's effects and
test the goal, which is exactly what this does. It supports only what the domain uses:
typed objects, positive preconditions, add/delete effects, goals that are a conjunction.
"""
import re
from dataclasses import dataclass
from pathlib import Path


def _tokenize(text):
    text = re.sub(r";[^\n]*", "", text)
    return re.findall(r"\(|\)|[^\s()]+", text.lower())


def _parse(tokens):
    def read(i):
        if tokens[i] == "(":
            out, i = [], i + 1
            while tokens[i] != ")":
                node, i = read(i)
                out.append(node)
            return out, i + 1
        return tokens[i], i + 1
    return read(0)[0]


def _typed_list(tokens):
    """['a','b','-','t','c','-','u'] -> [('a','t'),('b','t'),('c','u')]"""
    out, pending, i = [], [], 0
    while i < len(tokens):
        if tokens[i] == "-":
            out += [(n, tokens[i + 1]) for n in pending]
            pending, i = [], i + 2
        else:
            pending.append(tokens[i])
            i += 1
    assert not pending, f"untyped names in {tokens}"
    return out


def _conjuncts(expr):
    if expr[0] == "and":
        for e in expr[1:]:
            yield from _conjuncts(e)
    else:
        yield expr


@dataclass
class Action:
    name: str
    params: list      # [(variable, type)]
    pre: list         # [atom tuples with variables]
    add: list
    delete: list


@dataclass
class Domain:
    actions: dict


@dataclass
class Problem:
    name: str
    objects: dict     # name -> type
    init: frozenset
    goal: list        # atoms that must all hold


def parse_domain(path):
    tree = _parse(_tokenize(Path(path).read_text()))
    actions = {}
    for item in tree:
        if isinstance(item, list) and item and item[0] == ":action":
            props = {item[k]: item[k + 1] for k in range(2, len(item), 2)}
            add, delete = [], []
            for e in _conjuncts(props[":effect"]):
                (delete if e[0] == "not" else add).append(tuple(e[1] if e[0] == "not" else e))
            pre = []
            for e in _conjuncts(props[":precondition"]):
                assert e[0] != "not", "negative preconditions are not supported"
                pre.append(tuple(e))
            actions[item[1]] = Action(item[1], _typed_list(props[":parameters"]), pre, add, delete)
    return Domain(actions)


def parse_problem(path):
    tree = _parse(_tokenize(Path(path).read_text()))
    name, objects, init, goal = "", {}, set(), []
    for item in tree:
        if not isinstance(item, list) or not item:
            continue
        if item[0] == "problem":
            name = item[1]
        elif item[0] == ":objects":
            objects = dict(_typed_list(item[1:]))
        elif item[0] == ":init":
            init = {tuple(a) for a in item[1:]}
        elif item[0] == ":goal":
            goal = [tuple(a) for a in _conjuncts(item[1])]
    return Problem(name, objects, frozenset(init), goal)


def parse_plan(text):
    """Fast Downward plan file -> [('pick', ['red', 't1']), ...]"""
    plan = []
    for line in text.splitlines():
        line = line.strip().lower()
        if line.startswith("("):
            parts = line.strip("()").split()
            plan.append((parts[0], parts[1:]))
    return plan


def apply_action(domain, problem, state, name, args):
    """Return the new state, or raise ValueError if the action is not applicable."""
    act = domain.actions[name]
    if len(args) != len(act.params):
        raise ValueError(f"{name} needs {len(act.params)} arguments, got {args}")
    for a, (_, typ) in zip(args, act.params):
        if problem.objects.get(a) != typ:
            raise ValueError(f"{a} is not of type {typ} in {name}")
    binding = {v: a for (v, _), a in zip(act.params, args)}
    ground = lambda atom: tuple(binding.get(t, t) for t in atom)
    missing = [ground(p) for p in act.pre if ground(p) not in state]
    if missing:
        raise ValueError(f"({name} {' '.join(args)}) not applicable, missing: {missing}")
    return frozenset((state - {ground(d) for d in act.delete}) | {ground(a) for a in act.add})


def goal_reached(problem, state):
    return all(g in state for g in problem.goal)


def validate_plan(domain, problem, plan):
    """Execute the plan from the initial state. Returns (ok, message, final_state)."""
    state = problem.init
    for step, (name, args) in enumerate(plan, 1):
        try:
            state = apply_action(domain, problem, state, name, args)
        except ValueError as e:
            return False, f"step {step}: {e}", state
    if not goal_reached(problem, state):
        return False, "plan executes but the goal does not hold at the end", state
    return True, "ok", state
