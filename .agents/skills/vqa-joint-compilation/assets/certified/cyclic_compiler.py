"""Exact occupancy synthesis for an *unbounded-round* finite public-state VQA.

Contract. Each action has a positive rational conditional mean duration, a
complete outcome law, and certified peak and additional sequential physical
depth. Control state is fully observable and sufficient even after feedback.
A terminal failure event may share its ordinary reported output label with a
success. Logical failure counts the flag, fidelity sees the declared labels.
The comparison distribution is a fixed EXTERNAL ideal job specification;
a policy-dependent ideal model requires its own sufficient joint statistics.
Uncharged, independent classical randomization is allowed at public states.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction

import z3

from optimal_compiler import Diagram, Result, solve

Rational = int | Fraction
State = tuple[str, int | None]
Choice = tuple[State, str]


def _f(x: Rational) -> Fraction:
    if isinstance(x, bool) or not isinstance(x, (int, Fraction)):
        raise TypeError("exact int or Fraction required")
    return Fraction(x)


def _q(x: Rational):
    q = _f(x)
    return z3.RealVal(f"{q.numerator}/{q.denominator}")


def _sum(xs):
    return z3.simplify(sum(xs, z3.RealVal(0)))


def _true(expr):
    simple = z3.simplify(expr)
    if not (z3.is_true(simple) or z3.is_false(simple)):
        raise ArithmeticError("verification requires an exact ground truth value")
    return z3.is_true(simple)


@dataclass(frozen=True)
class Outcome:
    probability: Rational
    next_state: str | None = None
    output: str | None = None
    failed: bool = False


@dataclass(frozen=True)
class Action:
    name: str
    duration: Rational
    peak_atoms: int
    depth: int
    outcomes: tuple[Outcome, ...]


@dataclass(frozen=True)
class Job:
    start: str
    actions: dict[str, tuple[Action, ...]]
    ideal: dict[str, Rational]
    max_atoms: int
    max_depth: int | None
    error_limit: Rational
    fidelity_min: Rational


@dataclass(frozen=True)
class Statistics:
    expected_time: z3.ArithRef
    failure: z3.ArithRef
    output: dict[str, z3.ArithRef]
    fidelity: z3.ArithRef
    visits: dict[State, z3.ArithRef]


@dataclass(frozen=True)
class Solution:
    result: Result
    policy: dict[State, dict[str, z3.ArithRef]] | None
    statistics: Statistics | None


def _validate(j: Job):
    if j.start not in j.actions:
        raise ValueError("missing start state")
    if type(j.max_atoms) is not int or j.max_atoms < 0:
        raise ValueError("invalid atom budget")
    if j.max_depth is not None and (type(j.max_depth) is not int or j.max_depth < 0):
        raise ValueError("invalid depth budget")
    if not all(0 <= _f(t) <= 1 for t in (j.error_limit, j.fidelity_min)):
        raise ValueError("invalid quality budgets")
    if not j.ideal or any(_f(v) < 0 for v in j.ideal.values()) or sum(j.ideal.values()) != 1:
        raise ValueError("ideal terminal distribution must be normalized")
    for ops in j.actions.values():
        if len(set(op.name for op in ops)) != len(ops):
            raise ValueError("ambiguous action names at one state")
        for op in ops:
            if _f(op.duration) <= 0:
                raise ValueError("all actions require strictly positive expected duration")
            if any(type(t) is not int or t < 0 for t in (op.peak_atoms, op.depth)):
                raise ValueError("invalid resource transfer")
            for o in op.outcomes:
                if _f(o.probability) <= 0:
                    raise ValueError("declare exactly the positive support of each outcome law")
                if (o.next_state is None) == (o.output is None):
                    raise ValueError("each outcome has exactly one of successor or output")
                if o.next_state is not None and (o.next_state not in j.actions or o.failed):
                    raise ValueError("nonterminal outcomes require valid successor and no failure flag")
                if o.output is not None and o.output not in j.ideal:
                    raise ValueError("include even zero-ideal-mass output labels explicitly")
            if sum(_f(o.probability) for o in op.outcomes) != 1:
                raise ValueError("outcome law must have unit total mass")


def _start(j):
    return (j.start, None if j.max_depth is None else 0)


def _next(s: State, op: Action, o: Outcome) -> State:
    return o.next_state, None if s[1] is None else s[1] + op.depth


def resource_graph(j: Job) -> dict[State, dict[str, Action]]:
    """Prune actions that already violate hard limits, never their positive outcomes."""
    _validate(j)
    graph = {}
    todo = deque([_start(j)])
    while todo:
        s = todo.popleft()
        if s in graph:
            continue
        allowed = {op.name: op for op in j.actions[s[0]] if op.peak_atoms <= j.max_atoms
                   and (s[1] is None or s[1] + op.depth <= j.max_depth)}
        graph[s] = allowed
        for op in allowed.values():
            todo.extend(_next(s, op, o) for o in op.outcomes if o.next_state is not None)
    return graph


def _equations(j, graph):
    x = {(s, a): z3.Real(f"visits_{i}") for i, (s, a) in enumerate(
        (s, a) for s, ops in graph.items() for a in ops)}
    incoming = {s: [] for s in graph}
    delivered = {y: [] for y in j.ideal}
    failed = []
    for (s, a), val in x.items():
        op = graph[s][a]
        for o in op.outcomes:
            mass = _q(o.probability) * val
            if o.next_state is not None:
                incoming[_next(s, op, o)].append(mass)
            else:
                delivered[o.output].append(mass)
                if o.failed:
                    failed.append(mass)
    guards = [v >= 0 for v in x.values()]
    for s in graph:
        guards.append(_sum(x[s, a] for a in graph[s]) ==
                      int(s == _start(j)) + _sum(incoming[s]))
    output = {y: _sum(ts) for y, ts in delivered.items()}
    guards.extend((_sum(output.values()) == 1, _sum(failed) <= _q(j.error_limit)))
    aux = []
    if j.fidelity_min == 1:
        guards.extend(output[y] == _q(prob) for y, prob in j.ideal.items())
    elif j.fidelity_min > 0:
        positive = [(y, prob) for y, prob in j.ideal.items() if prob > 0]
        if len(positive) == 1:
            guards.append(output[positive[0][0]] >= _q(j.fidelity_min))
        elif len(j.ideal) == 2:
            # In two dimensions sqrt-F is the dot product of sqrt-probability
            # unit vectors in the first quadrant. Intersect the allowable angle
            # interval with that quadrant, and square its endpoints. Express
            # the resulting root comparisons as rational polynomials.
            y, q = positive[0]
            q, f = _f(q), _f(j.fidelity_min)
            center = q * f + (1 - q) * (1 - f)
            radius_squared = 4 * q * f * (1 - q) * (1 - f)
            delta = output[y] - _q(center)
            between = delta * delta <= _q(radius_squared)
            if q + f > 1:
                guards.append(z3.Or(delta >= 0, between))
            if q < f:
                guards.append(z3.Or(delta <= 0, between))
        else:
            for i, (y, prob) in enumerate(positive):
                root = z3.Real(f"sqrt_overlap_{i}")
                aux.append(root)
                guards.extend((root >= 0, root * root <= _q(prob) * output[y]))
            guards.append(_sum(aux) ** 2 >= _q(j.fidelity_min))
    time = _sum(val * _q(graph[s][a].duration) for (s, a), val in x.items())
    return x, Diagram("stopping_vqa", tuple(x.values()) + tuple(aux), z3.And(*guards),
                      time, (j.start, tuple(graph)))


def _gauss(matrix, right):
    """Exact elimination; a singular reachable system has no finite proper replay."""
    n = len(right)
    rows = [[z3.simplify(t) for t in row] + [right[i]] for i, row in enumerate(matrix)]
    for i in range(n):
        k = next((k for k in range(i, n) if _true(rows[k][i] != 0)), None)
        if k is None:
            raise ValueError("reachable closed recurrent class: improper policy")
        rows[i], rows[k] = rows[k], rows[i]
        pivot = rows[i][i]
        rows[i] = [z3.simplify(t / pivot) for t in rows[i]]
        for k in range(n):
            if k == i or _true(rows[k][i] == 0):
                continue
            factor = rows[k][i]
            rows[k] = [z3.simplify(t - factor * u) for t, u in zip(rows[k], rows[i])]
    return [row[-1] for row in rows]


def replay(j: Job, policy: dict[State, dict[str, z3.ArithRef]]) -> Statistics:
    """Independently solve visit balance on the actual reachable positive-support class."""
    g = resource_graph(j)
    reached = {}
    todo = deque([_start(j)])
    while todo:
        s = todo.popleft()
        if s in reached:
            continue
        reached[s] = len(reached)
        probs = policy.get(s, {})
        if not probs or not _true(_sum(probs.values()) == 1):
            raise ValueError("missing or unnormalized action law at reachable state")
        for a, p in probs.items():
            if a not in g[s] or not _true(p >= 0):
                raise ValueError("illegal action or probability")
            if _true(p > 0):
                op = g[s][a]
                todo.extend(_next(s, op, o) for o in op.outcomes if o.next_state is not None)
    n = len(reached)
    m = [[_q(int(i == k)) for k in range(n)] for i in range(n)]
    for s, i in reached.items():
        for a, p in policy[s].items():
            if _true(p == 0):
                continue
            op = g[s][a]
            for o in op.outcomes:
                if o.next_state is not None:
                    k = reached[_next(s, op, o)]
                    m[k][i] = m[k][i] - p * _q(o.probability)
    visits = dict(zip(reached, _gauss(m, [_q(int(s == _start(j))) for s in reached])))
    out = {y: [] for y in j.ideal}
    times, bad = [], []
    for s, count in visits.items():
        if not _true(count > 0):
            raise ValueError("nonpositive expected visits on reachable class")
        for a, p in policy[s].items():
            op = g[s][a]
            val = count * p
            times.append(val * _q(op.duration))
            for o in op.outcomes:
                if o.output is not None:
                    mass = val * _q(o.probability)
                    out[o.output].append(mass)
                    if o.failed:
                        bad.append(mass)
    law = {k: _sum(ts) for k, ts in out.items()}
    fidelity = z3.simplify(_sum(z3.Sqrt(_q(j.ideal[k]) * prob)
                                for k, prob in law.items()) ** 2)
    failure = _sum(bad)
    if not _true(z3.And(_sum(law.values()) == 1, failure <= _q(j.error_limit),
                        fidelity >= _q(j.fidelity_min))):
        raise ValueError("policy violates termination or output contract")
    return Statistics(_sum(times), failure, law, fidelity, visits)


def _extract(graph, x, raw_result):
    values = dict(raw_result.assignment)
    policy = {}
    for s, ops in graph.items():
        v = _sum(values[x[s, a]] for a in ops)
        if _true(v > 0):
            policy[s] = {a: z3.simplify(values[x[s, a]] / v) for a in ops
                         if _true(values[x[s, a]] > 0)}
    return policy


def compile_stopping_job(j: Job) -> Solution:
    """Minimum expected physical time amongst all proper allowed history policies.

    Expected finite occupations of any admissible policy obey these constraints.
    The Markovized finite flow induces the same reachable terminal occupations.
    Any extra circulations are rooted outside that class and have strictly
    positive cost; at an optimal flow they vanish. A feasible time bound bounds
    every occupation by that time divided by minimum action cost. Closed finite
    constraints and bounded fidelity auxiliaries thus attain any finite optimum.
    """
    graph = resource_graph(j)
    x, diagram = _equations(j, graph)
    result = solve([diagram])
    if result.status == "infeasible":
        return Solution(result, None, None)
    if result.status != "optimal":
        raise ArithmeticError("closed finite-flow problem with positive time must attain its optimum")
    policy = _extract(graph, x, result)
    stats = replay(j, policy)
    if not _true(stats.expected_time == result.time):
        raise ArithmeticError("independent optimum-time replay differs from synthesis")
    return Solution(result, {s: policy[s] for s in stats.visits}, stats)
