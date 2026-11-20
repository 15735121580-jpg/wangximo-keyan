"""Exact finite-support synthesis on a supplied, sufficient public interface.

States are finite; each action at s is a vector of real controls. The backend
supplies its semialgebraic validity guard, nonterminal transition probabilities,
k terminal moment increments and mean (possibly mixture-reweighted) duration.
In a joint ideal/noisy model the same public state and policy apply to both.
Guards must encode every pathwise atom/depth constraint or use an augmented
public resource monitor. Noise, quantum semantics and interface sufficiency
must be established for the backend. No assumption of a fixed retry limit.

Covers randomized policies with finite expected total visits under the joint
reference law. Continuous independent classical randomization on the extracted
finite support must be permitted. A standalone certificate proves optimality
for this input model, not for unmodeled physical controls.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

import z3

from cyclic_compiler import _gauss
from optimal_compiler import Diagram, Result, solve

Expr = z3.ArithRef
Number = Expr | Fraction | int
Guard = z3.BoolRef | bool


def _real(x: Number) -> Expr:
    if z3.is_real(x):
        return x
    if isinstance(x, Fraction) or type(x) is int:
        return z3.RealVal(str(x))
    raise TypeError("supply exact real expressions, integers, or fractions")


def _bool(x: Guard) -> z3.BoolRef:
    if type(x) is bool:
        return z3.BoolVal(x)
    if z3.is_bool(x):
        return x
    raise TypeError("expected a Boolean predicate")


def _sum(xs):
    return z3.simplify(sum(xs, z3.RealVal(0)))


def _holds(ground: z3.BoolRef) -> bool:
    value = z3.simplify(ground)
    if z3.is_true(value) or z3.is_false(value):
        return z3.is_true(value)
    # Quantified semialgebraic terminal/validity tests may survive simplification.
    check = Diagram("closed-replay-predicate", (), value, z3.RealVal(0), None)
    return solve((check,)).status == "optimal"


@dataclass(frozen=True)
class Transfer:
    allowed: Guard
    next_prob: tuple[Number, ...]  # missing mass terminates
    end: tuple[Number, ...]  # unnormalized contributions, already weighted by terminal mass
    duration: Number  # per visit, no prepayment for future visits


@dataclass(frozen=True)
class Backend:
    start: int
    dimensions: tuple[int, ...]
    moments: int
    transfer: Callable[[int, tuple[Expr, ...]], Transfer]
    terminal_ok: Callable[[tuple[Expr, ...]], Guard]


@dataclass(frozen=True)
class Choice:
    probability: Expr
    control: tuple[Expr, ...]


@dataclass(frozen=True)
class Stats:
    time: Expr
    end: tuple[Expr, ...]
    visits: dict[int, Expr]


@dataclass(frozen=True)
class Solution:
    result: Result
    policy: dict[int, tuple[Choice, ...]] | None
    stats: Stats | None


def _check_backend(b: Backend):
    if type(b.start) is not int or not 0 <= b.start < len(b.dimensions):
        raise ValueError("initial public state outside interface")
    if any(type(p) is not int or p < 0 for p in b.dimensions):
        raise ValueError("invalid action-coordinate dimension")
    if type(b.moments) is not int or b.moments < 0:
        raise ValueError("invalid number of terminal moments")


def _transfer(b, s, u):
    k = b.transfer(s, u)
    if len(k.next_prob) != len(b.dimensions) or len(k.end) != b.moments:
        raise ValueError("incorrect transition or moment shape")
    prob = tuple(_real(v) for v in k.next_prob)
    stats, time = tuple(_real(v) for v in k.end), _real(k.duration)
    guard = z3.And(_bool(k.allowed), time >= 0, _sum(prob) <= 1,
                   *[v >= 0 for v in prob])
    return guard, prob, stats, time


def _support(b, dim):
    if dim == 0:
        return 1  # this state has only the empty control tuple
    if len(b.dimensions) == 1 and b.moments == 0:
        # For one state with no output constraints, any proper policy's
        # time is a termination-mass weighted mean of c(a)/(1-P(a)),
        # plus nonnegative cost from zero-termination actions. A pure action
        # therefore suffices for the optimum (and for approximating its infimum).
        return 1
    return len(b.dimensions) + b.moments + 2


def _formulate(b: Backend):
    _check_backend(b)
    n = len(b.dimensions)
    variables, guards, slots = [], [], []
    flows, state_mass, moments, times = [], [], [[] for _ in range(b.moments)], []
    for s, dim in enumerate(b.dimensions):
        row, masses, trans = [], [], [[] for _ in range(n)]
        for i in range(_support(b, dim)):
            w = z3.FreshReal(f"mass_{s}_{i}")
            u = tuple(z3.FreshReal(f"control_{s}_{i}_{j}") for j in range(dim))
            valid, prob, end, time = _transfer(b, s, u)
            variables.extend((w, *u))
            guards.extend((w >= 0, z3.Implies(w > 0, valid)))
            for terms, value in zip(trans, prob):
                terms.append(w * value)
            for terms, value in zip(moments, end):
                terms.append(w * value)
            times.append(w * time)
            masses.append(w)
            row.append((w, u))
        slots.append(row)
        state_mass.append(_sum(masses))
        flows.append(tuple(_sum(terms) for terms in trans))
    for s in range(n):
        guards.append(state_mass[s] == int(s == b.start) + _sum(row[s] for row in flows))

    # Only mass reachable via positive flows is realizable. The path need
    # not repeat a vertex, so n-1 synchronous graph expansions suffice.
    reached = [z3.BoolVal(s == b.start) for s in range(n)]
    for _ in range(n - 1):
        reached = [z3.Or(reached[y], *[z3.And(reached[s], flows[s][y] > 0)
                                      for s in range(n)]) for y in range(n)]
    guards.extend(z3.Implies(x > 0, r) for x, r in zip(state_mass, reached))
    guards.append(_bool(b.terminal_ok(tuple(_sum(terms) for terms in moments))))
    return Diagram("continuous-stop", tuple(variables), z3.And(*guards),
                   _sum(times), slots)


def replay(b: Backend, policy: dict[int, tuple[Choice, ...]]) -> Stats:
    """Compute expected visits afresh for the actual positive-support chain."""
    _check_backend(b)
    n = len(b.dimensions)
    todo, rows = [b.start], {}
    while todo:
        s = todo.pop()
        if s in rows:
            continue
        if not policy.get(s):
            raise ValueError("no action law at reachable state")
        entries = []
        for atom in policy[s]:
            q, u = atom.probability, atom.control
            if (len(u) != b.dimensions[s]
                    or any(not (z3.is_rational_value(x) or z3.is_algebraic_value(x))
                           for x in (q, *u))):
                raise ValueError("expected closed exact real data for each action atom")
            valid, prob, end, t = _transfer(b, s, u)
            if not _holds(z3.And(q > 0, valid)):
                raise ValueError("invalid action or nonpositive extracted probability")
            entries.append((q, prob, end, t))
        if not _holds(_sum(q for q, *_ in entries) == 1):
            raise ValueError("action probabilities do not sum to one")
        p = tuple(_sum(q * prob[y] for q, prob, _, _ in entries) for y in range(n))
        rows[s] = (p, tuple(_sum(q * end[j] for q, _, end, _ in entries)
                            for j in range(b.moments)), _sum(q*t for q, _, _, t in entries))
        todo.extend(y for y, prob in enumerate(p) if _holds(prob > 0))
    order = tuple(rows)
    mat = [[z3.simplify(_real(int(i == j)) - rows[j][0][i]) for j in order]
           for i in order]
    visits = dict(zip(order, _gauss(mat, [_real(int(s == b.start)) for s in order])))
    if any(not _holds(v > 0) for v in visits.values()):
        raise ValueError("invalid occupancy on reachable states")
    stop = _sum(v * (1 - _sum(rows[s][0])) for s, v in visits.items())
    end = tuple(_sum(v*rows[s][1][j] for s,v in visits.items()) for j in range(b.moments))
    if not _holds(z3.And(stop == 1, _bool(b.terminal_ok(end)))):
        raise ValueError("terminal probability or contract failed on replay")
    return Stats(_sum(v*rows[s][2] for s,v in visits.items()), end, visits)


def compile_cycle(b: Backend, incumbent: dict[int, tuple[Choice, ...]] | None = None) -> Solution:
    """An optional feasible policy accelerates search; the full domain still bounds it."""
    diagram = _formulate(b)
    if incumbent is not None:
        witness = replay(b, incumbent)
        seed = Diagram("replayed-policy", (), z3.BoolVal(True), witness.time, incumbent)
        result = solve((seed, diagram))
        if result.diagram is seed:
            return Solution(result, {s: incumbent[s] for s in witness.visits}, witness)
    else:
        result = solve((diagram,))
    if result.status != "optimal":
        return Solution(result, None, None)
    assignment = dict(result.assignment)
    policy = {}
    for s, row in enumerate(diagram.program):
        active = [(assignment[w], tuple(assignment[x] for x in u)) for w,u in row
                  if _holds(assignment[w] > 0)]
        if active:
            total = _sum(w for w,_ in active)
            policy[s] = tuple(Choice(z3.simplify(w/total), u) for w,u in active)
    stats = replay(b, policy)
    if not _holds(stats.time == result.time):
        raise ArithmeticError("independent replay disagrees with certified optimum")
    return Solution(result, policy, stats)


def verify_solution(b: Backend, solution: Solution) -> bool:
    """Rebuild the search domain and compare an independently replayed policy."""
    from optimal_compiler import verify

    domain = _formulate(b)
    outcome = solution.result
    if outcome.status in ("infeasible", "infimum"):
        return (solution.policy is None and solution.stats is None
                and verify((domain,), outcome))
    if outcome.status != "optimal" or solution.policy is None or solution.stats is None:
        return False
    try:
        evaluated = replay(b, solution.policy)
    except (ValueError, TypeError):
        return False
    recorded = solution.stats
    if recorded.visits.keys() != evaluated.visits.keys() or len(recorded.end) != len(evaluated.end):
        return False
    equalities = [recorded.time == evaluated.time, outcome.time == evaluated.time]
    equalities.extend(x == y for x, y in zip(recorded.end, evaluated.end))
    equalities.extend(recorded.visits[s] == evaluated.visits[s] for s in evaluated.visits)
    if not _holds(z3.And(*equalities)):
        return False
    fixed = Diagram("verified-fixed-policy", (), z3.BoolVal(True), evaluated.time, None)
    return verify((fixed, domain), Result("optimal", evaluated.time, fixed))
