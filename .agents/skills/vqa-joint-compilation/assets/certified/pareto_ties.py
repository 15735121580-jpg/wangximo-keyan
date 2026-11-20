"""Choose a Pareto-undominated witness from the fastest feasible full programs.

Use 1-F for infidelity and declare all other costs in the minimizing direction.
Original feasibility includes all hard budgets and the complete task semantics.
On a nonempty compact candidate domain with continuous costs, all six successive
minima exist. More generally, we check attainment at every stage. If an
infimum is not attained, no six-cost lexicographic witness can exist for this
order. Incomparable Pareto-optimal vectors may be selected by different orders.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import z3

from optimal_compiler import Diagram, Result, solve, verify


DEFAULT = ("time", "infidelity", "logical_error", "depth", "gates", "peak_atoms")
ZERO = z3.RealVal(0)


def _real(x):
    if z3.is_real(x):
        return x
    if type(x) is int or isinstance(x, Fraction):
        return z3.RealVal(str(x))
    raise TypeError("each cost needs an exact real expression")


@dataclass(frozen=True)
class Endpoint:
    diagram: Diagram
    infidelity: object
    logical_error: object
    depth: object
    gates: object
    peak_atoms: object

    def costs(self, order):
        return tuple(_real(self.diagram.time if k == "time" else getattr(self, k))
                     for k in order)


@dataclass(frozen=True)
class LexResult:
    status: str
    priorities: tuple[str, ...]
    values: tuple  # attained prefix, in order; length six for optimal
    infimum: z3.ArithRef | None = None  # first nonattained coordinate, if present
    diagram: Diagram | None = None
    assignment: tuple = ()


def _check(catalog, order):
    if len(order) != 6 or order[0] != "time" or set(order) != set(DEFAULT):
        raise ValueError("time must be first, followed by the other five costs once each")
    if len({e.diagram.name for e in catalog}) != len(catalog):
        raise ValueError("duplicate diagram name")
    invalid = []
    for e in catalog:
        d = e.diagram
        c = e.costs(DEFAULT)
        out_of_range = z3.Or(*(q < 0 for q in c), c[1] > 1, c[2] > 1)
        invalid.append(Diagram(d.name, d.variables, z3.And(d.feasible, out_of_range), ZERO, d.program))
    if not verify(invalid, Result("infeasible")):
        raise ValueError("negative cost or quality probability outside the unit interval")


def _stage(catalog, order, values):
    diagrams = []
    for e in catalog:
        d = e.diagram
        costs = e.costs(order)
        guard = z3.And(d.feasible, *(c == v for c, v in zip(costs, values)))
        diagrams.append(Diagram(d.name, d.variables, guard, costs[len(values)], d.program))
    return diagrams


def compile_lex(catalog, priorities=DEFAULT):
    catalog, order = tuple(catalog), tuple(priorities)
    _check(catalog, order)
    values = ()
    for _ in order:
        best = solve(_stage(catalog, order, values))
        if best.status == "infeasible":
            if values:
                raise ArithmeticError("previously attained prefix lost its feasible extension")
            return LexResult("infeasible", order, ())
        if best.status == "infimum":
            return LexResult("infimum", order, values, infimum=best.time)
        values += (best.time,)
    diagram = next(e.diagram for e in catalog if e.diagram.name == best.diagram.name)
    return LexResult("optimal", order, values, diagram=diagram, assignment=best.assignment)


def _exact(x):
    return z3.is_algebraic_value(x) or z3.is_rational_value(x)


def verify_lex(catalog, answer: LexResult):
    """Eliminate every lexicographically better competitor and replay one witness."""
    catalog, order = tuple(catalog), tuple(answer.priorities)
    _check(catalog, order)
    vals = answer.values
    if not isinstance(vals, tuple) or len(vals) > 6 or not all(_exact(v) for v in vals):
        return False
    if answer.status == "infeasible":
        return (not vals and answer.infimum is None and answer.diagram is None and not answer.assignment
                and verify(_stage(catalog, order, ()), Result("infeasible")))
    better, tied = [], []
    for e in catalog:
        d = e.diagram
        prefix, improvements = [], []
        for cost, v in zip(e.costs(order), vals):
            improvements.append(z3.And(*prefix, cost < v))
            prefix.append(cost == v)
        better.append(Diagram(d.name, d.variables, z3.And(d.feasible, z3.Or(improvements)),
                              ZERO, d.program))
        if d is answer.diagram:
            tied.append(Diagram(d.name, d.variables, z3.And(d.feasible, *prefix), ZERO, d.program))
    if not verify(better, Result("infeasible")):
        return False
    if answer.status == "infimum":
        return (len(vals) < 6 and _exact(answer.infimum) and answer.diagram is None
                and not answer.assignment and verify(_stage(catalog, order, vals),
                                                      Result("infimum", answer.infimum)))
    return (answer.status == "optimal" and len(vals) == 6 and answer.infimum is None
            and len(tied) == 1 and verify(tied, Result("optimal", ZERO, tied[0], answer.assignment)))
