"""Exact synthesis over finitely many typed, lowered diagrams with real controls.

Input formulas are over the ordered real field. A diagram's feasible formula
comes from its task, hardware, external semantics and resource monitors. The
solver compares full-job costs. It cannot establish completeness of an
arbitrary hardware catalog. The program object is carried through unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import z3


@dataclass(frozen=True)
class Diagram:
    name: str
    variables: tuple
    feasible: z3.BoolRef
    time: z3.ArithRef
    program: Any


@dataclass(frozen=True)
class Result:
    status: str
    time: z3.ArithRef | None = None
    diagram: Diagram | None = None
    assignment: tuple = ()

    def exact_values(self):
        return {str(var): val.sexpr() for var, val in self.assignment}


class Undecided(RuntimeError):
    """Backend returned unknown or a non-algebraic witness; no optimum certified."""


def _some(variables, formula):
    return z3.Exists(variables, formula) if variables else formula


def _check(formula):
    # This tactic is for polynomial arithmetic over the reals, including
    # quantifier alternation. A generic SMT solver is not a substitute here.
    formula = z3.simplify(formula)
    pending, seen = [formula], set()
    quantified = False
    algebraics = []
    while pending:
        term = pending.pop()
        if term.get_id() in seen:
            continue
        seen.add(term.get_id())
        if z3.is_quantifier(term):
            quantified = True
            pending.append(term.body())
        elif z3.is_algebraic_value(term):
            algebraics.append(term)
        else:
            pending.extend(term.children())
    if quantified and algebraics:
        # NLQSAT accepts rational polynomials, but may reject a root object
        # inside quantifiers. Isolate each constant first, outside all binders.
        replacements, constraints = [], []
        purify = z3.Tactic("purify-arith")
        for value in algebraics:
            root = z3.FreshReal("isolated_root")
            definition = purify(root == value).as_expr()
            replacements.append((value, root))
            constraints.append(definition)
        formula = z3.And(z3.substitute(formula, *replacements), *constraints)
    solver = z3.Tactic("nlqsat").solver() if quantified else z3.SolverFor("QF_NRA")
    solver.add(formula)
    status = solver.check()
    if status == z3.unknown:
        raise Undecided(solver.reason_unknown())
    return solver.model() if status == z3.sat else None


def _number(model, term):
    value = z3.simplify(model.eval(term, model_completion=True))
    if not (z3.is_algebraic_value(value) or z3.is_rational_value(value)
            or z3.is_int_value(value)):
        raise Undecided(f"expected exact real value for {term}: {value}")
    return value


def _validate(diagram):
    allowed_ops = {
        z3.Z3_OP_TRUE, z3.Z3_OP_FALSE, z3.Z3_OP_AND, z3.Z3_OP_OR, z3.Z3_OP_NOT,
        z3.Z3_OP_IMPLIES, z3.Z3_OP_XOR, z3.Z3_OP_EQ, z3.Z3_OP_DISTINCT,
        z3.Z3_OP_LT, z3.Z3_OP_LE, z3.Z3_OP_GT, z3.Z3_OP_GE, z3.Z3_OP_ITE,
        z3.Z3_OP_ADD, z3.Z3_OP_SUB, z3.Z3_OP_MUL, z3.Z3_OP_UMINUS, z3.Z3_OP_TO_REAL,
    }
    if not z3.is_bool(diagram.feasible) or not z3.is_real(diagram.time):
        raise ValueError("expected a Boolean feasibility formula and a real time term")
    symbols = set(diagram.variables)
    if len(symbols) != len(diagram.variables) or any(
        not z3.is_real(v) or not z3.is_const(v)
        or v.decl().kind() != z3.Z3_OP_UNINTERPRETED for v in symbols
    ):
        raise ValueError("variables must be distinct free real constants")
    pending = [(diagram.time, 0), (diagram.feasible, 0)]
    seen = set()
    while pending:
        term, depth = pending.pop()
        key = (term.get_id(), depth)
        if key in seen:
            continue
        seen.add(key)
        if z3.is_quantifier(term):
            if term.is_lambda() or any(term.var_sort(i).kind() != z3.Z3_REAL_SORT
                                       for i in range(term.num_vars())):
                raise ValueError("only first-order quantification over real numbers")
            pending.append((term.body(), depth + term.num_vars()))
            continue
        if z3.is_var(term):
            if z3.get_var_index(term) >= depth or not z3.is_real(term):
                raise ValueError("unbound or non-real variable in formula")
            continue
        if (z3.is_rational_value(term) or z3.is_int_value(term)
                or z3.is_algebraic_value(term)):
            continue
        op = term.decl().kind()
        children = term.children()
        if op == z3.Z3_OP_UNINTERPRETED and term in symbols:
            continue
        if op == z3.Z3_OP_DIV:
            denominator = z3.simplify(children[1])
            if not z3.is_rational_value(denominator) or denominator.numerator_as_long() == 0:
                raise ValueError("clear variable denominators with an explicit sign guard")
        elif op == z3.Z3_OP_POWER:
            exponent = z3.simplify(children[1])
            if (not z3.is_rational_value(exponent) or exponent.denominator_as_long() != 1
                    or exponent.numerator_as_long() < 1):
                raise ValueError("use strictly positive integer powers; write constant 1 for x**0")
        elif op not in allowed_ops:
            raise ValueError(f"undeclared symbol or non-real-field operation: {term}")
        pending.extend((child, depth) for child in children)


def _language(diagrams, predicate):
    # All local variables, including witnesses for algebraic square roots,
    # are bound separately per policy. No private information is a public choice.
    return z3.Or([_some(d.variables, z3.And(d.feasible, predicate(d.time)))
                  for d in diagrams])


def verify(diagrams: Iterable[Diagram], result: Result) -> bool:
    catalog = tuple(diagrams)
    for d in catalog:
        _validate(d)
    if result.status == "infeasible":
        return (result.time is None and result.diagram is None and not result.assignment
                and all(_check(d.feasible) is None for d in catalog))
    if not (z3.is_algebraic_value(result.time) or z3.is_rational_value(result.time)):
        return False
    if any(_check(z3.And(d.feasible, d.time < result.time))
                                  is not None for d in catalog):
        return False
    if result.status == "infimum":
        if result.diagram is not None or result.assignment:
            return False
        delta = z3.FreshReal("tolerance")
        reached = _language(catalog, lambda t: t == result.time)
        can_approach = z3.ForAll(delta, z3.Implies(delta > 0,
            _language(catalog, lambda t: t < result.time + delta)))
        return _check(reached) is None and _check(z3.Not(can_approach)) is None
    if result.status != "optimal" or result.diagram is None:
        return False
    # Re-check against the catalog's object, not just a matching policy name.
    d = next((item for item in catalog if item is result.diagram), None)
    if d is None or tuple(v.get_id() for v, _ in result.assignment) != tuple(
        v.get_id() for v in d.variables
    ):
        return False
    if any(not (z3.is_algebraic_value(val) or z3.is_rational_value(val))
           for _, val in result.assignment):
        return False
    exact = z3.And(d.feasible, d.time == result.time)
    if result.assignment:
        exact = z3.substitute(exact, *result.assignment)
    return _check(z3.Not(exact)) is None


def solve(diagrams: Iterable[Diagram]) -> Result:
    catalog = tuple(diagrams)
    if len({d.name for d in catalog}) != len(catalog):
        raise ValueError("each diagram needs a unique name")
    for d in catalog:
        _validate(d)
    witnesses = [(d, _check(d.feasible)) for d in catalog]
    feasible = [(d, model) for d, model in witnesses if model is not None]
    if not feasible:
        return Result("infeasible")
    if any(_check(z3.And(d.feasible, d.time < 0)) is not None for d, _ in feasible):
        raise ValueError("a feasible diagram has negative execution time")
    # Try a feasible upper bound and a *separate exact lower-bound query*.
    # For constant-cost diagrams this avoids quantified optimization entirely.
    incumbent, model = feasible[0]
    for other, other_model in feasible[1:]:
        comparison = _number(other_model, other.time) < _number(model, incumbent.time)
        if z3.is_true(z3.simplify(comparison)):
            incumbent, model = other, other_model
    sample = Result("optimal", _number(model, incumbent.time), incumbent,
                    tuple((v, _number(model, v)) for v in incumbent.variables))
    if verify(catalog, sample):
        return sample
    t = z3.FreshReal("job_time")
    attainable = _language(catalog, lambda duration: duration == t)
    lower = z3.Not(_language(catalog, lambda duration: duration < t))
    minimum = _check(z3.And(attainable, lower))
    if minimum is None:
        gap = z3.FreshReal("gap")
        near = z3.ForAll(gap, z3.Implies(gap > 0,
            _language(catalog, lambda duration: duration < t + gap)))
        boundary = _check(z3.And(lower, near))
        if boundary is None:
            raise Undecided("nonempty nonnegative times must have an infimum")
        result = Result("infimum", _number(boundary, t))
    else:
        optimum = _number(minimum, t)
        for d in catalog:
            witness = _check(z3.And(d.feasible, d.time == optimum))
            if witness is not None:
                values = tuple((v, _number(witness, v)) for v in d.variables)
                result = Result("optimal", optimum, d, values)
                break
        else:
            raise Undecided("optimizer returned a value without an attainable witness")
    if not verify(catalog, result):
        raise Undecided("cost or witness replay failed")
    return result
