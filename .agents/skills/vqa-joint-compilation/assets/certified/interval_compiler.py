"""Certified constrained minimization over a finite catalog of compact rational boxes.

A deterministic trusted backend encloses the true complete-job time T and every
residual g_i on the *entire* queried box. Every budget is g_i <= 0, including
at a witness. No tolerance is added to a budget. Precision is a nonnegative
integer: the backend may tighten its arithmetic when this number increases.
None means that no enclosure is available. Floating-point endpoints are rejected.

Longest-side bisection in simultaneous rounds is fair over surviving boxes.
Their centers are dense in regions that keep being subdivided. All generated
points and declared seeds are retried at successively higher precision.
The default call stops at its round cap and can return an undecided certificate.
With max_rounds=None search continues until it certifies the gap or infeasibility.

For eventual positive-delta termination it suffices that T and g_i are continuous,
box enclosures converge to the true point values along every sequence of shrinking
boxes at increasing precision, and strictly feasible rationally samplable points
approach the optimum in time. Alternatively, appropriately near-optimal seeds
with certifiable nonpositive residuals and convergent time bounds suffice.
These conditions concern *this catalog* and trusted physical cost/constraint
oracles. An isolated feasible point may need an exact symbolic witness. At a
rational singleton the box oracle also acts as the point verifier. An optional
named-point callback certifies a particular executable nonrational point, its
location, and costs for that same point. Noncomputable or unknown physics is not
certified by repeated numerical evaluations.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import count
from typing import Callable, Iterable


def _q(x):
    if type(x) is int or isinstance(x, Fraction):
        return Fraction(x)
    raise TypeError("expected an exact rational (int or Fraction)")


@dataclass(frozen=True)
class Interval:
    lo: Fraction
    hi: Fraction

    def __post_init__(self):
        object.__setattr__(self, "lo", _q(self.lo))
        object.__setattr__(self, "hi", _q(self.hi))
        if self.lo > self.hi:
            raise ValueError("reversed interval")

    def contains(self, other: Interval):
        return self.lo <= other.lo and other.hi <= self.hi


Box = tuple[Interval, ...]


@dataclass(frozen=True)
class Evaluation:
    time: Interval
    constraints: tuple[Interval, ...]


@dataclass(frozen=True)
class Located:
    box: Box
    values: Evaluation


@dataclass(frozen=True)
class Domain:
    name: str
    box: Box
    constraints: int
    enclose: Callable[[Box, int], Evaluation | None]
    named_point: Callable[[str, int], Located | None] | None = None


@dataclass(frozen=True)
class Candidate:
    domain: str
    point: tuple[Fraction, ...] | str

    def __post_init__(self):
        if not isinstance(self.point, str):
            object.__setattr__(self, "point", tuple(_q(x) for x in self.point))


@dataclass(frozen=True)
class Witness:
    candidate: Candidate
    precision: int
    values: Evaluation


@dataclass(frozen=True)
class Leaf:
    domain: str
    path: tuple[int, ...]   # deterministic longest-side bisection, left=0, right=1
    precision: int
    values: Evaluation | None


@dataclass(frozen=True)
class Certificate:
    status: str
    lower: Fraction | None
    upper: Fraction | None
    witness: Witness | None
    cover: tuple[Leaf, ...]


def _catalog(domains):
    models = {}
    for d in domains:
        if (not isinstance(d.name, str) or not d.name or d.name in models
                or not isinstance(d.box, tuple) or not all(isinstance(i, Interval) for i in d.box)
                or type(d.constraints) is not int or d.constraints < 0):
            raise ValueError("malformed or duplicate domain")
        models[d.name] = d
    return models


def _check_values(d, values):
    if values is None:
        return None
    if (not isinstance(values, Evaluation) or not isinstance(values.time, Interval)
            or not isinstance(values.constraints, tuple) or len(values.constraints) != d.constraints
            or not all(isinstance(v, Interval) for v in values.constraints)):
        raise ValueError("malformed time or constraint enclosure")
    return values


def _contained(outer, inner):
    return len(outer) == len(inner) and all(x.contains(y) for x, y in zip(outer, inner))


def _point(d, candidate, precision):
    if isinstance(candidate.point, str):
        if d.named_point is None:
            raise ValueError("no interpreter for named executable points")
        located = d.named_point(candidate.point, precision)
        if located is None:
            return None
        if not _contained(d.box, located.box):
            return None  # this precision did not prove domain membership
        return _check_values(d, located.values)
    box = tuple(Interval(x, x) for x in candidate.point)
    if not _contained(d.box, box):
        raise ValueError("rational candidate lies outside its domain")
    return _check_values(d, d.enclose(box, precision))


def _split(box):
    if not box:
        return None
    axis = max(range(len(box)), key=lambda i: box[i].hi - box[i].lo)
    a, b = box[axis].lo, box[axis].hi
    if a == b:
        return None
    mid = (a + b) / 2
    return tuple(box[:axis] + (half,) + box[axis+1:]
                 for half in (Interval(a, mid), Interval(mid, b)))


def _excluded(values):
    return values is not None and any(g.lo > 0 for g in values.constraints)


def _witness_ok(values):
    return values is not None and all(g.hi <= 0 for g in values.constraints)


def _limits(cover):
    possible = [leaf.values for leaf in cover if not _excluded(leaf.values)]
    if not possible:
        return True, None
    if any(v is None for v in possible):
        return False, None
    return False, min(v.time.lo for v in possible)


def _result(cover, witness, delta):
    empty, lower = _limits(cover)
    upper = witness.values.time.hi if witness else None
    if upper is not None and (empty or lower is not None and lower > upper):
        raise ValueError("enclosures disagree about the feasible witness")
    if empty:
        status = "infeasible"
    elif lower is not None and upper is not None and upper - lower <= delta:
        status = "delta_optimal"
    else:
        status = "undecided"
    return Certificate(status, lower, upper, witness, tuple(cover))


def solve(domains: Iterable[Domain], delta, max_rounds: int | None = 12,
          seeds: Iterable[Candidate] = ()) -> Certificate:
    """Round zero checks the roots; max_rounds counts subsequent bisection rounds.

    None keeps one search alive until it certifies a result. Termination for
    positive delta needs the documented enclosure and feasibility conditions;
    for zero delta an exact certificate may never exist at finite precision.
    """
    models = _catalog(domains)
    delta = _q(delta)
    if delta < 0 or (max_rounds is not None and (type(max_rounds) is not int or max_rounds < 0)):
        raise ValueError("expected nonnegative tolerance and round budget")
    active = [(d.name, (), d.box) for d in models.values()]
    points = dict.fromkeys(seeds)
    if any(p.domain not in models for p in points):
        raise ValueError("seed has no matching domain")
    frozen, best_cover, best_lower, witness = [], None, None, None
    for precision in count():
        cells = []
        for name, path, box in active:
            d = models[name]
            values = _check_values(d, d.enclose(box, precision))
            cells.append((Leaf(name, path, precision, values), box))
            if not _excluded(values):
                center = tuple((x.lo + x.hi) / 2 for x in box)
                points.setdefault(Candidate(name, center), None)
        for point in points:
            values = _point(models[point.domain], point, precision)
            if _witness_ok(values) and (witness is None or values.time.hi < witness.values.time.hi):
                witness = Witness(point, precision, values)
        cover = (*frozen, *(leaf for leaf, _ in cells))
        result = _result(cover, witness, delta)
        if result.status != "undecided":
            return result
        # Even non-nested trustworthy oracle bounds yield monotone reported bounds.
        if best_cover is None or (result.lower is not None and
                                   (best_lower is None or result.lower > best_lower)):
            best_cover, best_lower = cover, result.lower
        result = _result(best_cover, witness, delta)
        if result.status != "undecided" or precision == max_rounds:
            return result
        active = []
        for leaf, box in cells:
            v = leaf.values
            if _excluded(v) or (v is not None and witness and v.time.lo >= witness.values.time.hi):
                frozen.append(leaf)
                continue
            halves = _split(box)
            if halves is None:
                active.append((leaf.domain, leaf.path, box))
            else:
                active.extend((leaf.domain, leaf.path+(bit,), half) for bit, half in enumerate(halves))


def _recheck_leaf(models, cover):
    leaves = {name: {} for name in models}
    for leaf in cover:
        if (leaf.domain not in leaves or not isinstance(leaf.path, tuple)
                or any(type(bit) is not int or bit not in (0, 1) for bit in leaf.path)
                or type(leaf.precision) is not int or leaf.precision < 0
                or leaf.path in leaves[leaf.domain]):
            return False
        leaves[leaf.domain][leaf.path] = leaf
    for name, records in leaves.items():
        d = models[name]
        ancestors = {path[:j] for path in records for j in range(len(path))}
        if records.keys() & ancestors:   # partition leaves must be prefix-free
            return False
        todo = [((), d.box)]
        while todo:
            path, box = todo.pop()
            if path in records:
                leaf = records[path]
                if leaf.values is not None:
                    seen = _check_values(d, leaf.values)
                    fresh = _check_values(d, d.enclose(box, leaf.precision))
                    if not _supported(seen, fresh):
                        return False
                continue
            if path not in ancestors:
                return False  # missing or uncovered half
            halves = _split(box)
            if halves is None:
                return False
            todo.extend((path+(i,), b) for i, b in enumerate(halves))
    return True


def _supported(claim, fresh):
    return (fresh is not None and claim.time.contains(fresh.time)
            and _contained(claim.constraints, fresh.constraints))


def verify(domains: Iterable[Domain], delta, certificate: Certificate) -> bool:
    """Independently replay all leaf bounds, exact point membership and budget checks."""
    try:
        models, delta = _catalog(domains), _q(delta)
        if delta < 0 or not _recheck_leaf(models, certificate.cover):
            return False
        w = certificate.witness
        if w is not None:
            if type(w.precision) is not int or w.precision < 0:
                return False
            d = models[w.candidate.domain]
            recorded = _check_values(d, w.values)
            fresh = _point(d, w.candidate, w.precision)
            if not _witness_ok(recorded) or not _supported(recorded, fresh):
                return False
        reconstructed = _result(certificate.cover, w, delta)
        return reconstructed == certificate
    except (ValueError, TypeError, KeyError, AttributeError):
        return False
