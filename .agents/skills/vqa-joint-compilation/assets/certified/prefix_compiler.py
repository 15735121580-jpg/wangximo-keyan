"""Exact incumbent-bounded prefix synthesis, with no user-specified round limit.

The trusted digital backend supplies the COMPLETE finite public action/outcome
menu and strictly positive actual probabilities; its positive support covers
all histories that the ideal semantics needs to check. A hard stop or exhausted
retry is a *terminal public output* with an explicit flag in the port. A dead
state without actions is never a delivered leaf. Private noise is not a choice.
The caller's qualifies(tree) validates the complete semantic/FT/quality/resource
contract. This solver witnesses time and coverage in that specified model.

Termination condition: on each hypothetical infinite observable-action path,
the sum of prefix_probability times action_mean_time eventually exceeds the
incumbent cost U. Finite branching then makes the explored tree finite (Koenig).
A sufficient hardware contract is at most B branching actions per path, every
positive conditional probability >=q>0, and every action duration >=tau>0.
The algorithm never truncates at an assumed horizon. Without a valid proof of
this condition the enumeration is a semialgorithm, and may run forever. In
particular a uniform lower bound q<1 on conditional probabilities alone does
not rule out geometric decay of actual prefix masses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from typing import Callable, Protocol

from policy_ir import Action, Decision, History, Leaf, PublicPort, Tree


def _q(x: int | Fraction) -> Fraction:
    if isinstance(x, bool) or not isinstance(x, (int, Fraction)):
        raise TypeError("actual probabilities and mean times must be exact rationals")
    return Fraction(x)


@dataclass(frozen=True)
class Law:
    mean_time: int | Fraction
    probabilities: tuple[int | Fraction, ...]  # in public outcome order


class DigitalBackend(Protocol):
    def is_terminal(self, port: PublicPort, history: History) -> bool: ...
    def actions(self, port: PublicPort, history: History) -> tuple[Action, ...]: ...
    def law(self, port: PublicPort, history: History, action: Action) -> Law: ...


class InvalidIncumbent(ValueError):
    """No valid upper bound was supplied; does not mean the job is infeasible."""


class InvalidGrowthCertificate(ValueError):
    pass


@dataclass(frozen=True)
class BranchingBound:
    branchings: int
    q: int | Fraction
    tau: int | Fraction

    def __post_init__(self):
        if type(self.branchings) is not int or self.branchings < 0:
            raise ValueError("B must be a nonnegative integer")
        if not 0 < _q(self.q) <= 1 or _q(self.tau) <= 0:
            raise ValueError("require 0<q<=1 and tau>0")


@dataclass
class _Node:
    port: PublicPort
    history: History
    mass: Fraction
    past_lb: Fraction
    branchings: int
    terminal: bool = False
    edges: list[tuple[Action, Fraction, tuple[int, ...]]] = field(default_factory=list)


@dataclass(frozen=True)
class PrefixResult:
    time: Fraction
    program: Tree
    incumbent_time: Fraction
    visited_prefixes: int
    pruned_actions: int
    pruned_combinations: int
    dead_prefixes: int
    root_candidates: int
    depth_bound_from_contract: int | None


def _action_laws(backend: DigitalBackend, port: PublicPort, history: History):
    seen = set()
    for a in backend.actions(port, history):
        if not isinstance(a, Action) or a.source_type != port.type_name or a.name in seen:
            raise ValueError("invalid/duplicate public action or incompatible source")
        seen.add(a.name)
        if (not a.outcomes or len({o.label for o in a.outcomes}) != len(a.outcomes)
                or any(not isinstance(o.target, PublicPort) for o in a.outcomes)):
            raise ValueError("declare all distinct typed public outcomes")
        law = backend.law(port, history, a)
        cost = _q(law.mean_time)
        probs = tuple(_q(v) for v in law.probabilities)
        if (cost <= 0 or len(probs) != len(a.outcomes) or any(p <= 0 for p in probs)
                or sum(probs) != 1):
            raise ValueError("require positive conditional time and normalized full positive support")
        yield a, cost, probs


def replay_time(backend: DigitalBackend, root: PublicPort, tree: Tree) -> Fraction:
    """Verify every explicit branch against the public IR and recompute expected cost."""
    stack = [(root, (), Fraction(1), tree)]
    total = Fraction(0)
    while stack:
        port, history, mass, node = stack.pop()
        if node.port != port or node.history != history:
            raise InvalidIncumbent("incorrect typed public port or history")
        if backend.is_terminal(port, history):
            if not isinstance(node, Leaf) or node.kind != "delivered":
                raise InvalidIncumbent("terminal port must explicitly deliver its outcome")
            continue
        if not isinstance(node, Decision):
            raise InvalidIncumbent("premature end: no silent failures/dead leaves")
        actions = list(_action_laws(backend, port, history))
        match = next((m for m in actions if m[0] == node.action), None)
        if match is None:
            raise InvalidIncumbent("program contains an illegal action")
        a, cost, probs = match
        if tuple(o for o, _ in node.branches) != a.outcomes:
            raise InvalidIncumbent("candidate omits, reorders or changes a public branch")
        total += mass * cost
        for (o, child), p in zip(node.branches, probs):
            stack.append((o.target, history + ((a.name, o.label),), mass * p, child))
    return total


def compile_with_incumbent(backend: DigitalBackend, root: PublicPort, incumbent: Tree | None,
                           qualifies: Callable[[Tree], bool],
                           growth: BranchingBound | None = None) -> PrefixResult:
    """Enumerate until exhausted. With no growth proof, termination is not guaranteed."""
    if incumbent is None:
        raise InvalidIncumbent("a concrete feasible upper-bound program is required")
    upper = replay_time(backend, root, incumbent)
    if qualifies(incumbent) is not True:
        raise InvalidIncumbent("incumbent violates the complete job contract")
    tree_nodes = [_Node(root, (), Fraction(1), Fraction(0), 0)]
    cuts = dead = 0
    # Discover public prefixes iteratively, without Python recursion-depth cutoffs.
    for node in tree_nodes:
        if backend.is_terminal(node.port, node.history):
            node.terminal = True
            continue
        laws = list(_action_laws(backend, node.port, node.history))
        if not laws:
            dead += 1
        for a, cost, probs in laws:
            k = node.branchings + int(len(probs) > 1)
            if growth is not None and (k > growth.branchings or min(probs) < growth.q
                                       or cost < growth.tau):
                raise InvalidGrowthCertificate("encountered a transition outside the growth contract")
            charge = node.mass * cost
            lower = node.past_lb + charge
            if lower > upper:
                cuts += 1
                continue
            child_ids = []
            for o, p in zip(a.outcomes, probs):
                child_ids.append(len(tree_nodes))
                tree_nodes.append(_Node(o.target, node.history + ((a.name, o.label),),
                                        node.mass * p, lower, k))
            node.edges.append((a, charge, tuple(child_ids)))
    # Bottom-up: each subtree cost is already multiplied by its root's true mass.
    # Ancestor charges enter once; disjoint child-subtree charges are summed.
    plans: list[list[tuple[Fraction, Tree]]] = [[] for _ in tree_nodes]
    combo_cuts = 0
    for i in reversed(range(len(tree_nodes))):
        node = tree_nodes[i]
        if node.terminal:
            plans[i].append((Fraction(0), Leaf(node.port, node.history, "delivered")))
            continue
        for a, charge, child_ids in node.edges:
            for children in product(*(plans[j] for j in child_ids)):
                cost = charge + sum(v for v, t in children)
                if node.past_lb + cost > upper:
                    combo_cuts += 1
                    continue
                branches = tuple((o, t) for o, (v, t) in zip(a.outcomes, children))
                plans[i].append((cost, Decision(node.port, node.history, a, branches)))
    best, best_tree = upper, incumbent
    saw_feasible = False
    for cost, candidate in plans[0]:
        if qualifies(candidate) is True:
            saw_feasible = True
            if cost < best:
                best, best_tree = cost, candidate
    if not saw_feasible:
        raise ArithmeticError("verified incumbent was lost by expansion or the contract changed")
    if best != replay_time(backend, root, best_tree):
        raise ArithmeticError("independent replay disagrees with extracted cost")
    cap = None if growth is None else int(upper // (_q(growth.q)**growth.branchings * _q(growth.tau)))
    return PrefixResult(best, best_tree, upper, len(tree_nodes), cuts, combo_cuts, dead,
                        len(plans[0]), cap)
