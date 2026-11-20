"""Exact finite public-history policy generation for a trusted hardware backend.

The controller API receives only PublicPort and public action/outcome history.
Choose the finite action alphabet, expose only physically readable events, and
provide total outcome instruments in the backend. Model/quantum/noise state is
used in backend.lower and must never be supplied to actions or is_terminal.
Completeness relative to hardware still requires a separate backend proof.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Callable, Iterator, Literal, Protocol

import z3

from optimal_compiler import Diagram, Result, solve


@dataclass(frozen=True)
class PublicPort:
    type_name: str
    fields: tuple = ()


@dataclass(frozen=True)
class Outcome:
    label: str
    target: PublicPort


@dataclass(frozen=True)
class Action:
    name: str
    source_type: str
    outcomes: tuple[Outcome, ...]


History = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Leaf:
    port: PublicPort
    history: History
    kind: Literal["delivered", "dead_horizon", "dead_end"]


@dataclass(frozen=True)
class Decision:
    port: PublicPort
    history: History
    action: Action
    branches: tuple[tuple[Outcome, Leaf | Decision], ...]


Tree = Leaf | Decision


class Backend(Protocol):
    def is_terminal(self, port: PublicPort, history: History) -> bool: ...
    def actions(self, port: PublicPort, history: History) -> tuple[Action, ...]: ...
    def lower(self, tree: Tree) -> Diagram | tuple[Diagram, ...]: ...


@dataclass(frozen=True)
class Unreachable:
    ideal_mass: z3.ArithRef
    noisy_mass: z3.ArithRef


@dataclass(frozen=True)
class RecordedFailure:
    """Obligations from the trusted semantics of the *same* lowered diagram.

    All quantities are unconditional branch masses. The credited masses must
    refer specifically to this leaf's explicit failure outcome in the output
    instruments. Backend must propagate its elapsed time and resources to its
    termination. Keeping these evidence formulas is not a hardware calibration.
    """
    permitted: z3.BoolRef
    ideal_mass: z3.ArithRef
    noisy_mass: z3.ArithRef
    credited_ideal: z3.ArithRef
    credited_noisy: z3.ArithRef


@dataclass(frozen=True)
class Witness:
    tree: Tree
    native_program: Any


def enumerate_policies(backend: Backend, root: PublicPort, horizon: int) -> Iterator[Tree]:
    """Enumerate every deterministic action choice for every visible branch."""
    if type(horizon) is not int or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer")

    def visit(port: PublicPort, history: History, remaining: int) -> Iterator[Tree]:
        if backend.is_terminal(port, history):
            yield Leaf(port, history, "delivered")
            return
        if remaining == 0:
            yield Leaf(port, history, "dead_horizon")
            return
        choices = tuple(backend.actions(port, history))
        if not choices:
            yield Leaf(port, history, "dead_end")
            return
        names = set()
        for action in choices:
            if not isinstance(action, Action) or action.source_type != port.type_name:
                raise ValueError("action incompatible with public source port")
            if action.name in names:
                raise ValueError("duplicate public action name at one history")
            names.add(action.name)
            outcomes = action.outcomes
            if not outcomes or len({o.label for o in outcomes}) != len(outcomes):
                raise ValueError("every action needs nonempty distinct outcome labels")
            if any(not isinstance(o.target, PublicPort) for o in outcomes):
                raise ValueError("outcomes require typed public target ports")
            pools = (visit(o.target, history + ((action.name, o.label),), remaining - 1)
                     for o in outcomes)
            for subtrees in product(*pools):
                yield Decision(port, history, action, tuple(zip(outcomes, subtrees)))

    return visit(root, (), horizon)


def dead_leaves(tree: Tree) -> Iterator[Leaf]:
    if isinstance(tree, Leaf):
        if tree.kind != "delivered":
            yield tree
        return
    for _, child in tree.branches:
        yield from dead_leaves(child)


def _dead_formula(record: Unreachable | RecordedFailure) -> z3.BoolRef:
    if isinstance(record, Unreachable):
        return z3.And(record.ideal_mass == 0, record.noisy_mass == 0)
    if isinstance(record, RecordedFailure):
        return z3.And(record.permitted, record.ideal_mass == record.credited_ideal,
                      record.noisy_mass == record.credited_noisy,
                      record.ideal_mass >= 0, record.noisy_mass >= 0)
    raise ValueError("each dead leaf requires explicit zero-mass or failure accounting")


DeadResolver = Callable[[Tree, Leaf, Diagram], Unreachable | RecordedFailure]


def compile_diagrams(backend: Backend, root: PublicPort, horizon: int,
                     dead_resolver: DeadResolver | None = None) -> list[Diagram]:
    """Lower every public strategy. An empty/failed lowering is an error.

    If a topology has no admissible implementations, lower to a Diagram with
    its proven infeasibility formula, rather than returning an empty catalog.
    A dead leaf is retained unless zero mass in both models is verified, or
    its termination is explicitly permitted and charged in both semantics.
    """
    catalog = []
    for i, tree in enumerate(enumerate_policies(backend, root, horizon)):
        obligations = tuple(dead_leaves(tree))
        if obligations and dead_resolver is None:
            raise ValueError("unresolved dead leaf: provide an explicit dead_resolver")
        emitted = backend.lower(tree)
        ds = (emitted,) if isinstance(emitted, Diagram) else emitted
        if not isinstance(ds, tuple) or not ds or any(not isinstance(d, Diagram) for d in ds):
            raise ValueError("lower must return a Diagram or a nonempty tuple of Diagrams")
        for j, d in enumerate(ds):
            extra = [_dead_formula(dead_resolver(tree, leaf, d)) for leaf in obligations]
            catalog.append(replace(d, name=f"{i}.{j}/{d.name}",
                                   feasible=z3.And(d.feasible, *extra),
                                   program=Witness(tree, d.program)))
    if not catalog:
        raise ValueError("policy generation unexpectedly produced an empty catalog")
    return catalog


def synthesize(backend: Backend, root: PublicPort, horizon: int,
               dead_resolver: DeadResolver | None = None) -> Result:
    return solve(compile_diagrams(backend, root, horizon, dead_resolver))
