"""Finite typed rewrite system for a shared-angle Max-Cut QAOA kernel.

Expressions execute from left to right. A permutation maps old wire i to new
wire g[i], so a measured bit from i appears at g[i]. The final declared
controller sees the corrected, ordered record only. The native alphabet below
is an explicit gate/record backend contract, not a neutral-atom pulse library.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

Permutation = tuple[int, ...]


@dataclass(frozen=True)
class Model:
    n: int
    edges: tuple[tuple[int, int, Fraction], ...]  # undirected, a < b
    layers: tuple[str, ...]  # one shared gamma and beta per layer name
    feedback: str  # opaque fixed controller on canonical visible records
    pinned: tuple[int, ...] = ()

    def __post_init__(self):
        if self.n < 1 or not self.feedback or not all(self.layers):
            raise ValueError("positive wire count and nonempty operation names required")
        if any(not 0 <= v < self.n for v in self.pinned):
            raise ValueError("pinned port outside register")
        if len({(i, j) for i, j, _ in self.edges}) != len(self.edges):
            raise ValueError("use one weight per edge")
        for i, j, w in self.edges:
            if not 0 <= i < j < self.n or not isinstance(w, (int, Fraction)) or w == 0:
                raise ValueError("expected ordered, nonzero, exactly weighted edges")

    def check_permutation(self, g: Permutation) -> None:
        if sorted(g) != list(range(self.n)) or any(g[i] != i for i in self.pinned):
            raise ValueError("permutation does not preserve register and pinned ports")

    def preserves_layer(self, g: Permutation) -> bool:
        self.check_permutation(g)
        weights = {(i, j): w for i, j, w in self.edges}
        return all(weights.get(tuple(sorted((g[i], g[j])))) == w
                   for i, j, w in self.edges)


@dataclass(frozen=True)
class Op:
    kind: str
    arg: str | Permutation | None = None


Program = tuple[Op, ...]


def validate(model: Model, program: Program) -> None:
    """Q* -> ordered Z record -> issued feedback. No hidden outcomes."""
    port = "quantum"
    for op in program:
        if op.kind in ("Q", "C"):
            source = "quantum" if op.kind == "Q" else "record"
            if source != port or not isinstance(op.arg, tuple):
                raise ValueError("permutation port/type mismatch")
            model.check_permutation(op.arg)
        elif op.kind == "Layer" and port == "quantum" and op.arg in model.layers:
            continue
        elif op.kind == "M" and port == "quantum" and op.arg is None:
            port = "record"
        elif op.kind == "Fixed" and port == "record" and op.arg == model.feedback:
            port = "delivered"
        else:
            raise ValueError("illegal operation or feedback interface")
    if port != "delivered":
        raise ValueError("expected one complete readout and visible feedback")


@dataclass(frozen=True)
class Step:
    rule: str
    index: int


def rewrite(model: Model, program: Program, step: Step) -> Program:
    """Check every local side condition; return an equally typed expression."""
    i = step.index
    if not 0 <= i < len(program) - 1:
        raise ValueError("invalid rewrite position")
    left, right = program[i:i + 2]
    pair = left.kind, right.kind
    if ((step.rule == "QL->LQ" and pair == ("Q", "Layer"))
            or (step.rule == "LQ->QL" and pair == ("Layer", "Q"))):
        g = left.arg if left.kind == "Q" else right.arg
        if not model.preserves_layer(g):
            raise ValueError("weighted graph breaks layer symmetry")
        replacement = (right, left)
    elif step.rule == "QM->MC" and pair == ("Q", "M"):
        replacement = (right, Op("C", left.arg))
    elif step.rule == "MC->QM" and pair == ("M", "C"):
        replacement = (Op("Q", right.arg), left)
    else:
        raise ValueError("rule does not match expression")
    result = program[:i] + replacement + program[i + 2:]
    validate(model, result)
    return result


def replay_trace(model: Model, seed: Program, trace: Iterable[Step]) -> Program:
    """Verify a submitted proof step by step without trusting search or costs."""
    validate(model, seed)
    program = seed
    for step in trace:
        program = rewrite(model, program, step)
    return program


def relabel(record: tuple, g: Permutation) -> tuple:
    if sorted(g) != list(range(len(record))):
        raise ValueError("wrong record permutation")
    result = list(record)
    for i, j in enumerate(g):
        result[j] = record[i]
    return tuple(result)


def native_word(model: Model, program: Program) -> tuple[tuple, ...]:
    """Logical gate-level backend: edge RZZ, RX, neighbour SWAP, Z readout.

    RZZ(t)=exp(-it ZZ/2). Shared logical angles remain symbolic. A policy
    back end must lower these operations to legal fault-tolerant controls
    and evaluate duration/noise; ideal-equivalence proofs do not price them.
    """
    validate(model, program)
    code = []
    for op in program:
        if op.kind == "Q":
            current = list(range(model.n))
            target = relabel(tuple(current), op.arg)
            for j, needed in enumerate(target):
                k = current.index(needed)
                while k > j:
                    code.append(("SWAP", k - 1, k))
                    current[k - 1], current[k] = current[k], current[k - 1]
                    k -= 1
        elif op.kind == "Layer":
            # exp(-i gamma*C) for C=sum w*(I-ZZ)/2, up to global phase.
            code.extend(("RZZ", i, j, op.arg, "gamma", -w)
                        for i, j, w in model.edges)
            code.extend(("RX", i, op.arg, "beta", 2) for i in range(model.n))
        elif op.kind == "M":
            code.append(("MEASURE_Z_ALL",))
        elif op.kind == "C":
            code.append(("RELABEL_RECORD", op.arg))
        else:
            code.append(("VISIBLE_FEEDBACK", op.arg))
    return tuple(code)


@dataclass(frozen=True)
class Candidate:
    program: Program
    trace: tuple[Step, ...]
    native: tuple[tuple, ...]


def saturate(model: Model, seed: Program,
             max_steps: int | None = None) -> tuple[Candidate, ...]:
    """BFS of all generated words; None exhausts this finite rewrite orbit."""
    validate(model, seed)
    if max_steps is not None and max_steps < 0:
        raise ValueError("rewrite bound must be nonnegative")
    rules = {("Q", "Layer"): "QL->LQ", ("Layer", "Q"): "LQ->QL",
             ("Q", "M"): "QM->MC", ("M", "C"): "MC->QM"}
    queue = deque([(seed, ())])
    visited = {seed}
    result = []
    while queue:
        word, trace = queue.popleft()
        result.append(Candidate(word, trace, native_word(model, word)))
        if max_steps is not None and len(trace) >= max_steps:
            continue
        for i, (left, right) in enumerate(zip(word, word[1:])):
            rule = rules.get((left.kind, right.kind))
            if rule is None:
                continue
            if rule in ("QL->LQ", "LQ->QL"):
                permutation = left.arg if left.kind == "Q" else right.arg
                if not model.preserves_layer(permutation):
                    continue
            step = Step(rule, i)
            other = rewrite(model, word, step)
            if other not in visited:
                visited.add(other)
                queue.append((other, trace + (step,)))
    return tuple(result)


def example_kernel_catalogue() -> tuple[Model, Program, tuple[Candidate, ...]]:
    """Three-vertex equal-weight cycle; incoming quantum ports can be arbitrary.

    An input wire cycle must be respected by all implementations. Moving it
    through the shared-angle layer allows an actual set of SWAPs or a record
    reindexing at readout. Feedback receives identical canonical bit strings.
    """
    model = Model(3, ((0, 1, 1), (0, 2, 1), (1, 2, 1)), ("block1",), "train_once")
    seed = (Op("Q", (1, 2, 0)), Op("Layer", "block1"), Op("M"),
            Op("Fixed", model.feedback))
    return model, seed, saturate(model, seed)
