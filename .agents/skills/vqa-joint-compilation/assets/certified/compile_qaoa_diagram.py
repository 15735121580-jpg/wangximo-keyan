"""Compile one typed, fixed-feedback QAOA kernel in a serial native interface.

The input prices describe a model, not a device calibration. Each issued
primitive i executes in the declared time, even on a failure. It independently
sets a persistent erasure flag with probability r_i. On no flag the ideal
program is exact. On a flag the verifier reports an orthogonal failure symbol.
Thus for the complete flagged output, p_fail = 1 - prod_i(1-r_i), and squared
state fidelity to the ideal output is prod_i(1-r_i), for every fixed input.
The physical flag cannot choose rewrites or alter the fixed outer controller.
A general correlated-noise or overlapping schedule needs a new evaluator.

The example seed has 3 shared-angle RZZ gates, 3 RX gates, Z readout, and
fixed feedback. Let their total duration be T0. The three costs are T0+2*u,
T0+2*v, T0+w. Here u, v price a neighbour SWAP before/after the layer, and
w prices a classical record permutation. Subject to the four budgets, the
classical implementation is strictly faster if w < 2*min(u,v) and all three
are feasible. Optimality covers exactly the declared rewrite orbit and costs.
"""
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

import z3

from diagram_vqa import (Candidate, Model, Program, example_kernel_catalogue,
                         native_word, replay_trace, saturate)
from optimal_compiler import Diagram, Result, solve

Number = int | Fraction


def _rational(x: Number) -> Fraction:
    if not isinstance(x, (int, Fraction)) or isinstance(x, bool):
        raise ValueError("use integer or Fraction for exact model constants")
    return Fraction(x)


@dataclass(frozen=True)
class PrimitiveCost:
    time: Number
    depth: int
    extra_atoms: int
    erasure: Number = 0

    def __post_init__(self):
        if (_rational(self.time) < 0 or not 0 <= _rational(self.erasure) <= 1
                or type(self.depth) is not int or self.depth < 0
                or type(self.extra_atoms) is not int or self.extra_atoms < 0):
            raise ValueError("negative resource or out-of-range erasure rate")


@dataclass(frozen=True)
class Limits:
    atoms: int
    depth: int
    error: Number
    fidelity: Number

    def __post_init__(self):
        if (type(self.atoms) is not int or self.atoms < 0
                or type(self.depth) is not int or self.depth < 0
                or not 0 <= _rational(self.error) <= 1
                or not 0 <= _rational(self.fidelity) <= 1):
            raise ValueError("invalid kernel resource or quality limits")


@dataclass(frozen=True)
class KernelProgram:
    candidate: Candidate
    price_keys: tuple[str, ...]
    time: Fraction
    depth: int
    atoms: int
    failure: Fraction
    fidelity: Fraction


def price_kernel(candidate: Candidate, prices: Mapping[str, PrimitiveCost],
                 data_atoms: int) -> KernelProgram:
    if type(data_atoms) is not int or data_atoms < 0:
        raise ValueError("data_atoms must be a nonnegative integer")
    time = Fraction(0)
    fidelity = Fraction(1)
    depth, atoms = 0, data_atoms
    keys = []
    layer_issued = False
    for operation in candidate.native:
        key = operation[0]
        if key == "SWAP":
            key = "SWAP_AFTER_LAYER" if layer_issued else "SWAP_BEFORE_LAYER"
        if key in ("RZZ", "RX"):
            layer_issued = True
        if key not in prices:
            raise ValueError(f"missing physical/online contract for {key}")
        cost = prices[key]
        time += _rational(cost.time)
        fidelity *= 1 - _rational(cost.erasure)
        depth += cost.depth
        atoms = max(atoms, data_atoms + cost.extra_atoms)
        keys.append(key)
    return KernelProgram(candidate, tuple(keys), time, depth, atoms, 1 - fidelity, fidelity)


def kernel_catalogue(model: Model, seed: Program, prices: Mapping[str, PrimitiveCost],
                     limits: Limits, data_atoms: int) -> tuple[Diagram, ...]:
    """Generate the orbit from seed, check every trace, lower and price every word."""
    if type(data_atoms) is not int or data_atoms < model.n:
        raise ValueError("explicit data-atom occupancy must cover the register")
    catalog = []
    for i, candidate in enumerate(saturate(model, seed)):
        restored = replay_trace(model, seed, candidate.trace)
        if restored != candidate.program or native_word(model, restored) != candidate.native:
            raise ValueError("kernel extraction or rewrite replay failed")
        p = price_kernel(candidate, prices, data_atoms)
        feasible = (p.atoms <= limits.atoms and p.depth <= limits.depth
                    and p.failure <= limits.error and p.fidelity >= limits.fidelity)
        name = f"kernel-{i}:" + ".".join(op.kind for op in candidate.program)
        catalog.append(Diagram(name, (), z3.BoolVal(feasible),
                               z3.RealVal(str(p.time)), p))
    return tuple(catalog)


def compile_kernel(prices: Mapping[str, PrimitiveCost], limits: Limits, *,
                   data_atoms: int, model: Model | None = None,
                   seed: Program | None = None) -> Result:
    """Return exact Result; result.diagram.program contains native word and proof."""
    if (model is None) != (seed is None):
        raise ValueError("model and seed must be passed together")
    if model is None:
        model, seed, _ = example_kernel_catalogue()
    catalog = kernel_catalogue(model, seed, prices, limits, data_atoms)
    return solve(catalog)
