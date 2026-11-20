"""Exact two-round feedback VQE from the theorem, with no sampling.

The hardware contract supplies two first-round orientations and one least-time
second-round encoded kernel per orientation and parameter. Each kernel includes
preparation, mapping, routing, FT rotation, syndrome extraction, decoding and
logical Z readout. Its time includes the blocking online work. Atom and depth
values below are exact properties of this *declared* transfer catalog; certifying
them for a particular atom array is an independent instruction-level obligation.

Run with optimal_compiler.py in the same directory and z3-solver installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import z3

from optimal_compiler import Diagram
from policy_ir import Action, Decision, History, Leaf, Outcome, PublicPort, Tree, compile_diagrams


Number = Fraction | int
Layout = Literal["e", "s"]


def _q(value: Number) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("Use int or Fraction for exact rational input.")
    return Fraction(value)


def _real(value: Number) -> z3.ArithRef:
    value = _q(value)
    return z3.RealVal(f"{value.numerator}/{value.denominator}")


@dataclass(frozen=True)
class Catalog:
    a: Number
    b: Number
    delta: Number
    eta: Number
    kappa: Number
    flip: Number
    peak_atoms: int
    prefix_depth: int
    suffix_depth: int

    def __post_init__(self) -> None:
        if any(_q(c) <= 0 for c in (self.a, self.b, self.delta, self.eta, self.kappa)):
            raise ValueError("Durations must be positive.")
        if not 0 <= _q(self.flip) < Fraction(1, 2):
            raise ValueError("Readout-flip probability must lie in [0,1/2).")
        if any(type(n) is not int or n <= 0
               for n in (self.peak_atoms, self.prefix_depth, self.suffix_depth)):
            raise ValueError("Use positive integer catalog atom counts and depths.")


@dataclass(frozen=True)
class Budgets:
    atoms: int
    depth: int
    error: Number
    fidelity: Number

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 0 for n in (self.atoms, self.depth)):
            raise ValueError("Atom and depth limits must be nonnegative integers.")
        if not all(0 <= _q(x) <= 1 for x in (self.error, self.fidelity)):
            raise ValueError("Quality thresholds must lie in [0,1].")


def _positive_probability(parameter: str, flip: Number) -> Fraction:
    ideal = {"A": Fraction(3, 4), "B": Fraction(1, 4)}[parameter]
    noise = _q(flip)
    return ideal * (1 - noise) + (1 - ideal) * noise


def leaf_distribution(flip: Number) -> dict[tuple[int, int], Fraction]:
    """Exact law of both retained readouts for the ideal (flip=0) or noisy job."""
    p = _positive_probability("A", flip)
    result = {}
    for first, first_prob, parameter in ((1, p, "B"), (-1, 1 - p, "A")):
        second_p = _positive_probability(parameter, flip)
        result[first, 1] = first_prob * second_p
        result[first, -1] = first_prob * (1 - second_p)
    return result


@dataclass(frozen=True)
class Program:
    """A recoverable adaptive program over the specified FT kernel interface."""
    first_orientation: Layout

    def instructions(self) -> tuple:
        return (
            ("run_encoded_kernel", "A", "angle=pi/3", "retain_Z_as_m1",
             "commit_layout_before_m1", self.first_orientation),
            ("branch_on_visible_m1", (
                (1, ("run_encoded_kernel", "B", "angle=2*pi/3",
                     "entry_layout", self.first_orientation, "retain_Z_as_m2")),
                (-1, ("run_encoded_kernel", "A", "angle=pi/3",
                      "entry_layout", self.first_orientation, "retain_Z_as_m2")),
            )),
            ("deliver", "m1", "m2", "sample-means by visited parameter",
             "lowest empirical energy; choose A on ties"),
        )

    def replay(self, first: int, second: int) -> dict:
        """Deterministically replay final delivery for any visible history."""
        if type(first) is not int or type(second) is not int or (
                first not in (-1, 1) or second not in (-1, 1)):
            raise ValueError("Two Pauli eigenvalues +1 or -1 are required.")
        second_parameter = "B" if first == 1 else "A"
        if second_parameter == "A":
            energies = {"A": Fraction(first + second, 2)}
        else:
            energies = {"A": Fraction(first), "B": Fraction(second)}
        return {
            "first_layout": self.first_orientation,
            "parameters": ("A", second_parameter),
            "readouts": (first, second),
            "empirical_energies": energies,
            "choice": min(energies, key=lambda key: (energies[key], key)),
        }


@dataclass(frozen=True)
class FeedbackBackend:
    """Public protocol of the two-round job; noisy hidden flips are evaluated in lower."""
    catalog: Catalog
    budgets: Budgets

    def is_terminal(self, port: PublicPort, history: History) -> bool:
        return port.type_name == "delivered"

    def actions(self, port: PublicPort, history: History) -> tuple[Action, ...]:
        if port.type_name == "start":
            return tuple(
                Action(f"encoded-A-finish-{layout}", "start", tuple(
                    Outcome(label, PublicPort("after-A", (layout, label)))
                    for label in ("+", "-")))
                for layout in ("e", "s")
            )
        if port.type_name == "after-A":
            layout, first = port.fields
            parameter = "B" if first == "+" else "A"
            return (Action(f"encoded-{parameter}-from-{layout}", "after-A", tuple(
                Outcome(label, PublicPort("delivered", (first, label)))
                for label in ("+", "-"))),)
        raise ValueError("unexpected public VQE port")

    def lower(self, tree: Tree) -> Diagram:
        if not isinstance(tree, Decision) or tree.action.name not in {
                "encoded-A-finish-e", "encoded-A-finish-s"}:
            raise ValueError("VQE requires a first encoded sampling action")
        layout = tree.action.name[-1]
        r = _q(self.catalog.flip)
        ideal, noisy = {}, {}
        total = _q(self.catalog.a) + (_q(self.catalog.delta) if layout == "e" else 0)
        for outcome, branch in tree.branches:
            first = outcome.label
            parameter = "B" if first == "+" else "A"
            if (not isinstance(branch, Decision)
                    or branch.action.name != f"encoded-{parameter}-from-{layout}"):
                raise ValueError("second parameter must depend on the *visible* first result")
            prefix_p = _positive_probability("A", r)
            prefix_p = prefix_p if first == "+" else 1 - prefix_p
            second_time = _q(self.catalog.b)
            if first == "+" and layout == "s":
                second_time += _q(self.catalog.eta)
            if first == "-" and layout == "e":
                second_time += _q(self.catalog.kappa)
            total += prefix_p * second_time
            for second, leaf in branch.branches:
                if not isinstance(leaf, Leaf) or leaf.kind != "delivered":
                    raise ValueError("a VQE readout branch did not deliver both samples")
                for law, flip in ((ideal, 0), (noisy, r)):
                    prob_1 = _positive_probability("A", flip)
                    prob_2 = _positive_probability(parameter, flip)
                    law[first, second.label] = (
                        (prob_1 if first == "+" else 1 - prob_1)
                        * (prob_2 if second.label == "+" else 1 - prob_2)
                    )
        if len(ideal) != 4 or sum(ideal.values()) != 1 or sum(noisy.values()) != 1:
            raise ValueError("incomplete binary VQE instrument")
        affinity = sum(z3.Sqrt(_real(ideal[y] * noisy[y])) for y in ideal)
        feasible = z3.And(
            self.catalog.peak_atoms <= self.budgets.atoms,
            self.catalog.prefix_depth + self.catalog.suffix_depth <= self.budgets.depth,
            _real(1 - (1 - r) ** 2) <= _real(self.budgets.error),
            z3.simplify(affinity ** 2 >= _real(self.budgets.fidelity)),
        )
        return Diagram(name=f"feedback_vqe_{layout}", variables=(),
                       feasible=feasible, time=_real(total),
                       program=Program(layout).instructions())


def build_diagrams(catalog: Catalog, budgets: Budgets) -> list[Diagram]:
    """Generate public action/observation trees from the backend and lower each."""
    return compile_diagrams(FeedbackBackend(catalog, budgets), PublicPort("start"), 2)


def analytic_optimum(catalog: Catalog, budgets: Budgets) -> tuple[Fraction | None, tuple]:
    """Independent rational-only feasibility/optimality oracle; returns all ties."""
    r = _q(catalog.flip)
    p = Fraction(3, 4) - r / 2
    x = Fraction(3, 4) * p
    y = Fraction(1, 4) * (1 - p)
    # (sqrt(x)+sqrt(y))^4 = base + 4*(x+y)*sqrt(x*y).
    base = (x + y) ** 2 + 4 * x * y
    excess = _q(budgets.fidelity) - base
    fidelity_ok = excess <= 0 or excess ** 2 <= 16 * (x + y) ** 2 * x * y
    if (catalog.peak_atoms > budgets.atoms
            or catalog.prefix_depth + catalog.suffix_depth > budgets.depth
            or 2 * r - r ** 2 > _q(budgets.error)
            or not fidelity_ok):
        return None, ()
    e = _q(catalog.delta) + (1 - p) * _q(catalog.kappa)
    s = p * _q(catalog.eta)
    least = min(e, s)
    options = tuple(layout for layout, cost in (("e", e), ("s", s)) if cost == least)
    return _q(catalog.a) + _q(catalog.b) + least, options


def solve_feedback_vqe(catalog: Catalog, budgets: Budgets):
    """Solve in the exact end-to-end compiler; returned diagram carries a branch program."""
    from optimal_compiler import solve

    return solve(build_diagrams(catalog, budgets))


def symbolic_fixture() -> tuple[Catalog, Budgets]:
    """Synthetic exact transfer-table input for code verification, in model ticks."""
    return (Catalog(3, 4, 1, 4, 1, Fraction(1, 10), 12, 5, 8),
            Budgets(12, 13, Fraction(1, 5), Fraction(99, 100)))


if __name__ == "__main__":
    table, limits = symbolic_fixture()
    result = solve_feedback_vqe(table, limits)
    expected, orientations = analytic_optimum(table, limits)
    print("status", result.status, "time", result.time)
    print("hand-derived time", expected, "optimal layout(s)", orientations)
    print("delivered output if readouts are +1,-1",
          Program("e").replay(1, -1))
