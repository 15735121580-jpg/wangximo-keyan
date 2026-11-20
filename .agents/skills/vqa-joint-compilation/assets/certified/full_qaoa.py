"""A complete, fixed-budget, feedback Max-Cut QAOA job in a declared FT catalog.

Two vertices, one edge, C=(I-ZZ)/2. Prepare |++>, apply exp(-i gamma C),
then exp(-i beta (X1+X2)), beta=pi/8, and decode/read both logical bits.
Round one uses A (gamma=-pi/6). If it reports a cut, repeat A; otherwise
select B (gamma=+pi/6). Independently reprepare and run round two. Deliver
the sampled string with best cut score and that score; ties go to round one.

The declared encoded instrument has two 3-atom bit-flip repetition blocks
and two syndrome ancillas. Its preparation, logical gates, routing and
syndrome operations are exact in the assumed readout-only noise model.
The three raw final bits of block 1 flip independently with probability
1/10 before majority decoding; block 2 reports exactly. A fresh shot has
independent faults. The conditional costs below are aggregate charged costs
of complete native instruments, including online feedback/decoding; depth
and peak are declared resource bounds attained by both implementations.
This is a finite abstract instruction contract, not a calibration. It tests
neither general coherent/phase faults nor universal control over a device.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F

import z3

from feedback_vqe import Budgets
from optimal_compiler import Diagram, solve
from policy_ir import (Action, Decision, Leaf, Outcome, PublicPort, compile_diagrams)


STRINGS = ("00", "01", "10", "11")
ANGLES = {"A": "-pi/6", "B": "+pi/6"}
PHYSICAL_FLIP = F(1, 10)
RESIDUAL = 3 * PHYSICAL_FLIP ** 2 - 2 * PHYSICAL_FLIP ** 3
BUDGET = Budgets(8, 15, F(3, 50), F(999, 1000))


def score(word: str) -> int:
    if word not in STRINGS:
        raise ValueError("expected a two-bit computational-basis readout")
    return int(word[0] != word[1])


def update(first: str) -> str:
    return "A" if score(first) else "B"


def deliver(first: str, second: str) -> tuple[str, int]:
    best = second if score(second) > score(first) else first
    return best, score(best)


def law(parameter: str, r: F) -> dict[str, F]:
    """Per-string probabilities after decoding, one round.

    In the equal/unequal Bell basis the cost layer is diag(1,exp(-i gamma))
    and the mixer is exp(-2i beta X). The cut probability is therefore
    (1+sin(4 beta)sin(gamma))/2. Within each parity the two strings are
    equiprobable. A single residual first-bit flip exchanges the parities.
    """
    ideal = {"A": F(1, 4), "B": F(3, 4)}[parameter]
    if not 0 <= r <= 1:
        raise ValueError("invalid readout flip probability")
    cut = ideal * (1-r) + (1-ideal) * r
    return {w: (cut if score(w) else 1-cut) / 2 for w in STRINGS}


def encoded_shot(parameter, layout, round_index):
    return ("prepare fresh two encoded |+> blocks", ("map, route, address", layout),
            ("cost layer exp(-i gamma C)", ANGLES[parameter]),
            ("mixer exp(-i beta (X0+X1))", "pi/8"),
            ("extract bit-flip syndrome; apply/track correction", "each block"),
            ("measure physical bits; majority-decode each block", f"z{round_index}"))


@dataclass(frozen=True)
class Program:
    first_layout: str

    def instructions(self):
        if self.first_layout not in ("e", "s"):
            raise ValueError("unknown QAOA layout")
        return (encoded_shot("A", self.first_layout, 1),
                ("branch on decoded z1", tuple(
                    (w, encoded_shot(update(w), self.first_layout, 2)) for w in STRINGS)),
                ("deliver", "higher-cut sampled string", "score", "first on ties"))

    def replay(self, first, second):
        return {"angles": (ANGLES["A"], ANGLES[update(first)]),
                "readouts": (first, second), "delivered": deliver(first, second)}


def _z(q: F):
    q = F(q)
    return z3.RealVal(f"{q.numerator}/{q.denominator}")


@dataclass(frozen=True)
class Backend:
    budgets: Budgets = BUDGET

    def is_terminal(self, port, history):
        return port.type_name == "delivered"

    def actions(self, port, history):
        if port.type_name == "start":
            return tuple(Action(layout, "start", tuple(Outcome(w, PublicPort(
                "first", (layout, w))) for w in STRINGS)) for layout in ("e", "s"))
        if port.type_name == "first":
            layout, first = port.fields
            return (Action(f"{update(first)}-{layout}", "first",
                           tuple(Outcome(w, PublicPort("delivered", (first, w))) for w in STRINGS)),)
        raise ValueError("unexpected quantum-program port")

    def lower(self, tree):
        if not isinstance(tree, Decision) or tree.action.name not in ("e", "s"):
            raise ValueError("missing first shot")
        layout = tree.action.name
        laws = [{(w, score(w)): F(0) for w in STRINGS} for _ in range(2)]
        time = F(4 if layout == "e" else 3)
        for branch, subtree in tree.branches:
            first = branch.label
            next_parameter = update(first)
            if not isinstance(subtree, Decision) or subtree.action.name != f"{next_parameter}-{layout}":
                raise ValueError("parameter update must use the actual first readout")
            # Kernel time includes classical update, readout and delivery.
            extra = int(layout == "e" and next_parameter == "A")
            extra += 4 * int(layout == "s" and next_parameter == "B")
            time += law("A", RESIDUAL)[first] * (4 + extra)
            for outcome, leaf in subtree.branches:
                if not isinstance(leaf, Leaf) or leaf.kind != "delivered":
                    raise ValueError("missing a required second readout")
                second = outcome.label
                for emitted, residual in zip(laws, (F(0), RESIDUAL)):
                    emitted[deliver(first, second)] += (
                        law("A", residual)[first] * law(next_parameter, residual)[second])
        ideal, noisy = laws
        if any(sum(p.values()) != 1 for p in laws):
            raise ValueError("incomplete feedback tree")
        affinity = sum(z3.Sqrt(_z(ideal[y] * noisy[y])) for y in ideal)
        budget = self.budgets
        feasible = z3.And(8 <= budget.atoms, 7 + 8 <= budget.depth,
                          _z(1 - (1-RESIDUAL)**2) <= _z(budget.error),
                          z3.simplify(affinity**2 >= _z(budget.fidelity)))
        return Diagram(f"qaoa_{layout}", (), feasible, _z(time), Program(layout))


def compile_job(budgets: Budgets = BUDGET):
    """Enumerate the two permitted orientations and all sixteen visible branches each."""
    return solve(compile_diagrams(Backend(budgets), PublicPort("start"), 2))


def analytic_result(budget: Budgets = BUDGET):
    """Rational check independent of tree enumeration and SMT.

    Failure of the first cut selects B, whose failure probability equals
    the cut probability of A. Thus the delivered score is zero with
    probability f(1-f), f=P(first score=0). Global string complementation
    preserves the whole policy; within each delivered score its two possible
    strings have equal probabilities even with the first-round tie rule.
    """
    f = (3*(1-RESIDUAL)+RESIDUAL)/4
    p0 = f*(1-f)
    ideal0 = F(3, 16)
    a, b = ideal0*p0, (1-ideal0)*(1-p0)
    excess = budget.fidelity - a - b
    fidelity_ok = excess <= 0 or excess**2 <= 4*a*b
    failure = 1-(1-RESIDUAL)**2
    if budget.atoms < 8 or budget.depth < 15 or failure > budget.error or not fidelity_ok:
        return None
    # Every contract-complete physical policy commits to e or s before z1;
    # all post-z1 moves are included in the conditional native kernel costs.
    return min(((8+(1-f), "e"), (7+4*f, "s")))


if __name__ == "__main__":
    answer = compile_job()
    print(answer.status, answer.time, analytic_result())
    print(answer.diagram.program.native_program.instructions())
