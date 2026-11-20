"""Certified continuous-control compiler for the closed, two-readout VQE job.

The job and readout contract are those in feedback_vqe.Program: A first,
then B for a reported + and A for a reported -, returning both observed
readouts, energies and the parameter choice. The same service setting x in
[0,1] is committed before either readout. Each complete FT kernel pays an
extra blocking time x, with residual independently flipped decoded bit of
probability r(x)=exp(-x)/10. All staged costs, noise laws, resource numbers
and the completeness of the two native layout families are declared physical
interface assumptions, not estimates or a calibration of a neutral-atom array.
No pulse, trajectory, shot or random event is sampled.

The compiler compares the two layout families on the entire real interval.
The same integer and rational arithmetic validates every box in its proof.
The independent optimum/convexity arguments below use analytic properties
of this particular model; the solver only receives interval enclosures.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from math import isqrt

from feedback_vqe import Budgets, Layout, Program, leaf_distribution, symbolic_fixture
from interval_compiler import Candidate, Domain, Evaluation, Interval, solve, verify


CATALOG, _ = symbolic_fixture()
TARGET_FLIP = F(23, 338)
LIMITS = Budgets(12, 13, 1 - (1 - TARGET_FLIP) ** 2, F(675, 676) ** 2)
DELTA = F(1, 100)


def _unit_rational(x) -> F:
    if type(x) not in (F, int) or not 0 <= x <= 1:
        raise ValueError("the service setting must be a rational number in [0,1]")
    return F(x)


def exp_minus(x: F, bits: int) -> Interval:
    """Bracket exp(-x) by two adjacent partial sums of the alternating series."""
    x = _unit_rational(x)
    if type(bits) is not int or bits < 1:
        raise ValueError("positive precision in bits is required")
    tol = F(1, 1 << bits)
    partial = term = F(1)
    n = 0
    while True:
        n += 1
        term *= x / n
        previous = partial
        partial += -term if n % 2 else term
        if term <= tol:
            return Interval(min(previous, partial), max(previous, partial))


def _root(x: F, bits: int) -> Interval:
    scale = 1 << bits
    k = isqrt((x.numerator * scale * scale) // x.denominator)
    lower = F(k, scale)
    upper = lower if lower * lower == x else F(k + 1, scale)
    return Interval(lower, upper)


def _fidelity_at_r(r: F, bits: int) -> Interval:
    """Two retained readouts: F=(sqrt(3p/4)+sqrt((1-p)/4))^4."""
    p = F(3, 4) - r / 2
    first = _root(F(3, 4) * p, bits)
    second = _root((1 - p) / 4, bits)
    return Interval((first.lo + second.lo) ** 4,
                    min(F(1), (first.hi + second.hi) ** 4))


def _flip_at(x: F, bits: int) -> Interval:
    attenuation = exp_minus(x, bits)
    return Interval(attenuation.lo / 10, attenuation.hi / 10)


def _failure(r: F) -> F:
    return 1 - (1 - r) ** 2


def _time(layout: Layout, x: F, r: F) -> F:
    # Both orientations pay exactly the same two online readout services.
    p = F(3, 4) - r / 2
    base = F(CATALOG.a + CATALOG.b) + 2 * x
    if layout == "e":
        return base + CATALOG.delta + (1 - p) * CATALOG.kappa
    if layout == "s":
        return base + p * CATALOG.eta
    raise ValueError("unknown physical orientation")


def _enclose(layout: Layout, box: tuple[Interval, ...], precision: int) -> Evaluation:
    """Enclose whole-job time and all four budget residuals, each in <=0 form.

    r decreases with x. Failure increases with r. The classical one-readout
    affinity grows with p <= 3/4, so the two-readout fidelity decreases with
    r. Both physical durations increase with x: T_e'=2-r/2, T_s'=2+2r.
    At the same x, T_e increases with r and T_s decreases with r.
    """
    if len(box) != 1 or type(precision) is not int or precision < 0:
        raise ValueError("one control coordinate and a nonnegative precision are required")
    a, b = _unit_rational(box[0].lo), _unit_rational(box[0].hi)
    if a > b:
        raise ValueError("invalid service interval")
    bits = max(24, precision)
    ra, rb = _flip_at(a, bits), _flip_at(b, bits)
    time = (Interval(_time(layout, a, ra.lo), _time(layout, b, rb.hi))
            if layout == "e" else Interval(_time(layout, a, ra.hi), _time(layout, b, rb.lo)))
    f_low = _fidelity_at_r(ra.hi, bits).lo
    f_high = _fidelity_at_r(rb.lo, bits).hi
    atom_cost = CATALOG.peak_atoms - LIMITS.atoms
    depth_cost = CATALOG.prefix_depth + CATALOG.suffix_depth - LIMITS.depth
    constraints = (Interval(atom_cost, atom_cost), Interval(depth_cost, depth_cost),
                   Interval(_failure(rb.lo) - LIMITS.error, _failure(ra.hi) - LIMITS.error),
                   Interval(LIMITS.fidelity - f_high, LIMITS.fidelity - f_low))
    return Evaluation(time, constraints)


def domains() -> tuple[Domain, Domain]:
    return tuple(Domain(layout, (Interval(0, 1),), 4,
                        lambda b, prec, layout=layout: _enclose(layout, b, prec))
                 for layout in ("e", "s"))


@dataclass(frozen=True)
class ControlledProgram:
    layout: Layout
    control: F

    def __post_init__(self):
        _unit_rational(self.control)
        if self.layout not in ("e", "s"):
            raise ValueError("unknown two-round native implementation")

    def instructions(self) -> tuple:
        setup = ("use_shared_readout_service_setting", self.control,
                 "apply and charge once in each of the two encoded kernels")
        return (setup, *Program(self.layout).instructions())

    def replay(self, first: int, second: int) -> dict:
        # These fields exactly specify the user's delivered register. Physical
        # layout, control knob and hidden faults are backend data.
        record = Program(self.layout).replay(first, second)
        return {key: record[key] for key in ("parameters", "readouts", "empirical_energies", "choice")}


def certified_job(delta: F = DELTA, max_rounds: int = 14):
    ds = domains()
    cert = solve(ds, delta, max_rounds=max_rounds, seeds=(Candidate("e", (F(1),)),))
    if not verify(ds, delta, cert):
        raise ArithmeticError("the interval cover or its feasible point failed replay")
    if cert.status != "delta_optimal":
        return cert, None
    point = cert.witness.candidate
    return cert, ControlledProgram(point.domain, point.point[0])


def analytic_optimum(terms: int = 12) -> Interval:
    """An independent rational bracket for T*=1400/169 + 2*log(169/115).

    At r*=23/338, p*=121/169 and both quality thresholds are exactly
    attained. The single-readout affinity squared and the two-readout
    affinity both equal 675/676. Both qualities improve with x, so feasibility means
    x >= log(169/115). On [0,1], T_s-T_e=7/4-5r/2 > 0 and T_e'>0.
    The feasible boundary is interior (1 < 169/115 < 2 < exp(1)).

    Even an initial independent lottery over (layout,x) cannot beat this
    result. For y=2r-r^2 and t(y)=33/4-2 log(10r)+r/2, differentiation
    gives t'<0 and t''=(4/r^2-8/r+1)/(2-2r)^3 > 0. Thus E[T] >=
    t(E[y]) >= t(error_limit) by Jensen. This uses the linear failure
    probability budget; it does not average a nonlinear fidelity across
    controllers. The code searches fixed shared settings. Independent
    branch-specific settings or an analog pulse library require new domains.
    """
    if type(terms) is not int or terms < 1:
        raise ValueError("a positive number of logarithm terms is required")
    t = F(27, 142)  # (169/115-1)/(169/115+1)
    lower = 2 * sum((t ** (2 * k + 1) / (2 * k + 1) for k in range(terms)), F(0))
    remainder = 2 * t ** (2 * terms + 1) / ((2 * terms + 1) * (1 - t * t))
    constant = F(1400, 169)
    return Interval(constant + 2 * lower, constant + 2 * (lower + remainder))


if __name__ == "__main__":
    certificate, program = certified_job()
    print("status", certificate.status, "gap tolerance", DELTA)
    print("certified global bounds", certificate.lower, certificate.upper)
    if program is not None:
        print("executable witness", program.layout, program.control)
        for readings in leaf_distribution(0):
            print(program.replay(*readings))
