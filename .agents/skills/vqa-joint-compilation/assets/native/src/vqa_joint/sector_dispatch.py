"""Structural preparation/evolution/readout rewriting of fully measured bound calls."""
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
import json
import os
import subprocess
import tempfile
from typing import TYPE_CHECKING

import numpy as np

from vqa_joint.hosts.native import HOSTS, WORK, compile_qasm
from vqa_joint.ports import WirePort
from vqa_joint.rewrite import normalize, qasm, standard
from vqa_joint.sector_vqe.affine_port import recognize as recognize_affine
from vqa_joint.sector_qaoa.quotient import FlipQuotient
from vqa_joint.sector_qaoa.parity_port import apply_binary, parity_ports
from vqa_joint.sector_qaoa.permutation_orbit import recognize_two_orbit

if TYPE_CHECKING:
    from vqa_joint.joint import LoweredCall


@dataclass(frozen=True)
class Readout:
    """Kernel from logical slots of the reduced word to canonical full-register bits."""
    kind: str
    width: int
    native_width: int
    random_width: int
    reference: int = -1
    mask: int = 0
    decode_rows: tuple = ()
    classes: tuple = ()
    xor_offset: int = 0

    def choice_count(self, measured):
        if not isinstance(measured, Integral) or not 0 <= measured < 1 << self.native_width:
            raise ValueError("Measured sample outside the native register")
        return len(self.classes[measured]) if self.kind == "orbit_sample" else 1 << self.random_width

    def mean_random_bits(self, measured):
        count = self.choice_count(measured)
        bits = (count - 1).bit_length()
        return bits * (1 << bits) / count

    def sample_index(self, measured, fair_bits):
        count = self.choice_count(measured)
        width = (count - 1).bit_length()
        if width == 0:
            return 0
        while True:
            sample = fair_bits(width)
            if sample < count:
                return sample

    def decode_bits(self, measured, random_bits):
        if (not isinstance(measured, Integral) or not 0 <= measured < 1 << self.native_width
                or not isinstance(random_bits, Integral) or not 0 <= random_bits < self.choice_count(measured)):
            raise ValueError("Readout sample outside declared register width")
        measured, random_bits = int(measured), int(random_bits)
        if self.kind == "identity":
            return measured
        if self.kind in ("binary_parity", "affine_two_word"):
            return apply_binary(self.decode_rows, measured | (random_bits << self.native_width)) ^ self.xor_offset
        if self.kind == "orbit_sample":
            return self.classes[measured][random_bits]
        if self.kind == "global_flip":
            return FlipQuotient(self.width, self.reference).pair(measured)[random_bits]
        if self.kind == "fanout_Z":
            return self.mask * measured
        if self.kind != "fanout_X":
            raise ValueError(f"Unsupported readout {self.kind}")
        # Choose the pivot so dot(mask, output) == measured; the other n-1 bits are uniform.
        pivot = self.mask.bit_length() - 1
        lower = (1 << pivot) - 1
        free = (random_bits & lower) | ((random_bits & ~lower) << 1)
        return free | ((measured ^ ((self.mask & free).bit_count() % 2)) << pivot)

    def lift_distribution(self, p):
        p = np.asarray(p, dtype=float)
        if (p.shape != (1 << self.native_width,) or not np.isfinite(p).all()
                or np.any(p < 0) or not np.isclose(p.sum(), 1., rtol=0., atol=1e-10)):
            raise ValueError("Expected normalized finite nonnegative native probabilities")
        if self.kind == "identity":
            return p.copy()
        result = np.zeros(1 << self.width)
        for measured, mass in enumerate(p):
            count = self.choice_count(measured)
            for free in range(count):
                result[self.decode_bits(measured, free)] += mass / count
        return result


@dataclass(frozen=True)
class InstrumentPlan:
    family: str
    variant: str
    ops: tuple
    readout: Readout


@dataclass(frozen=True)
class InstrumentCall:
    family: str
    variant: str
    lowered: "LoweredCall"
    readout: Readout

    def logical_probabilities(self, native_p):
        # The native schedule may have a wire permutation from lower_call.
        native_p = np.asarray(native_p)
        if native_p.shape != (1 << self.readout.native_width,):
            raise ValueError("Native probability vector has wrong size")
        ordered = np.empty_like(native_p)
        for raw, mass in enumerate(native_p):
            ordered[self.lowered.wire_port.logical_outcome(raw)] = mass
        return self.readout.lift_distribution(ordered)

    def decode_shot(self, native_outcome, rng):
        slot = self.lowered.wire_port.logical_outcome(native_outcome)
        random_bits = self.readout.sample_index(slot, lambda bits: int(rng.integers(1 << bits)))
        return self.readout.decode_bits(slot, random_bits)


def _all_h(gates, width):
    operands = [w for g, w, angle in gates if g == "h" and angle is None]
    return len(gates) == width and sorted(operands) == [(q,) for q in range(width)]


def _fanout(gates, n):
    """Recognize a one-qubit input encoded by a binary linear map.

    Its state has support on 0 and mask, the selected column of the CNOT
    network. Z readout returns those two words. In the X basis, every word
    with a given parity against mask has the same Born weight. That parity
    follows the X readout of the active input, for arbitrary complex states.
    """
    first = next((i for i, op in enumerate(gates) if op[0] == "cx"), len(gates))
    if first == 0 or first == len(gates):
        return None
    prep = gates[:first]
    rotations = {"rx", "ry", "rz"}
    fixed = {"h", "x", "y", "z", "s", "sdg", "t", "tdg"}
    if not all((g in rotations and angle is not None and np.isfinite(angle) or g in fixed and angle is None)
               and len(w) == 1 and w == prep[0][1] for g, w, angle in prep):
        return None
    source = prep[0][1][0]
    tail = gates[first:]
    x_basis = _all_h(tail[-n:], n)
    linear = tail[:-n] if x_basis else tail
    mask = 1 << source
    for g, w, angle in linear:
        if g != "cx" or len(w) != 2 or angle is not None:
            return None
        control, target = w
        mask ^= ((mask >> control) & 1) << target
    basis = "X" if x_basis else "Z"
    small = [(g, (0,), a) for g, _, a in prep]
    if x_basis:
        small.append(("h", (0,), None))
    small.append(("measure", (0,), None))
    return InstrumentPlan("cnot_fanout", f"fanout_{basis}", tuple(small),
                          Readout(f"fanout_{basis}", n, 1, n-1 if x_basis else 0, mask=mask))


def instrument_plans(ops, n, enabled=True):
    """Emit identity and exact structural alternatives from canonical indexed final Z readout.

    Gates have the usual tuple (name, operand tuple, numeric angle or None).
    The source starts in |0...0>. A full uniform X readout is written as H on
    each wire before the final Z measurements. Ancillas and mid-call effects
    require an explicit interface beyond this entry point.
    """
    if n < 1:
        raise ValueError("Instrument needs a nonempty register")
    ops = tuple((g.lower(), tuple(w), a) for g, w, a in ops)
    if any(not w or len(set(w)) != len(w) or any(not isinstance(q, Integral) or not 0 <= q < n for q in w)
           for _, w, _ in ops):
        raise ValueError("Invalid instrument wire")
    suffix = tuple(("measure", (q,), None) for q in range(n))
    if ops[-n:] != suffix or any(g == "measure" for g, _, _ in ops[:-n]):
        raise ValueError("Expected full terminal measurements in canonical register order")
    identity = InstrumentPlan("identity", "identity", ops, Readout("identity", n, n, 0))
    plans = [identity]
    if not enabled or n == 1:
        return plans
    body = ops[:-n]
    evolution = body[n:]
    orbit = recognize_two_orbit(body, n)
    if orbit is not None:
        reduced, classes = orbit
        plans.append(InstrumentPlan("permutation_orbits", "triangle_two_orbits", reduced,
                     Readout("orbit_sample", n, 1, 3, classes=classes)))
    if _all_h(body[:n], n):
        for variant, port in parity_ports(evolution, n):
            rank = len(port.checks)
            plans.append(InstrumentPlan("binary_parity", variant, port.operations,
                         Readout("binary_parity", n, rank, n-rank, decode_rows=port.decode_rows)))
    if _all_h(body[:n], n) and all(
        (g == "rx" and len(w) == 1 or g == "rzz" and len(w) == 2) and a is not None and np.isfinite(a)
        for g, w, a in evolution
    ):
        for ref in range(n):
            port = FlipQuotient(n, ref)
            uniform = [("h", (q,), None) for q in range(n-1)]
            reduced = uniform + port.layers(evolution) + list(suffix[:-1])
            plans.append(InstrumentPlan("global_flip", f"reference_{ref}", tuple(reduced),
                                        Readout("global_flip", n, n-1, 1, reference=ref)))
    fanout = _fanout(body, n)
    if fanout is not None:
        plans.append(fanout)
    else:
        affine = recognize_affine(body, n)
        if affine is not None:
            kernel = Readout("affine_two_word", n, 1, affine.random_width,
                             decode_rows=affine.rows, xor_offset=affine.offset)
            plans.append(InstrumentPlan("affine_two_word", "affine_two_word", affine.operations, kernel))
    return plans


def _zac_single(plan, profile, directory):
    """Specialize the empty-interaction branch using ZAC's native one-site router."""
    from vqa_joint.joint import LoweredCall
    source, envdir, _ = HOSTS["zac"]
    source = os.environ.get("JOINT_ZAC_SOURCE", str(WORK / source))
    python = os.environ.get("JOINT_ZAC_PYTHON", str(WORK / envdir / "bin/python"))
    worker = Path(__file__).resolve().parent / "sector_vqe" / "zac_single.py"
    out = Path(directory) / profile / "single-site"
    with tempfile.TemporaryDirectory(prefix="sector-zac-") as td:
        scratch = Path(td)
        (scratch / "call.qasm").write_text(qasm(standard(normalize(plan.ops)), 1))
        result = subprocess.run([python, str(worker), source, profile], cwd=scratch, text=True, capture_output=True,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", RAYON_NUM_THREADS="1",
                                         QISKIT_PARALLEL="FALSE", MPLBACKEND="Agg"))
        if result.returncode or "[Error]" in result.stdout:
            raise ValueError((result.stdout + result.stderr)[-1800:])
        metrics = json.loads((scratch / "metrics.json").read_text())
        if metrics["geometry_findings"] or metrics["unitary_operator_error"] is None or metrics["unitary_operator_error"] > 1e-8:
            raise ValueError("Single-site schedule failed the native certificate")
        out.mkdir(parents=True, exist_ok=True)
        for name in ("call.qasm", "metrics.json", "bound_schedule.json", "native_trace.json"):
            data = (scratch / name).read_bytes()
            with tempfile.NamedTemporaryFile(dir=out, delete=False) as staged:
                staged.write(data)
            os.replace(staged.name, out / name)
    return LoweredCall("single-site", profile, out / "bound_schedule.json", metrics, WirePort((0,)))


def lower_instrument(algorithm, ops, n, directory, profiles=None, mode_filter=None, allow_wire_relabel=False):
    plans = instrument_plans(ops, n, enabled=algorithm.changed)
    profiles = tuple(profiles) if profiles is not None else algorithm.host.profiles
    mode_filter = tuple(mode_filter) if mode_filter is not None else None
    accepted, rejected = [], []
    for plan in plans:
        out = Path(directory) / plan.variant
        calls, issues = algorithm.lower_call(plan.ops, plan.readout.native_width, out, profiles, mode_filter, allow_wire_relabel)
        rejected.extend((plan.variant, *issue) for issue in issues)
        # Upstream solve has an empty-list failure in its reuse pass on a one-atom program.
        if algorithm.host.native_compile is compile_qasm and algorithm.host.name == "zac" and plan.readout.native_width == 1:
            for profile in profiles:
                if any(c.profile == profile for c in calls):
                    continue
                if any(i[1] == profile and i[2] == "native compilation" and "IndexError" in str(i[3])
                       and "collect_reuse_qubit" in str(i[3]) for i in issues):
                    try:
                        calls.append(_zac_single(plan, profile, out))
                    except (OSError, RuntimeError, ValueError) as exc:
                        rejected.append((plan.variant, "single-site", profile, "native compilation", str(exc)))
        accepted.extend(InstrumentCall(plan.family, plan.variant, call, plan.readout) for call in calls)
    return accepted, rejected
