"""Quotient a measured variational circuit by its binary translation symmetries."""
from dataclasses import dataclass
from math import isfinite

from vqa_joint.rewrite import coordinates


def apply_binary(rows, word):
    return sum(((row & word).bit_count() & 1) << i for i, row in enumerate(rows))


@dataclass(frozen=True)
class ParityPort:
    # The first len(checks) coordinates are measured qubits. The rest are fair bits.
    checks: tuple
    final_frame: tuple
    decode_rows: tuple
    operations: tuple


def _collect(gates, width):
    # x_physical = frame * x, and inverse_columns[j] = frame^-1 * unit_j.
    frame = [1 << i for i in range(width)]
    inverse_columns = frame.copy()
    rotations = []
    for name, wires, theta in gates:
        if name == "cx" and theta is None and len(wires) == 2:
            control, target = wires
            frame[target] ^= frame[control]
            inverse_columns[control] ^= inverse_columns[target]
        elif name in ("rx", "rz", "rzz") and theta is not None and isfinite(theta):
            if len(wires) != (2 if name == "rzz" else 1):
                return None
            mask = inverse_columns[wires[0]] if name == "rx" else frame[wires[0]]
            if name == "rzz":
                mask ^= frame[wires[1]]
            rotations.append(("x" if name == "rx" else "z", mask, theta))
        else:
            return None
    return rotations, tuple(frame)


def _independent(rows, pivots):
    retained = []
    for row in rows:
        remainder = row
        while remainder:
            lead = remainder.bit_length() - 1
            if lead not in pivots:
                pivots[lead] = remainder
                retained.append(row)
                break
            remainder ^= pivots[lead]
    return tuple(retained)


def _rotation(axis, mask, theta):
    if not mask:
        return []  # X on a translated fibre acts as the identity, up to scalar phase.
    wires = [i for i in range(mask.bit_length()) if mask & (1 << i)]
    pivot = wires[0]
    if axis == "x":
        compute = [("cx", (pivot, j), None) for j in wires[1:]]
    else:
        compute = [("cx", (j, pivot), None) for j in wires[1:]]
    return compute + [("r" + axis, (pivot,), theta)] + list(reversed(compute))


def parity_ports(gates, width):
    """Input is the evolution following all-plus preparation, before terminal Z readout.

    Each valid basis of the actual Z-string row space yields a native quantum word
    and a linear stochastic readout. Changing the basis can change physical routing.
    """
    collected = _collect(gates, width)
    if collected is None:
        return
    rotations, frame = collected
    rows = tuple(dict.fromkeys(mask for axis, mask, _ in rotations if axis == "z"))
    seen = set()
    for variant, ordered in (("parity_first", rows), ("parity_reverse", reversed(rows))):
        pivots = {}
        checks = _independent(ordered, pivots)
        rank = len(checks)
        if not 0 < rank < width or checks in seen:
            continue
        seen.add(checks)
        coordinates_full = checks + _independent((1 << i for i in range(width)), pivots)
        decode_rows = tuple(coordinates(coordinates_full, row) for row in frame)
        ops = [("h", (i,), None) for i in range(rank)]
        for axis, vec, theta in rotations:
            reduced = coordinates(checks, vec) if axis == "z" else apply_binary(checks, vec)
            ops.extend(_rotation(axis, reduced, theta))
        ops.extend(("measure", (i,), None) for i in range(rank))
        yield variant, ParityPort(checks, frame, decode_rows, tuple(ops))
