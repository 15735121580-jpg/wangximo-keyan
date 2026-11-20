"""Exact, dense compilation of diagonal-cost and XY-mixer programs at fixed weight.

The little-endian source register starts at zero. Source preparation flips the
bits of seed_mask before the declared preparation exchanges. Cost and XY layers
may repeat. Every coherent label in the fixed-weight sector is certified.
"""
from dataclasses import dataclass
from itertools import combinations
import math
from operator import index
from typing import Callable

import numpy as np


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _angle(value):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError("A finite real phase or exchange angle is required")
    return x


def _verify(actual, expected, tolerance):
    """Compare rectangular isometries up to one common global phase."""
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("Incompatible coherent maps")
    overlap = np.vdot(expected, actual)
    if abs(overlap) < 1e-14:
        raise ValueError("The proposed image is orthogonal to the target")
    error = float(np.max(abs(actual - (overlap / abs(overlap)) * expected)))
    if error > tolerance:
        raise ValueError(f"Coherent task-port discrepancy {error:.3g} exceeds tolerance")
    return error


@dataclass(frozen=True)
class CostMixerLayer:
    """energy(mask) returns a real, already scaled value for a source mask.

    exchanges contains ordered (wire_i, wire_j, beta) triples. Repeated edges
    and edge-specific angles are allowed.
    """
    gamma: float
    energy: Callable[[int], float]
    exchanges: tuple


@dataclass(frozen=True)
class FixedWeightImplementation:
    qasm: str | None
    width: int
    entry: np.ndarray
    target: np.ndarray
    readout: tuple  # source bit mask or None (unused label after leakage)
    verified_residual: float


class FixedWeightPort:
    """An explicit rank encoding with pre-allocation limits for dense auditing."""

    def __init__(self, n, k, *, seed_mask=None, max_sector_dimension=64, max_full_dimension=256):
        self.n, self.k = _integer(n, "n"), _integer(k, "k")
        max_sector_dimension = _integer(max_sector_dimension, "max_sector_dimension")
        max_full_dimension = _integer(max_full_dimension, "max_full_dimension")
        if self.n < 1 or not 0 <= self.k <= self.n:
            raise ValueError("Require n >= 1 and 0 <= k <= n")
        if max_full_dimension < 1 or self.n >= max_full_dimension.bit_length():
            raise ValueError("Full-register dimension exceeds the dense audit budget")
        self.dimension = math.comb(self.n, self.k)
        if self.dimension > max_sector_dimension:
            raise ValueError("Fixed-weight sector exceeds the dense synthesis budget")
        self.width = (self.dimension - 1).bit_length()
        self.labels = tuple(sorted(sum(1 << q for q in bits) for bits in combinations(range(self.n), self.k)))
        self._rank = {mask: rank for rank, mask in enumerate(self.labels)}
        self.seed_mask = self.labels[0] if seed_mask is None else _integer(seed_mask, "seed_mask")
        if self.seed_mask not in self._rank:
            raise ValueError("The source seed must have the declared weight")

    def encode(self, mask):
        """Rank a feasible source mask."""
        word = _integer(mask, "mask")
        if word not in self._rank:
            raise ValueError("Cannot encode a mask outside the fixed-weight sector")
        return self._rank[word]

    def decode(self, rank):
        """Decode a measured compact word; unused padded words have no feasible image."""
        y = _integer(rank, "rank")
        if not 0 <= y < 1 << self.width:
            raise ValueError("Readout word exceeds the compact physical register")
        return self.labels[y] if y < self.dimension else None

    def _basis(self, physical_width, rows):
        result = np.zeros((1 << physical_width, self.dimension))
        result[list(rows), np.arange(self.dimension)] = 1.0
        return result

    def exchange(self, i, j, angle):
        """Restricted exp[-i angle (XX+YY)/2] in increasing mask order."""
        i, j, angle = _integer(i, "wire"), _integer(j, "wire"), _angle(angle)
        if i == j or i not in range(self.n) or j not in range(self.n):
            raise ValueError("XY exchange requires two distinct source wires in range")
        out = np.eye(self.dimension, dtype=complex)
        for rank, mask in enumerate(self.labels):
            if ((mask >> i) ^ (mask >> j)) & 1:
                out[rank, rank] = np.cos(angle)
                out[self._rank[mask ^ (1 << i) ^ (1 << j)], rank] = -1j * np.sin(angle)
        return out

    def bind(self, preparation, layers):
        """Compose the preparation exchanges, diagonal cost phases and ordered mixers."""
        value = np.eye(self.dimension, dtype=complex)
        for exchange in preparation:
            value = self.exchange(*exchange) @ value
        for layer in layers:
            energies = np.asarray([_angle(layer.energy(mask)) for mask in self.labels])
            phases = np.exp(-1j * _angle(layer.gamma) * energies)
            if not np.isfinite(phases).all():
                raise ValueError("The scaled cost phases exceed numerical range")
            value = phases[:, None] * value
            for exchange in layer.exchanges:
                value = self.exchange(*exchange) @ value
        return value

    def implementations(self, source_qasm, value, *, tolerance=1e-9):
        """Return verified original, same-O3-original and compact physical choices.

        Requires the host Qiskit synthesis runtime. A one-dimensional sector
        needs zero quantum instructions; its compact QASM is None. Readout
        returns the single valid source word. Dense synthesis is bounded by the
        constructor budgets. A fixed seed other than the first label is moved
        to compact physical word zero by a reversible input XOR.
        """
        from qiskit import QuantumCircuit, qasm2, transpile
        from qiskit.quantum_info import Operator

        value = np.asarray(value, dtype=complex)
        if value.shape != (self.dimension, self.dimension) or not np.isfinite(value).all():
            raise ValueError("The bound unitary does not match the sector")
        if not math.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError("Require 0 < tolerance < 1")
        if np.max(abs(value.conj().T @ value - np.eye(self.dimension))) > tolerance:
            raise ValueError("The proposed fixed-weight evolution is not unitary")
        source = qasm2.loads(source_qasm)
        if source.num_qubits != self.n:
            raise ValueError("Source QASM width differs from the declared interface")
        physical_entry = self._basis(self.n, (mask ^ self.seed_mask for mask in self.labels))
        target = self._basis(self.n, self.labels) @ value
        original = Operator(source).data
        original_residual = _verify(original @ physical_entry, target, tolerance)
        options = dict(basis_gates=["cz", "u3"], optimization_level=3, seed_transpiler=0)
        optimized = transpile(source, **options)
        optimized_unitary = Operator(optimized).data
        _verify(optimized_unitary, original, tolerance)
        optimized_residual = _verify(optimized_unitary @ physical_entry, target, tolerance)
        readout = tuple(range(1 << self.n))
        candidates = {
            "original": FixedWeightImplementation(source_qasm, self.n, physical_entry, target, readout, original_residual),
            "original_o3": FixedWeightImplementation(qasm2.dumps(optimized), self.n, physical_entry, target, readout, optimized_residual),
        }

        size = 1 << self.width
        seed_rank = self.encode(self.seed_mask)
        compact_entry = self._basis(self.width, (rank ^ seed_rank for rank in range(self.dimension)))
        compact_target = self._basis(self.width, range(self.dimension)) @ value
        compact_qasm = None
        if self.width:
            padded = np.eye(size, dtype=complex)
            padded[:self.dimension, :self.dimension] = value
            if seed_rank:
                padded = padded[:, [rank ^ seed_rank for rank in range(size)]]
            circuit = QuantumCircuit(self.width)
            circuit.unitary(padded, list(range(self.width)))
            native = transpile(circuit, **options)
            compact_qasm = qasm2.dumps(native)
            compact = Operator(native).data
        else:
            compact = np.ones((1, 1), complex)
        error = _verify(compact @ compact_entry, compact_target, tolerance)
        candidates["joint"] = FixedWeightImplementation(compact_qasm, self.width, compact_entry, compact_target,
                                                       tuple(self.decode(y) for y in range(size)), error)
        return candidates
