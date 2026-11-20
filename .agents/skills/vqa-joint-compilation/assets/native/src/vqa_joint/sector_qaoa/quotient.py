"""Exact +1 global-X sector for a measured, plus-initialized MaxCut QAOA call."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FlipQuotient:
    width: int
    reference: int

    def __post_init__(self):
        if self.width < 2 or not 0 <= self.reference < self.width:
            raise ValueError("Expected at least two spins and a valid reference")

    @property
    def others(self):
        return tuple(i for i in range(self.width) if i != self.reference)

    def pair(self, y):
        """The reference-zero representative and its complement, little endian."""
        if not 0 <= y < (1 << (self.width - 1)):
            raise ValueError("Outcome outside reduced register")
        z = sum(((y >> j) & 1) << wire for j, wire in enumerate(self.others))
        return z, z ^ ((1 << self.width) - 1)

    def lift_distribution(self, native):
        if len(native) != 1 << (self.width - 1):
            raise ValueError("Probability vector has wrong shape")
        full = np.empty(1 << self.width)
        for y, p in enumerate(native):
            a, b = self.pair(y)
            full[a] = full[b] = p / 2
        return full

    def isometry(self):
        V = np.zeros((1 << self.width, 1 << (self.width - 1)))
        for y in range(V.shape[1]):
            V[list(self.pair(y)), y] = 1 / math.sqrt(2)
        return V

    def layers(self, gates):
        wires = self.others
        reduced = []
        for name, targets, angle in gates:
            if not math.isfinite(angle) or any(t not in range(self.width) for t in targets):
                raise ValueError("Invalid rotation parameters or operands")
            if name == "rzz" and len(set(targets)) == len(targets) == 2:
                survivors = tuple(wires.index(t) for t in targets if t != self.reference)
                reduced.append(("rzz" if len(survivors) == 2 else "rz", survivors, angle))
            elif name == "rx" and len(targets) == 1:
                if targets[0] != self.reference:
                    reduced.append(("rx", (wires.index(targets[0]),), angle))
                    continue
                # Conjugate X_0 by a CNOT fanout; all m-1 computes and uncomputes are charged.
                fanout = [("cx", (0, j), None) for j in range(1, len(wires))]
                reduced.extend([*fanout, ("rx", (0,), angle), *reversed(fanout)])
            else:
                raise ValueError(f"Unsupported QAOA evolution {name} {targets}")
        return reduced

    def compile(self, circuit):
        n = self.width
        expected_h = [("h", (q,), None) for q in range(n)]
        expected_m = [("measure", (q,), None) for q in range(n)]
        if circuit[:n] != expected_h or circuit[-n:] != expected_m:
            raise ValueError("Expected all-plus preparation and full terminal Z readout")
        return expected_h[:-1] + self.layers(circuit[n:-n]) + expected_m[:-1]
