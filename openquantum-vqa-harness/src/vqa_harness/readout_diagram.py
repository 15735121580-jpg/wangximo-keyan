"""A typed preparation/evolution, parity measurement, and classical score diagram."""
from dataclasses import dataclass, replace
from functools import lru_cache
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Operator, Pauli


@dataclass(frozen=True)
class QuantumPort:
    physical_sites: tuple[int, ...]

    def __post_init__(self):
        if (type(self.physical_sites) is not tuple or not self.physical_sites
                or any(type(site) is not int or site < 0 for site in self.physical_sites)
                or len(set(self.physical_sites)) != len(self.physical_sites)):
            raise ValueError("Quantum wires need a nonempty injective physical-site map")


@dataclass(frozen=True)
class Evolution:
    register: QuantumPort
    gates: tuple  # (name, logical wire indices, bound parameters)


@dataclass(frozen=True)
class ParityPort:
    binding: int
    observable: str  # original controller key, retained through a basis change


@dataclass(frozen=True)
class Measurement:
    input: QuantumPort
    word: str  # wire zero first
    sign: int
    output: ParityPort


@dataclass(frozen=True)
class Score:
    input: ParityPort
    weight: float


@dataclass(frozen=True)
class ReadoutDiagram:
    evolution: Evolution
    measurement: Measurement
    score: Score

    def __post_init__(self):
        m = self.measurement
        n = len(m.input.physical_sites)
        if self.evolution.register != m.input or m.output != self.score.input:
            raise ValueError("Mismatched quantum or classical ports")
        if len(m.word) != n or set(m.word)-set("IXYZ") or type(m.sign) is not int or m.sign not in (-1,1):
            raise ValueError("Invalid signed measurement")
        if not math.isfinite(self.score.weight):
            raise ValueError("Invalid controller weight")
        arity = {"cx":2, "cz":2, "h":1, "s":1, "sdg":1, "x":1, "z":1, "sx":1}
        for name, wires, angles in self.evolution.gates:
            if name in arity and (len(wires) != arity[name] or angles):
                raise ValueError("Invalid Clifford primitive")
            if name == "rz" and (len(wires) != 1 or len(angles) != 1):
                raise ValueError("Invalid rotation primitive")
            if not wires or len(set(wires)) != len(wires) or any(q<0 or q>=n for q in wires):
                raise ValueError("Gate addresses an unavailable wire")
            if not all(math.isfinite(a) for a in angles):
                raise ValueError("Bind gate parameters first")

    @property
    def native_input(self):
        return self.evolution.gates, self.measurement.word, self.measurement.sign


@lru_cache(None)
def _pull_clifford(name, local):
    circuit = QuantumCircuit(len(local))
    getattr(circuit,name)(*range(len(local)))
    before = Pauli(local[::-1])
    pulled = before.evolve(Clifford(circuit),frame="h")
    factor = {0:1,2:-1}[int(pulled.phase)]
    pulled.phase = 0
    # Independently check the signed effect before committing the mapped interface.
    gate = Operator(circuit).data
    if not np.allclose(gate.conj().T @ before.to_matrix() @ gate,
                       factor * pulled.to_matrix(), rtol=0, atol=1e-12):
        raise ValueError("Failed signed effect pullback")
    return pulled.to_label()[::-1], factor


def rewrite_diagram(diagram, recipe):
    """Return the composed diagram after each registered P, D or R step.

    P transports the effect through a terminal Clifford suffix. R also drops a
    terminal Rz commuting with the current I/Z factor. D discards gates on
    wires disjoint from the backward support of every retained downstream gate
    and the terminal effect. The binary output key and its consumer are fixed.
    """
    for rule in recipe:
        body, word, sign = diagram.native_input
        if rule == "D":
            support = {i for i,s in enumerate(word) if s!="I"}
            kept = []
            for item in reversed(body):
                if support.intersection(item[1]):
                    kept.append(item)
                    support.update(item[1])
            body = tuple(reversed(kept))
        elif rule in ("P","R"):
            letters = list(word)
            keep = len(body)
            for name, wires, _ in reversed(body):
                if name in ("cx","cz","h","s","sdg","x","z","sx"):
                    after, factor = _pull_clifford(name,"".join(letters[q] for q in wires))
                    for q, axis in zip(wires,after):
                        letters[q] = axis
                    sign *= factor
                elif rule == "R" and name == "rz" and letters[wires[0]] in ("I","Z"):
                    pass
                else:
                    break
                keep -= 1
            body, word = body[:keep], "".join(letters)
        else:
            raise ValueError("Unknown readout diagram rule")
        diagram = replace(diagram,evolution=replace(diagram.evolution,gates=body),
                          measurement=replace(diagram.measurement,word=word,sign=sign))
    return diagram


def bound_readout(sites, gates, binding, word, weight):
    register = QuantumPort(tuple(sites))
    binary = ParityPort(binding, word)
    return ReadoutDiagram(Evolution(register,tuple(gates)),Measurement(register,word,1,binary),
                          Score(binary,weight))
