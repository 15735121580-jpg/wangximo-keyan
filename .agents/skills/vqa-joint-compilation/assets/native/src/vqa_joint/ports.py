"""Naturality of qubit permutations through preparation, native code, and readout."""
from dataclasses import dataclass
from itertools import permutations
from vqa_joint.rewrite import Op


@dataclass(frozen=True)
class WirePort:
    # Logical wire i is stored on native input/output slot physical_of_logical[i].
    physical_of_logical: tuple

    def __post_init__(self):
        if sorted(self.physical_of_logical) != list(range(len(self.physical_of_logical))):
            raise ValueError("Qubit ports must form a permutation")
    def native_word(self, word):
        return [Op(op.name, tuple(self.physical_of_logical[i] for i in op.wires),op.angle)
                for op in word]

    def logical_outcome(self, sample):
        if sample < 0 or sample >= 1 << len(self.physical_of_logical):
            raise ValueError("Outcome outside quantum register")
        return sum(((sample >> phys) & 1) << i for i,phys in enumerate(self.physical_of_logical))

    def native_outcome(self, sample):
        if sample < 0 or sample >= 1 << len(self.physical_of_logical):
            raise ValueError("Outcome outside quantum register")
        return sum(((sample >> i) & 1) << phys for i,phys in enumerate(self.physical_of_logical))

    def decode_counts(self, counts):
        return {self.logical_outcome(b): count for b,count in counts.items()}


def port_candidates(n, exhaustive=False):
    if n <= 0:
        raise ValueError("Register must be nonempty")
    wires = tuple(range(n))
    if exhaustive:
        return (WirePort(p) for p in permutations(wires))
    options = [wires, wires[::-1], wires[1:]+wires[:1], wires[::2]+wires[1::2]]
    return (WirePort(p) for p in dict.fromkeys(options))
