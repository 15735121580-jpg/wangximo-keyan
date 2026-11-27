"""Trusted, exact verifier for one scheduled native Boolean phase oracle."""
import itertools
from .guarded_release import Assessment, ReleaseContract, guarded_release


class ExactPhaseVerifier:
    model = "native-phase-anf-v1"
    metrics = {"gates", "depth", "active_wires"}
    arity = {"X": 1, "Xdg": 1, "Z": 1, "CZ": 2, "CCZ": 3}

    def __init__(self, target):
        if not isinstance(target, dict) or set(target) != {"wire_count", "supports"}:
            raise ValueError("Specify the exact phase target using wire_count and supports")
        self.width = target["wire_count"]
        terms = target["supports"]
        if type(self.width) is not int or self.width <= 0 or type(terms) is not list:
            raise ValueError("Invalid phase target width or support list")
        if any(not self.valid_wires(s) or not 1 <= len(s) <= 3 for s in terms):
            raise ValueError("Invalid phase target support")
        self.terms = {tuple(sorted(s)) for s in terms}
        if len(self.terms) != len(terms):
            raise ValueError("Expected a reduced phase target")

    def valid_wires(self, wires):
        return isinstance(wires, (tuple, list)) and all(type(q) is int and
            0 <= q < self.width for q in wires) and len(set(wires)) == len(wires)

    def assess(self, contract, policy):
        if policy is None:
            return Assessment("missing")
        if not isinstance(policy, dict) or set(policy) != {"wire_count", "gates"}:
            return Assessment("invalid")
        if type(policy["wire_count"]) is not int or policy["wire_count"] != self.width:
            return Assessment("invalid")
        gates = policy["gates"]
        if not isinstance(gates, list):
            return Assessment("invalid")
        flips, terms, last = set(), set(), {}
        quarter_turns = 0
        for pos, g in enumerate(gates, 1):
            if not isinstance(g, dict) or set(g) != {"position", "layer", "gate", "qubits"}:
                return Assessment("invalid")
            name, wires, layer = g["gate"], g["qubits"], g["layer"]
            if (type(name) is not str or name not in self.arity or not self.valid_wires(wires)
                    or len(wires) != self.arity[name] or type(g["position"]) is not int
                    or g["position"] != pos or type(layer) is not int or layer <= 0):
                return Assessment("invalid")
            if any(last.get(q, 0) >= layer for q in wires):
                return Assessment("infeasible", "wire collision or reversed dependency")
            last.update((q, layer) for q in wires)
            if name in ("X", "Xdg"):
                flips.symmetric_difference_update(wires)
                quarter_turns += 1 if name == "Xdg" else -1
            else:
                fixed = set(wires) - flips
                changing = set(wires) & flips
                for k in range(len(changing) + 1):
                    for subset in itertools.combinations(changing, k):
                        terms.symmetric_difference_update([tuple(sorted(fixed.union(subset)))])
        # Empty monomial contributes a scalar -1, which can cancel two quarter turns.
        if () in terms:
            terms.remove(())
            quarter_turns += 2
        if flips or quarter_turns % 4 or terms != self.terms:
            return Assessment("infeasible", "different signed operator")
        metrics = dict(gates=len(gates), depth=max(last.values(), default=0), active_wires=len(last))
        if any(k not in metrics for k, _ in contract.caps):
            return Assessment("missing", "unknown cap metric")
        if any(metrics[k] > limit for k, limit in contract.caps):
            return Assessment("infeasible", metrics)
        return Assessment("feasible", metrics)

    def upper_bounds(self, contract, original, proposal):
        return {k: proposal.evidence[k] - original.evidence[k]
                for k in contract.losses if k in self.metrics}


def phase_release(packet):
    keys = {"model", "losses", "caps", "target", "host", "candidates"}
    if not isinstance(packet, dict) or set(packet) != keys:
        raise ValueError("Supply only model, losses, caps, target, host and candidates")
    if (type(packet["losses"]) is not list or type(packet["caps"]) is not dict
            or type(packet["candidates"]) is not list):
        raise ValueError("Expected ordered loss and candidate lists and a resource-cap object")
    contract = ReleaseContract(packet["model"], tuple(packet["losses"]), tuple(packet["caps"].items()))
    return guarded_release(contract, packet["host"], packet["candidates"], ExactPhaseVerifier(packet["target"]))
