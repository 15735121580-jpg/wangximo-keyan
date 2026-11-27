"""Compile a basis-input unitary preparation for its declared Pauli readouts."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path


# Load the same lowering and erasure pass regardless of caller import path.
_host_spec = importlib.util.spec_from_file_location(
    "_backward_pauli_host", Path(__file__).resolve().parents[1] / "unseen" / "optimizer.py"
)
host = importlib.util.module_from_spec(_host_spec)
_host_spec.loader.exec_module(host)

METRICS = ("two_qubit_count", "instruction_count", "depth", "allocated_sites")


class OutsideSector(ValueError):
    """A requested Pauli sends every occupied basis state outside the orbit."""


def _masks(word):
    x = sum(1 << q for q, p in enumerate(word) if p in "XY")
    z = sum(1 << q for q, p in enumerate(word) if p in "ZY")
    return x, z


def _word(x, z, n):
    return "".join("IXZY"[((x >> q) & 1) + 2 * ((z >> q) & 1)] for q in range(n))


def _pull(word, frame):
    """For the gates in frame, return the sign and unsigned word of C† P C."""
    sign = 1
    for gate in reversed(frame):
        qs = gate["qubits"]
        local = "".join(word[q] for q in qs)
        positive, image = host.TABLES[gate["op"]][local]
        sign *= 1 if positive else -1
        word = host.scatter(word, qs, image)
    return sign, word


def _eliminate(vector, pivots):
    """Return echelon remainder and coefficients accumulated in the input basis."""
    coefficients = 0
    for bit in sorted(pivots, reverse=True):
        if vector & (1 << bit):
            row, coords = pivots[bit]
            vector ^= row
            coefficients ^= coords
    return vector, coefficients


def _basis(vectors):
    basis = []
    pivots = {}
    for vector in vectors:
        rest, coords = _eliminate(vector, pivots)
        if rest:
            pivot = rest.bit_length() - 1
            pivots[pivot] = rest, coords ^ (1 << len(basis))
            basis.append(vector)
    return basis, pivots


def _echelon_basis(vectors):
    _, pivots = _basis(vectors)
    # Clear all lower pivot bits so distinct generators have private pivot sites.
    rows = {bit: row for bit, (row, _) in pivots.items()}
    for low in sorted(rows):
        for high in rows:
            if high > low and rows[high] & (1 << low):
                rows[high] ^= rows[low]
    return _basis([rows[k] for k in sorted(rows)])


def _weight(mask):
    return bin(mask).count("1")


def _restrict(sign, word, offset, basis, pivots):
    x, z = _masks(word)
    rest, u = _eliminate(x, pivots)
    if rest:
        raise OutsideSector("readout flip mask is outside the rotation-generator span")
    v = sum(((_weight(b & z) & 1) << i) for i, b in enumerate(basis))
    phase = _weight(x & z) - _weight(u & v) + 2 * _weight(offset & z)
    if phase % 2:
        raise ArithmeticError("Non-Hermitian restriction of a Hermitian Pauli")
    return sign * (-1 if phase % 4 else 1), _word(u, v, len(basis))


def compress(task, basis_policy="generator"):
    n = task["n_qubits"]
    frame = []
    rotations = []
    for gate in task["gates"]:
        if gate["op"] in host.TABLES:
            frame.append(gate)
        else:
            local = host.ROTATIONS[gate["op"]] or gate["pauli"]
            word = host.scatter("I" * n, gate["qubits"], local)
            sign, word = _pull(word, frame)
            rotations.append((gate["angle"], sign, word))
    vectors = [_masks(word)[0] for _, _, word in rotations]
    if basis_policy == "generator":
        basis, pivots = _basis(vectors)
    elif basis_policy == "echelon":
        basis, pivots = _echelon_basis(vectors)
    else:
        raise ValueError("Unknown affine basis policy")
    offset = sum(bit << q for q, bit in enumerate(task["initial_bits"]))
    gates = []
    for angle, sign, word in rotations:
        sign, image = _restrict(sign, word, offset, basis, pivots)
        qs = [q for q, p in enumerate(image) if p != "I"]
        if not qs:
            continue  # Retains only an unobservable binding-dependent global phase.
        if sign == -1:
            angle = "-(" + angle + ")" if isinstance(angle, str) else -angle
        gates.append({"op": "pauli_rotation", "qubits": qs, "angle": angle,
                      "pauli": "".join(image[q] for q in qs)})
    obs, readout_signs = [], []
    for observable in task["observables"]:
        terms, signs = [], []
        for term in observable["terms"]:
            sign, word = _pull(term["pauli"], frame)
            sign, image = _restrict(sign, word, offset, basis, pivots)
            terms.append({"pauli": image, "coeff": sign * term["coeff"]})
            signs.append(sign)
        obs.append({"label": observable["label"], "terms": terms})
        readout_signs.append(signs)
    proof = {"operation": "affine_sector", "policy": basis_policy, "input_width": n,
             "initial_bits": list(task["initial_bits"]), "rank": len(basis),
             "basis_columns_q0_first": ["".join(str((b >> q) & 1) for q in range(n))
                                         for b in basis],
             "terminal_clifford_frame": copy.deepcopy(frame),
             "readout_signs": readout_signs}
    return {"n_qubits": len(basis), "initial_bits": [0] * len(basis),
            "gates": gates, "observables": obs}, proof


def prune(task):
    # Seed even zero-coefficient terms, since callers may request their samples.
    weighted = dict(task, observables=[
        {"label": obs["label"], "terms": [dict(t, coeff=1) for t in obs["terms"]]}
        for obs in task["observables"]])
    reduced, wires, erasures = host.rewrite(weighted)
    for new_obs, old_obs in zip(reduced["observables"], task["observables"]):
        for term, original in zip(new_obs["terms"], old_obs["terms"]):
            term["coeff"] = original["coeff"]
    return reduced, {"operation": "backward_pruning", "source_wires": wires,
                     "local_decisions": erasures}


def _cost(native):
    return {k: native[k] for k in METRICS}


def compile_task(task, device):
    """Return a native Pareto-safe selection and a replayable signed readout map."""
    # Reject explicitly declared semantics outside the unitary Pauli ABI.
    unsupported = {"initial_state", "noise", "noise_model", "channels", "measurements",
                   "leakage", "postselection", "postselect", "readout_povm"}
    if set(task) & unsupported:
        raise ValueError("Unsupported preparation or measurement model")
    for observable in task["observables"]:
        if set(observable) - {"label", "terms", "metadata", "id"}:
            raise ValueError("Unsupported observable fields")
        for term in observable["terms"]:
            if set(term) - {"pauli", "coeff", "metadata", "id"}:
                raise ValueError("Unsupported observable term fields")
    for gate in task["gates"]:
        keys = {"op", "qubits", "label", "metadata", "id"}
        if gate["op"] in host.ROTATIONS:
            keys.add("angle")
        if gate["op"] == "pauli_rotation":
            keys.add("pauli")
        if set(gate) - keys:
            raise ValueError("Unsupported gate fields: " + str(sorted(set(gate) - keys)))
    graph = host.validate(task, device)
    original = host.native_program(task, device, graph)
    reference_costs = _cost(original)
    best, program, best_cert, selected = task, original, [], "reference"
    candidates = [{"pass": "reference", "admitted": True, "costs": reference_costs}]

    def consider(name, candidate, chain):
        nonlocal best, program, best_cert, selected
        native = host.native_program(candidate, device, graph)
        costs = _cost(native)
        admit = all(costs[k] <= reference_costs[k] for k in METRICS)
        candidates.append({"pass": name, "admitted": admit, "costs": costs})
        if admit and tuple(costs[k] for k in METRICS) < tuple(program[k] for k in METRICS):
            best, program, best_cert, selected = candidate, native, chain, name

    pruned, pc = prune(task)
    consider("prune", pruned, [pc])
    for policy in ("generator", "echelon"):
        for source, chain, prefix in [(task, [], ""), (pruned, [pc], "prune+")]:
            name = prefix + "sector_" + policy
            try:
                small, sc = compress(source, policy)
            except OutsideSector as exc:
                candidates.append({"pass": name, "admitted": False, "reason": str(exc)})
                continue
            consider(name, small, chain + [sc])
            if not prefix:
                further, fc = prune(small)
                consider(name + "+prune", further, [sc, fc])

    sector = next((e for e in best_cert if e["operation"] == "affine_sector"), None)
    signs = (sector["readout_signs"] if sector else
             [[1 for _ in o["terms"]] for o in task["observables"]])
    wires = (None if sector else
             best_cert[0]["source_wires"] if best_cert else list(range(task["n_qubits"])))
    return {"format": "pauli-sector-compiler-v1", "optimized_task": best,
            "reference_native": original, "optimized_native": program,
            "source_wire_for_optimized_index": wires, "readout_signs": signs,
            "selected_pass": selected, "certificate": best_cert, "candidates": candidates,
            "rewrite_admitted": selected != "reference", "trial_native_costs": _cost(program)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("task", type=Path)
    cli.add_argument("device", type=Path)
    cli.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    result = compile_task(json.loads(args.task.read_text()), json.loads(args.device.read_text()))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
