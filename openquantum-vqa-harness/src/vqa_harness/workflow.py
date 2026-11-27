"""Compile and execute finite-grid VQE jobs on an explicit local device model."""
from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import math
import operator
from importlib.metadata import version
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, qasm2
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import DensityMatrix, Kraus, Pauli, SparsePauliOp, Statevector, state_fidelity
from scipy.linalg import eigvalsh

from .compiler.sectors.optimizer import compile_task


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected a finite real number")
    return float(value)


def _angle(expression, parameters):
    """Evaluate arithmetic only; function calls, attributes and indexing are rejected."""
    if type(expression) in (int, float):
        return _number(expression)
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow}

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            return _number(node.value)
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return math.pi
            if node.id not in parameters:
                raise ValueError("Unbound parameter: " + node.id)
            return _number(parameters[node.id])
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.USub, ast.UAdd):
            return _number((-1 if isinstance(node.op, ast.USub) else 1) * visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 16:
                raise ValueError("Angle exponent exceeds the arithmetic budget")
            try:
                return _number(binary[type(node.op)](left, right))
            except (ZeroDivisionError, OverflowError) as error:
                raise ValueError("Invalid angle arithmetic") from error
        raise ValueError("Angles permit finite arithmetic and named parameters only")

    if not isinstance(expression, str) or len(expression) > 2048:
        raise ValueError("Invalid angle expression")
    try:
        return visit(ast.parse(expression, mode="eval"))
    except SyntaxError as error:
        raise ValueError("Invalid angle expression") from error


def _circuit(instructions, width, parameters):
    circuit = QuantumCircuit(width)
    for gate in instructions:
        op, qs = gate["op"], gate["qubits"]
        if op == "pauli_rotation":
            # Input words are q0 first; Qiskit words are most significant first.
            circuit.append(PauliEvolutionGate(Pauli(gate["pauli"][::-1]),
                           time=_angle(gate["angle"], parameters) / 2), qs)
        elif op in ("rx", "ry", "rz"):
            getattr(circuit, op)(_angle(gate["angle"], parameters), *qs)
        else:
            getattr(circuit, op)(*qs)
    return circuit


def _logical_state(task, parameters):
    initial = [{"op": "x", "qubits": [q]} for q, b in enumerate(task["initial_bits"]) if b]
    return Statevector.from_instruction(_circuit(initial + task["gates"], task["n_qubits"], parameters))


def _probability(value):
    value = float(value)
    if not math.isfinite(value) or not -1e-10 <= value <= 1 + 1e-10:
        raise ValueError("Invalid measurement probability")
    return min(1., max(0., value))


def _basis_instructions(word):
    instructions = []
    for q, pauli in enumerate(word):
        if pauli == "Y":
            instructions.append({"op": "rz", "qubits": [q], "angle": -math.pi / 2})
        if pauli in "XY":
            instructions.extend([{"op": "rz", "qubits": [q], "angle": math.pi / 2},
                                 {"op": "sx", "qubits": [q]},
                                 {"op": "rz", "qubits": [q], "angle": math.pi / 2}])
    return instructions


def _channel(arity, fidelity):
    dimension = 2 ** arity
    strength = dimension * (1 - fidelity) / (dimension - 1)
    # E(rho) = (1-strength) rho + strength I/d. The Pauli mixture
    # implements this channel, with F_average = 1 - strength*(d-1)/d.
    weights = [strength / dimension ** 2] * dimension ** 2
    weights[0] += 1 - strength
    if min(weights) < -1e-12:
        raise ValueError("Average gate fidelity does not define a depolarizing channel")
    return Kraus([math.sqrt(max(0., p)) * Pauli("".join(word)).to_matrix()
                  for p, word in zip(weights, itertools.product("IXYZ", repeat=arity)) if p > 0])


def _evolve(instructions, width, parameters, channels=None):
    circuit = _circuit(instructions, width, parameters)
    if channels is None:
        return Statevector.from_instruction(circuit)
    state = DensityMatrix.from_label("0" * width)
    for item in circuit.data:
        qs = [circuit.find_bit(q).index for q in item.qubits]
        state = state.evolve(item.operation, qargs=qs)
        state = state.evolve(channels[len(qs)], qargs=qs)
    return state


def _resources(instructions, word, native, device):
    support = [q for q, p in enumerate(word) if p != "I"]
    if not support:
        return {"gate_count": 0, "two_qubit_count": 0, "peak_qubits": 0,
                "depth_events": 0, "duration_us": 0.}
    active = set(native["logical_to_physical"]) | set(support)
    active.update(q for g in instructions for q in g["qubits"])
    ends, layers = [0.] * device["n_qubits"], [0] * device["n_qubits"]
    for gate in instructions:
        qs = gate["qubits"]
        end = max(ends[q] for q in qs) + device["gate_times_us"][gate["op"]]
        layer = max(layers[q] for q in qs) + 1
        for q in qs:
            ends[q], layers[q] = end, layer
    return {"gate_count": len(instructions),
            "two_qubit_count": sum(len(g["qubits"]) == 2 for g in instructions),
            "peak_qubits": len(active), "depth_events": max(layers, default=0) + 2,
            "duration_us": device["reset_us"] + max(ends, default=0.) + device["readout_us"]}


def _setting(native, term, parameters, device, channels):
    word = term["pauli"]
    if set(word) <= {"I"}:
        return {"coefficient": term["coeff"], "pauli_q0_first": word,
                "ideal_expectation": 1., "noisy_expectation": 1., "positive_probability": 1.,
                "state_fidelity_to_own_ideal": None,
                "execution": "deterministic_identity_no_device_call",
                "per_hardware_shot": _resources([], word, native, device), "native_instructions": []}
    instructions = native["instructions"] + _basis_instructions(word)
    width = device["n_qubits"]
    ideal = _evolve(instructions, width, parameters)
    noisy = _evolve(instructions, width, parameters, channels)
    support_mask = sum(1 << q for q, p in enumerate(word) if p != "I")
    positive = np.array([(i & support_mask).bit_count() % 2 == 0 for i in range(2 ** width)])
    p = _probability(np.real(noisy.probabilities()[positive].sum()))
    # Each noisy state is compared only with its own ideal implementation.
    return {"coefficient": term["coeff"], "pauli_q0_first": word,
            "ideal_expectation": float(np.real(ideal.expectation_value(Pauli(
                "".join("Z" if p != "I" else "I" for p in word)[::-1])))),
            "noisy_expectation": 2*p - 1, "positive_probability": p,
            "execution": "native_preparation_basis_change_and_measurement",
            "state_fidelity_to_own_ideal": float(state_fidelity(ideal, noisy)),
            "per_hardware_shot": _resources(instructions, word, native, device),
            "native_instructions": instructions}


def _measure(settings, shots, rng, totals):
    records, energy, variance = [], 0., 0.
    for setting in settings:
        outcomes = np.where(rng.random(shots) < setting["positive_probability"], 1, -1)
        coeff = setting["coefficient"]
        energy += coeff * float(outcomes.mean())
        variance += coeff ** 2 * (float(outcomes.var(ddof=1)) / shots if shots > 1 else 0.)
        record = {"pauli_q0_first": setting["pauli_q0_first"], "coefficient": coeff,
                  "shots": shots, "positive_count": int((outcomes == 1).sum()),
                  "outcomes": outcomes.tolist()}
        records.append(record)
        resource = setting["per_hardware_shot"]
        physical = bool(resource["peak_qubits"])
        totals["requested_pauli_samples"] += shots
        totals["hardware_shots"] += shots if physical else 0
        totals["deterministic_samples"] += 0 if physical else shots
        for key in ("gate_count", "two_qubit_count", "depth_events", "duration_us"):
            totals[key] += shots * resource[key]
        totals["peak_qubits"] = max(totals["peak_qubits"], resource["peak_qubits"])
    stochastic = [s for s in settings if s["per_hardware_shot"]["peak_qubits"]]
    radius = sum(abs(s["coefficient"]) for s in stochastic) * math.sqrt(
        2 * math.log(2 * max(1, len(stochastic)) / .05) / shots)
    return {"energy_estimate": energy, "standard_error": math.sqrt(variance),
            "confidence_radius_95_hoeffding": radius,
            "confidence_target": "Expected energy under the supplied noise model, conditional on the selected binding",
            "records": records}


def _validate(job):
    if not isinstance(job, dict) or set(job) - {"task", "device", "vqa", "platform", "name", "metadata"}:
        raise ValueError("Expected task, device, vqa and optional platform, name or metadata")
    if "name" in job and not isinstance(job["name"], str):
        raise ValueError("Job name must be a string")
    if "metadata" in job and not isinstance(job["metadata"], dict):
        raise ValueError("Job metadata must be an object")
    task, device, vqa = job["task"], job["device"], job["vqa"]
    if len(task["observables"]) != 1 or not task["observables"][0]["terms"]:
        raise ValueError("A VQE job requires one nonempty Hamiltonian observable")
    if type(device["n_qubits"]) is not int or not 1 <= device["n_qubits"] <= 8:
        raise ValueError("Local density-matrix budget supports one through eight device qubits")
    for name in ("reset_us", "readout_us"):
        if _number(device[name]) < 0:
            raise ValueError("Physical times must be nonnegative")
    for gate in device["native"]:
        if _number(device["gate_times_us"][gate]) < 0:
            raise ValueError("Physical gate times must be nonnegative")
    for name, dimension in (("one_qubit", 2), ("two_qubit", 4)):
        f = _number(device["gate_fidelities"][name])
        if not 1/(dimension+1) <= f <= 1:
            raise ValueError("Gate fidelity is outside depolarizing-channel bounds")
    for name in ("shots_per_term", "terminal_shots_per_term"):
        if type(vqa[name]) is not int or vqa[name] < 1:
            raise ValueError("Shot budgets must be positive integers")
    if type(vqa["seed"]) is not int or vqa["seed"] < 0:
        raise ValueError("Seed must be a nonnegative integer")
    grid = vqa["parameter_grid"]
    if not isinstance(grid, list) or not grid or any(not isinstance(p, dict) for p in grid):
        raise ValueError("Supply a nonempty parameter grid")
    for params in grid:
        if any(not isinstance(k, str) or not k.isidentifier() or k == "pi" for k in params):
            raise ValueError("Parameter names must be identifiers other than pi")
        for value in params.values():
            _number(value)
        for gate in task["gates"]:
            if "angle" in gate:
                _angle(gate["angle"], params)
    return task, device, vqa


def run_job(job: dict, out: Path) -> dict:
    """Write a fresh, replayable host/joint complete finite-grid VQE comparison."""
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("Output already exists: " + str(out))
    job = copy.deepcopy(job)
    task, device, vqa = _validate(job)
    compilation = compile_task(task, device)  # Compile symbolically before any binding.
    channels = {n: _channel(n, device["gate_fidelities"][key])
                for n, key in ((1, "one_qubit"), (2, "two_qubit"))}
    grid = vqa["parameter_grid"]
    native_arms = {"host": compilation["reference_native"], "joint": compilation["optimized_native"]}
    circuits, checks, arms, ideal_energies = {}, [], {}, []
    original_terms = task["observables"][0]["terms"]
    signs = compilation["readout_signs"][0]
    for params in grid:
        state = _logical_state(task, params)
        ideal_energies.append(sum(t["coeff"] * float(np.real(state.expectation_value(
            Pauli(t["pauli"][::-1])))) for t in original_terms))
    for arm_index, (arm, native) in enumerate(native_arms.items()):
        settings_by_grid = []
        for index, params in enumerate(grid):
            settings = [_setting(native, term, params, device, channels)
                        for term in native["physical_observables"][0]["terms"]]
            source_state = _logical_state(task, params)
            for term_index, (original, setting) in enumerate(zip(original_terms, settings)):
                expected = float(np.real(source_state.expectation_value(Pauli(original["pauli"][::-1]))))
                sign = signs[term_index] if arm == "joint" else 1
                error = abs(expected - sign * setting["ideal_expectation"])
                coefficient_error = abs(setting["coefficient"] - sign * original["coeff"])
                if max(error, coefficient_error) > 1e-9:
                    raise RuntimeError("Independent ideal Pauli equivalence failed")
                checks.append({"arm": arm, "grid_index": index, "term_index": term_index,
                               "absolute_error": error, "coefficient_error": coefficient_error, "passed": True})
            settings_by_grid.append(settings)
        circuits[arm] = settings_by_grid
        totals = {k: 0 for k in ("requested_pauli_samples", "hardware_shots", "deterministic_samples",
                                "gate_count", "two_qubit_count", "peak_qubits", "depth_events", "duration_us")}
        rng = np.random.default_rng(np.random.SeedSequence([vqa["seed"], arm_index]))
        training = [_measure(settings, vqa["shots_per_term"], rng, totals)
                    for settings in settings_by_grid]
        selected = min(range(len(training)), key=lambda i: (training[i]["energy_estimate"], i))
        terminal = _measure(settings_by_grid[selected], vqa["terminal_shots_per_term"], rng, totals)
        settings = settings_by_grid[selected]
        terminal.update(parameters=grid[selected], selected_grid_index=selected,
                        ideal_energy=ideal_energies[selected],
                        modeled_noisy_energy=sum(s["coefficient"]*s["noisy_expectation"] for s in settings))
        fidelities = [s["state_fidelity_to_own_ideal"] for entries in settings_by_grid for s in entries
                      if s["state_fidelity_to_own_ideal"] is not None]
        arms[arm] = {"training": training, "answer": terminal, "resources": totals,
                     "minimum_modeled_state_fidelity": min(fidelities) if fidelities else None}
    hamiltonian = sum((t["coeff"] * Pauli(t["pauli"][::-1]).to_matrix() for t in original_terms))
    ground_energy = float(eigvalsh(hamiltonian, subset_by_index=[0, 0])[0])
    for data in arms.values():
        answer = data["answer"]
        answer["absolute_error_to_full_space_ground"] = abs(answer["energy_estimate"] - ground_energy)
        answer["ideal_ansatz_grid_gap_to_ground"] = min(ideal_energies) - ground_energy
        answer["absolute_error_to_selected_modeled_energy"] = abs(
            answer["energy_estimate"] - answer["modeled_noisy_energy"])
    resource_keys = ("gate_count", "two_qubit_count", "peak_qubits", "depth_events", "duration_us")
    source = Path(__file__).read_bytes()
    receipt = {"schema": "openquantum-vqa-complete-job-v1", "status": "completed_local_model",
        "name": job.get("name", ""), "platform": job.get("platform", {}), "seed": vqa["seed"],
        "noise_contract": {"gate_fidelities": device["gate_fidelities"],
                           "reset": "ideal", "readout": "ideal", "memory": "ideal",
                           "channel": "Local depolarization after every native gate, including basis changes"},
        "versions": {name: version(name) for name in ("qiskit", "numpy", "scipy")},
        "workflow_sha256": hashlib.sha256(source).hexdigest(),
        "input_sha256": hashlib.sha256(json.dumps(job, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "verification": {"all_ideal_pauli_checks_passed": True, "checks": checks,
                         "max_ideal_equivalence_residual": max(c["absolute_error"] for c in checks)},
        "reference": {"exact_ground_energy": ground_energy, "ideal_grid_energies": ideal_energies,
                      "ideal_grid_minimum": min(ideal_energies)},
        "arms": arms, "answer": arms["joint"]["answer"],
        "comparison": {"joint_minus_host": {k: arms["joint"]["resources"][k] - arms["host"]["resources"][k]
                        for k in resource_keys},
                       "all_resource_metrics_nonregressing": all(arms["joint"]["resources"][k] <=
                           arms["host"]["resources"][k] + 1e-10 for k in resource_keys)},
        "scope": {"optimization": "Finite grid controller and compiler-generated affine/pruning candidates; no global optimum claim",
                  "noise": "Independent native gate depolarization; ideal reset, readout and memory; all fidelities are supplied model assumptions",
                  "timing": "Serial shots and settings, ASAP disjoint-gate scheduling, parallel reset/readout; excludes offline search",
                  "fidelity": "State fidelity to each arm's own ideal circuit; no cross-width or process-fidelity claim",
                  "ground_reference": "Independent full-space ground energy; the declared preparation, ansatz and finite grid may not attain it",
                  "logical_error_rate": "Not applicable: this local unencoded model has no fault-tolerant code or logical decoder",
                  "identity_readouts": "Exact deterministic records use zero hardware shots at the same requested sample budget",
                  "statistics": "One seeded trajectory per arm; observed differences do not establish statistical superiority"}}
    out.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    save("input.json", job)
    save("compilation.json", compilation)
    save("circuits.json", circuits)
    for arm, native in native_arms.items():
        folder = out / arm / "terminal"
        folder.mkdir(parents=True)
        selected = arms[arm]["answer"]["selected_grid_index"]
        for i, setting in enumerate(circuits[arm][selected]):
            circuit = QuantumCircuit(device["n_qubits"])
            support = [q for q, p in enumerate(setting["pauli_q0_first"]) if p != "I"]
            if support:
                active = set(native["logical_to_physical"]) | set(support)
                active.update(q for g in setting["native_instructions"] for q in g["qubits"])
                circuit.reset(sorted(active))
                circuit.compose(_circuit(setting["native_instructions"], device["n_qubits"], grid[selected]), inplace=True)
                circuit.add_register(ClassicalRegister(len(support), "readout"))
                circuit.measure(support, range(len(support)))
            qasm = qasm2.dumps(circuit)
            if "sx " in qasm and "gate sx " not in qasm:
                # sx is absent from the original qelib1.inc; define it up to
                # an irrelevant global phase for portable OpenQASM 2 readers.
                qasm = qasm.replace('include "qelib1.inc";',
                    'include "qelib1.inc";\ngate sx q { u3(pi/2,-pi/2,pi/2) q; }')
            (folder / f"term_{i:03d}.qasm").write_text(qasm + "\n")
    receipt["artifacts"] = {name: str(out / name) for name in (
        "input.json", "compilation.json", "circuits.json", "strategy.md", "replay.py", "result.json")}
    receipt["artifacts"]["terminal_qasm"] = {
        arm: [str(path) for path in sorted((out / arm / "terminal").glob("*.qasm"))]
        for arm in native_arms}
    (out / "strategy.md").write_text(
        "# Executed complete VQE strategy\n\nCompile the symbolic preparation and terminal Pauli terms using "
        "the supplied device. Compare the original and admitted affine/pruned native programs. "
        "Evaluate every declared parameter binding using separate Pauli measurements at the fixed training budget; "
        "choose the lowest sampled energy, breaking ties by grid order, and collect a fresh terminal batch.\n\n"
        f"Selected compiler pass: `{compilation['selected_pass']}`. Joint terminal parameters: "
        f"`{json.dumps(receipt['answer']['parameters'], sort_keys=True)}`.\n\n"
        "Native measurement basis changes, reset, readout and every hardware shot are counted. "
        "Identity observables are exact classical records. Noise is evaluated after the ideal rewrite certificate; "
        "noise equivalence and hardware-wide optimality are not claimed. See result.json for model assumptions, "
        "resources, individual records and the complete answer.\n")
    (out / "replay.py").write_text(
        '"""Replay this input using the installed openquantum-vqa-harness package."""\n'
        'import argparse\nimport json\nfrom pathlib import Path\nfrom vqa_harness.workflow import run_job\n'
        'p = argparse.ArgumentParser()\np.add_argument("--out", type=Path, required=True)\na = p.parse_args()\n'
        'r = run_job(json.loads((Path(__file__).parent / "input.json").read_text()), a.out)\n'
        'print(json.dumps({"status": r["status"], "answer": r["answer"]}, indent=2))\n')
    save("result.json", receipt)
    return receipt
