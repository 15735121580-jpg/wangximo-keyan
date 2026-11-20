"""Called by native.py inside the corresponding pinned host environment."""
from collections import Counter, deque
from pathlib import Path
import json
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


def load_json(file):
    return json.loads(Path(file).read_text())


def write_json(file, data):
    Path(file).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run_zap(source, profile):
    from native import ZAP_PROFILES
    from zap import zap as module
    arch = load_json(source / "architecture/default.json")
    settings = load_json(source / "setting/tqe.json")
    arch.setdefault("routing", {}).update(settings.get("routing_cfg", {}))
    capture = capture_transpilation(module)
    schedule, route = ZAP_PROFILES[profile]
    compiler = module.Zap(str(Path("call.qasm").absolute()), arch, [], output_dir="call",
                          scheduling_strategy=schedule, routing_strategy=route)
    compiler.solve(simulation=True, animation=False)
    raw = load_json("results/call/code/call_code.json")
    score = load_json("results/call/log/call.json")[-1]
    # Evaluate the original trace using the workspace's independent geometry audit.
    from zap_geometry import inspect
    _, _, problems, _ = inspect(raw, arch)
    one = "1qGate"
    two = "2qGate"
    events = []
    for i, inst in enumerate(raw["instructions"]):
        if inst["type"] == one:
            events += [(i, [q], None) for q in inst["gates"]]
        if inst["type"] == two:
            events += [(i, list(edge), "cz") for edge in inst["gates"]]
    return raw, capture[0], events, {
        "n_data_atoms": raw["n_q"], "duration_us": score["total_duration"],
        "author_model_score": score["total_fidelity"], "rydberg_pulses": sum(
            x["type"] == two for x in raw["instructions"]),
        "transport_batches": sum(x["type"] in {"Park", "BigMove"} for x in raw["instructions"]),
        "geometry_check": "ZAP emitted AOD line, handover, occupancy, partner and gate checks",
        "geometry_findings": dict(problems), "profile":profile, "hardware_profile":"official TQE routing_cfg + default architecture"}


def run_zac(source, profile):
    from native import ZAC_PROFILES
    from zac import zac as module
    from zac.ds.architecture import Architecture
    from zac.simulator.simulator import Simulator
    opts = load_json(source/"exp_setting/comparison_technique.json")["zac_setting"][ZAC_PROFILES[profile]]
    spec_file = source / opts["arch_spec"]
    spec = load_json(spec_file)
    arch = Architecture(spec)
    arch.preprocessing()
    capture = capture_transpilation(module)
    compiler = module.ZAC()
    opts.update(dir="results/", name="call")
    compiler.parse_setting(opts)
    compiler.set_architecture_spec_path(str(spec_file))
    compiler.set_architecture(arch)
    compiler.set_program("call.qasm")
    for name in ("code", "time"):
        (Path("results") / name).mkdir(parents=True)
    compiler.solve(save_file=True)
    raw = load_json(compiler.code_filename)
    sim = Simulator()
    sim.set_arch_spec(spec)
    sim.parse(compiler.code_filename)
    score = sim.simulate()
    events = []
    gates = sorted(raw["instructions"], key=lambda x: (x["begin_time"], x["id"]))
    lookup = {x["id"]: x for x in gates}
    issues = Counter()
    for inst in gates:
        for dep in inst.get("dependency", {}).values():
            for predecessor in ([dep] if isinstance(dep, int) else dep):
                if lookup[predecessor]["end_time"] > inst["begin_time"]+1e-8 or predecessor == inst["id"]:
                    issues["timing_dependency"] += 1
        for gate in inst.get("gates", []):
            if inst["type"] == "1qGate":
                events.append((inst["id"], [gate["q"]], gate["name"]))
            elif inst["type"] == "rydberg":
                events.append((inst["id"], [gate["q0"], gate["q1"]], "cz"))
    return raw, capture[0], events, {
        "n_data_atoms": compiler.n_q, "duration_us": score["cir_duration"],
        "author_model_score": score["cir_fidelity"], "rydberg_pulses": sum(x["type"]=="rydberg" for x in gates),
        "transport_batches": sum(x["type"]=="rearrangeJob" for x in gates),
        "geometry_check":"ZAC built-in scheduling/initial placement and adapter dependency checks; continuous AOD trajectories not independently certified",
        "geometry_findings":dict(issues), "profile":profile, "hardware_profile":"official comparison_technique on full_architecture"}


def capture_transpilation(module):
    captured = []
    original = module.transpile

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result
    module.transpile = record
    return captured


def bind_schedule(orig, word, events):
    # Each occurrence retains its wire-local position. Ordered wire equality also
    # establishes equivalence without a full matrix for larger inputs.
    bound = QuantumCircuit(word.num_qubits)
    pending = [deque() for _ in word.qubits]
    for item in word.data:
        qs = tuple(word.find_bit(q).index for q in item.qubits)
        if len(qs) == 2 and item.operation.name != "cz":
            raise ValueError("Expected host-native CZ")
        for q in qs:
            pending[q].append((item.operation, tuple(sorted(qs))))
    records = []
    ncz = 0
    for slot, qs, name in events:
        heads = [pending[q].popleft() for q in qs]
        op, wire = heads[0]
        if wire != tuple(sorted(qs)) or any(pair[1] != wire for pair in heads):
            raise ValueError("Scheduled gate violates ordered operand dependencies")
        if name is not None and op.name != name:
            raise ValueError("Scheduled gate disagrees with captured host decomposition")
        bound.append(op, qs)
        ncz += int(len(qs) == 2)
        records.append({"instruction":slot, "gate":op.name, "qubits":qs, "params":[float(v) for v in op.params]})
    if any(pending):
        raise ValueError("Missing native gate occurrence")
    bound.global_phase = word.global_phase
    residual = None
    if orig.num_qubits <= 8:
        a, b = Operator(orig).data, Operator(bound).data
        phase = np.vdot(a,b)
        if not abs(phase):
            raise ValueError("Scheduled operator is orthogonal to input")
        residual = float(np.linalg.norm(b - phase/abs(phase)*a, ord=2))
        if residual > 1e-8:
            raise ValueError(f"Scheduled unitary differs from input: {residual}")
    write_json("bound_schedule.json", {"num_qubits":word.num_qubits, "global_phase":float(word.global_phase),
                                        "gates":records})
    return {"n_1q":len(records)-ncz, "n_2q":ncz, "gate_count":len(records),
            "gate_dependency_depth":bound.depth(), "unitary_operator_error":residual,
            "binding_check":"captured author transpilation and per-wire native occurrences"}


def main():
    from native import HOSTS
    from importlib.metadata import version
    host, source, profile = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    sys.path.insert(0, str(source))
    orig = QuantumCircuit.from_qasm_file("call.qasm")
    if len(orig.qregs) != 1 or orig.num_clbits or not orig.data:
        raise ValueError("Provide a nonempty, bound unitary call on a single quantum register")
    bad = {"measure","reset","initialize","barrier","delay","if_else"}
    if any(op.operation.name in bad or getattr(op.operation, "condition", None) is not None
           for op in orig.data):
        raise ValueError("The host bridge accepts unitary operations alone")
    if host == "zap" and orig.data[-1].operation.name == "swap":
        raise ValueError("ZAP silently strips final SWAPs; supply a composed output-frame certificate or decompose before entry")
    raw, word, events, metrics = (run_zap if host=="zap" else run_zac)(source, profile)
    metrics.update(bind_schedule(orig, word, events))
    metrics.update(host=host, qiskit=version("qiskit"), source_revision_target=HOSTS[host][2],
                   scope="one bound logical unitary compiled on unencoded host hardware; no readout, QEC, decoding or adaptive-loop cost")
    write_json("native_trace.json", raw)
    write_json("metrics.json", metrics)


if __name__ == "__main__":
    main()
