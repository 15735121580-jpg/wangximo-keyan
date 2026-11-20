"""Single-storage-site specialization executed in the upstream ZAC environment."""
import json
from importlib.metadata import version
from pathlib import Path
import sys

from qiskit import QuantumCircuit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hosts"))
from native import HOSTS, ZAC_PROFILES
from worker import bind_schedule, capture_transpilation, write_json


def compile_single(source, profile):
    sys.path.insert(0, str(source))
    from zac import zac as module
    from zac.ds.architecture import Architecture
    from zac.simulator.simulator import Simulator

    opts = json.loads((source / "exp_setting/comparison_technique.json").read_text())[
        "zac_setting"
    ][ZAC_PROFILES[profile]]
    spec_path = source / opts["arch_spec"]
    spec = json.loads(spec_path.read_text())
    arch = Architecture(spec)
    arch.preprocessing()
    capture = capture_transpilation(module)
    compiler = module.ZAC()
    compiler.parse_setting(opts)
    compiler.set_architecture_spec_path(str(spec_path))
    compiler.set_architecture(arch)
    compiler.set_program("call.qasm")
    if compiler.n_q != 1 or compiler.g_q or len(capture) != 1:
        raise ValueError(
            "Requires one transpiled data atom and an empty entangling schedule"
        )
    # No 2Q layer exists, so no reuse decisions or inter-layer placements exist.
    slm = spec["storage_zones"][0]["slms"][0]["id"]
    site = (slm, 0, 0)
    if not arch.is_valid_SLM_position(*site):
        raise ValueError("Invalid storage site")
    compiler.gate_scheduling = []
    compiler.qubit_mapping = [[site]]
    compiler.route_qubit()
    compiler.verify_scheduling([])
    compiler.verify_qubit_mapping(0)
    trace = compiler.result_json
    events = []
    if [inst["type"] for inst in trace["instructions"]] != ["init", "1qGate"]:
        raise ValueError(
            "Expected only initial placement and one local-gate instruction"
        )
    initial, local = trace["instructions"]
    if list(initial["init_locs"][0]) != [0, *site] or any(
        list(loc) != [0, *site] for loc in local["locs"]
    ):
        raise ValueError("Unexpected change in occupied storage site")
    for gate in local["gates"]:
        events.append((local["id"], [gate["q"]], gate["name"]))
    if local["dependency"]["qubit"] != [0] or local["begin_time"] != 0:
        raise ValueError("Unexpected gate timing or dependencies")
    write_json("native_trace.json", trace)
    sim = Simulator()
    sim.set_arch_spec(spec)
    sim.parse("native_trace.json")
    score = sim.simulate()
    metrics = bind_schedule(
        QuantumCircuit.from_qasm_file("call.qasm"), capture[0], events
    )
    if (
        metrics["n_2q"]
        or score["cir_duration"] != metrics["gate_count"] * arch.time_1qGate
    ):
        raise ValueError("Inconsistent one-wire replay")
    metrics.update(
        n_data_atoms=1,
        duration_us=score["cir_duration"],
        author_model_score=score["cir_fidelity"],
        rydberg_pulses=0,
        transport_batches=0,
        geometry_findings={},
        geometry_check="Single stationary atom at a verified storage site; ZAC placement verifier",
        host="zac",
        profile=profile,
        qiskit=version("qiskit"),
        source_revision_target=HOSTS["zac"][2],
        lowering="specialized 1q adapter invoking ZAC set_program, route_qubit, verifier, Simulator; original solve failed",
        scope="Unencoded one-wire unitary subtotal",
    )
    write_json("metrics.json", metrics)


if __name__ == "__main__":
    compile_single(Path(sys.argv[1]), sys.argv[2])
