"""Sample one ideal shot of a selected angle-bound native schedule with host Qiskit."""
import argparse
import json
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate, IGate, PhaseGate, U1Gate, U2Gate, U3Gate, UGate
from qiskit.quantum_info import Statevector


def sample(schedule, seed):
    circuit = QuantumCircuit(schedule["num_qubits"], global_phase=schedule["global_phase"])
    gates = {"cz": CZGate, "id": IGate, "p": PhaseGate, "u1": U1Gate, "u2": U2Gate,
             "u3": U3Gate, "u": UGate}
    for item in schedule["gates"]:
        circuit.append(gates[item["gate"]](*item["params"]), item["qubits"])
    state = Statevector.from_instruction(circuit)
    state.seed(seed)
    return int(state.sample_memory(1)[0], 2)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("result", type=Path)
    cli.add_argument("--stage", required=True)
    cli.add_argument("--binding", required=True)
    cli.add_argument("--seed", type=int, default=42)
    args = cli.parse_args()
    run = json.loads(args.result.read_text())
    item = run["selected"][args.stage][args.binding]
    path = args.result.parent / item["files"]["bound_schedule.json"]
    print(sample(json.loads(path.read_text()), args.seed))
