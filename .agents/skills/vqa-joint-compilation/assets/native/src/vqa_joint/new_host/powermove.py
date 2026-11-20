"""Bound adapter for the author's PowerMove single-AOD compiler.

The native router returns resource scalars. Observe its own intermediate gate
blocks and position updates to retain gate angles, group membership and sites.
This module runs inside an interpreter with Qiskit, matplotlib and NetworkX.
"""
from collections import Counter, defaultdict, deque
from contextlib import redirect_stdout
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from threading import RLock
from unittest.mock import patch
import inspect
import json
import os
import random
import subprocess
import sys

import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.quantum_info import Operator

if __package__ == "new_host":
    from vqa_joint.joint import Arrow, HostGraph
else:
    from ..joint import Arrow, HostGraph

REVISION = "abb420504b65276653220e385d31b75056f65912"
SOURCE = Path(
    os.environ.get(
        "JOINT_POWERMOVE_SOURCE", Path(__file__).resolve().parent / ".deps" / "PowerMove"
    )
)
PROFILES = ("single-aod-no-storage", "single-aod-storage")
ROW = 2
LOCK = RLock()


@lru_cache(maxsize=1)
def upstream():
    rev = subprocess.run(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    if rev.stdout.strip() != REVISION:
        raise ValueError("PowerMove source must match the pinned revision")
    status = subprocess.run(
        ["git", "-C", str(SOURCE), "status", "--porcelain", "--", "*.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    if status.stdout.strip():
        raise ValueError("PowerMove Python source differs from the pinned revision")
    sys.path.insert(0, str(SOURCE.resolve()))
    state = random.getstate()
    try:
        import Construct_Circuit as frontend
        import mvqc as backend
    finally:
        random.setstate(state)
        sys.path.remove(str(SOURCE.resolve()))

    if any(
        Path(module.__file__).resolve().parent != SOURCE.resolve() for module in (frontend, backend)
    ):
        raise ValueError("PowerMove module import collision")
    return frontend, backend


def blocks(native):
    frontend, _ = upstream()
    layers = []

    def capture(frame, event, value):
        if event == "return" and frame.f_code is frontend.get_cz_blocks.__code__:
            layers.extend(zip(frame.f_locals["block_types"], frame.f_locals["gate_blocks"]))

    previous = sys.getprofile()
    try:
        sys.setprofile(capture)
        cz = frontend.get_cz_blocks(native)
    finally:
        sys.setprofile(previous)
    if not layers and native.size():
        raise ValueError("PowerMove omitted the single-gate / CZ interleaving")
    return layers, tuple(tuple(sorted(block)) for block in cz)


@lru_cache(maxsize=128)
def route(n, cz, storage):
    """Same CZ-block input, chip, controls and SA seed give the same native route."""
    _, backend = upstream()
    if not cz:
        raise ValueError("PowerMove SA placement requires a nonempty CZ interaction graph")
    trace = []
    initial = []
    original_update = backend.update

    def observe_update(*args):
        ctx = inspect.currentframe().f_back.f_locals
        mapping = ctx["qubit_mapping"]
        if not initial:
            top = max(y for x, y in mapping)
            initial.extend((x, -2 - (top - y) if storage else y) for x, y in mapping)
        positions, graph, empty_space = original_update(*args)
        trace.append(
            dict(
                cz=sorted(tuple(sorted(gate)) for gate in ctx["mg"]),
                move_groups=[
                    [(wire, list(target)) for wire, target in group]
                    for group in ctx["parallel_move_groups"]
                ],
                positions=[positions[q] for q in range(n)],
            )
        )
        return positions, graph, empty_space

    state = random.getstate()
    try:
        random.seed(0)
        with patch.object(backend, "update", observe_update):
            values = backend.mvqc([set(block) for block in cz], ROW, n, storage)
    finally:
        random.setstate(state)
    return initial, trace, values


def check_geometry(initial, steps):
    sites = list(map(tuple, initial))
    if len(set(sites)) != len(sites):
        raise ValueError("PowerMove initial placement overlaps")
    for index, step in enumerate(steps):
        for group in step["move_groups"]:
            if len({wire for wire, _ in group}) != len(group):
                raise ValueError("A move group contains one atom twice")
            for (a, u), (b, v) in combinations(group, 2):
                for dim in (0, 1):
                    if np.sign(sites[a][dim] - sites[b][dim]) != np.sign(u[dim] - v[dim]):
                        raise ValueError(f"PowerMove AOD order conflict in stage {index}")
            for q, dst in group:
                x, y = dst
                if not (0 <= x < ROW and (-8 * ROW - 2 <= y <= -2 or 0 <= y < ROW)):
                    raise ValueError("Move leaves the declared chip and storage sites")
                sites[q] = tuple(dst)
        if sites != list(map(tuple, step["positions"])):
            raise ValueError("Movement replay differs from the native router position update")
        occupied = {
            tuple((a, b))
            for a, b in combinations(range(len(sites)), 2)
            if sites[a] == sites[b] and sites[a][1] >= 0
        }
        if occupied != set(map(tuple, step["cz"])):
            raise ValueError("PowerMove CZ stage contains missing or unintended interactions")


def bind_layers(native, layers, routed):
    single = defaultdict(deque)
    for item in native.data:
        if len(item.qubits) == 1:
            q = native.find_bit(item.qubits[0]).index
            single[q].append(item.operation)
    steps = deque(routed)
    bound = QuantumCircuit(native.num_qubits)
    gates = []
    for kind, wires in layers:
        if kind == "SG":
            for q in sorted(wires):
                op = single[q].popleft()
                bound.append(op, [q])
                gates.append(dict(gate=op.name, qubits=[q], params=list(map(float, op.params))))
        elif kind == "CZ":
            pending = Counter(tuple(sorted(w)) for w in wires)
            while pending:
                if not steps:
                    raise ValueError("Native router dropped a CZ block")
                emitted = steps.popleft()["cz"]
                if not emitted:
                    raise ValueError("Native router emitted an empty CZ layer")
                for pair in emitted:
                    key = tuple(pair)
                    if key not in pending:
                        raise ValueError("Native router crossed an intervening single-gate block")
                    pending[key] -= 1
                    if pending[key] == 0:
                        del pending[key]
                    bound.cz(*key)
                    gates.append(dict(gate="cz", qubits=list(key), params=[]))
        else:
            raise ValueError(f"Unrecognized author block {kind}")
    if steps or any(single.values()) or len(gates) != native.size():
        raise ValueError("Native execution did not consume all bound operations")
    return bound, gates


def compile_power(host, source_qasm, directory, profile=PROFILES[0]):
    if host != "powermove" or profile not in PROFILES:
        raise ValueError("Unknown PowerMove host or profile")
    circuit = qasm2.loads(source_qasm)
    n = circuit.num_qubits
    if n > ROW**2 or circuit.num_clbits:
        raise ValueError(
            "Configured PowerMove chip supports up to four data atoms and bound unitary calls"
        )
    native = transpile(
        circuit, basis_gates=["u1", "u2", "u3", "cz", "id"], optimization_level=2, seed_transpiler=0
    )
    with LOCK, open(os.devnull, "w") as sink, redirect_stdout(sink):
        layers, cz = blocks(native)
        try:
            initial, steps, values = route(n, cz, profile == PROFILES[1])
        except (IndexError, KeyError, ValueError, ArithmeticError) as exc:
            raise ValueError(
                f"PowerMove native router rejected the binding ({type(exc).__name__}: {exc})"
            ) from exc
    check_geometry(initial, steps)
    bound, gates = bind_layers(native, layers, steps)
    u, v = Operator(circuit).data, Operator(bound).data
    overlap = np.vdot(u, v)
    if abs(overlap) == 0:
        raise ValueError("Orthogonal emitted and reference unitaries")
    error = float(np.linalg.norm(v - u * (overlap / abs(overlap)), ord=2))
    transfer, motion, proxy, f1, f2, idle, atom, coherence, groups = values
    n2 = sum(g["gate"] == "cz" for g in gates)
    metrics = dict(
        host=host,
        profile=profile,
        source_revision=REVISION,
        geometry_findings=[],
        geometry_scope="observed site bounds, per-group AOD axis order, retained CZ pairs and bystanders at discrete CZ stages",
        unitary_operator_error=error,
        n_data_atoms=n,
        n_1q=len(gates) - n2,
        n_2q=n2,
        gate_count=len(gates),
        gate_dependency_depth=bound.depth(),
        transport_us=transfer + motion,
        transfer_us=transfer,
        motion_us=motion,
        move_groups=groups,
        author_fidelity_proxy=proxy,
        circuit_process_fidelity=None,
        execution_seconds=None,
    )
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "call.qasm").write_text(source_qasm)
    for name, payload in [
        ("native_trace", dict(initial=initial, steps=steps, author_model_return=values)),
        ("bound_schedule", dict(num_qubits=n, gates=gates)),
        ("metrics", metrics),
    ]:
        (dest / f"{name}.json").write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    return metrics


def power_host():
    return HostGraph(
        "powermove",
        (
            Arrow("generate", "LogicalCall", "EquivalentCircuits"),
            Arrow("commit", "EquivalentCircuits", "Circuit"),
            Arrow("native", "Circuit", "Schedule"),
            Arrow("apply", "Schedule", "Instrument"),
        ),
        PROFILES,
        compile_power,
    )
