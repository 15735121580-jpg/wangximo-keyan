"""Compile bound host regions under explicit hardware and readout capabilities."""
import math

from .compiler.native_ports import rewrite_native_ports
from .compiler.coherent_ports import rewrite_coherent_ports


def finite_parameters(params):
    return all(type(x) in (int, float) and math.isfinite(x) for x in params)


def compile_native_policy(job):
    """Emit one policy covering all settings. No simulator, samples or objective scores are inputs."""
    if not isinstance(job, dict):
        raise ValueError("Expected a native policy object")
    kind = job.get("interface")
    capabilities = job.get("capabilities")
    settings = job.get("settings")
    if not isinstance(capabilities, dict) or not all(type(v) is bool for v in capabilities.values()):
        raise ValueError("Supply explicit Boolean host capabilities")
    if not isinstance(settings, list) or not settings:
        raise ValueError("A nonempty binding grid is required")

    if kind == "cz-rotation-readout":
        required = {"fresh_zero", "terminal_z", "retune_local_and_decode_xor", "skip_whole_cz_run"}
        if set(capabilities) != required or not (capabilities["fresh_zero"] and capabilities["terminal_z"]):
            raise ValueError("The CZ port requires a fresh zero register and terminal Z measurement")
    elif kind == "cz-rotation-coherent":
        required = {"individual_u3", "addressed_row_hrzhrx"}
        if set(capabilities) != required or sum(capabilities.values()) != 1:
            raise ValueError("Declare one local control mode for the coherent port")
    elif kind == "two-wire-rzz-relay":
        if capabilities != {"arbitrary_local_u": True, "remote_cz": True, "terminal_z": True}:
            raise ValueError("The relay port requires arbitrary local U, remote CZ and terminal Z measurement")
        try:
            from .compiler.native_relay import compile_native_relay
        except ImportError as exc:
            raise RuntimeError("Install the simulator extra for the relay adapter") from exc
    else:
        raise ValueError("Unknown native port interface")

    policies = []
    keys = set()
    for setting in settings:
        if not isinstance(setting, dict) or not {"binding", "basis", "source"} <= setting.keys():
            raise ValueError("Each setting requires binding, basis and source")
        key = (setting["binding"], setting["basis"])
        if type(key[0]) is not int or not isinstance(key[1], str) or key in keys:
            raise ValueError("Use distinct integer binding and string basis pairs")
        keys.add(key)
        source = setting["source"]
        if kind in ("cz-rotation-readout", "cz-rotation-coherent"):
            if not isinstance(source, dict) or type(source.get("num_qubits")) is not int or source["num_qubits"] < 1:
                raise ValueError("Supply a positive declared native address width")
            if not isinstance(source.get("gates"), list):
                raise ValueError("Supply a bound native gate list")
            for op in source["gates"]:
                if (not isinstance(op, dict) or not isinstance(op.get("qubits"), list)
                    or not isinstance(op.get("gate"), str) or not isinstance(op.get("params", []), list)
                    or any(type(q) is not int for q in op["qubits"]) or not finite_parameters(op.get("params", []))):
                    raise ValueError("Invalid native operation")
            if kind == "cz-rotation-coherent":
                control = "individual-u3" if capabilities["individual_u3"] else "addressed-row-hrzhrx"
                plan = rewrite_coherent_ports(source, local_control=control)
            else:
                plan = rewrite_native_ports(source, allow_local_frames=capabilities["retune_local_and_decode_xor"],
                                            allow_skipping_cz_runs=capabilities["skip_whole_cz_run"])
        else:
            if not isinstance(source, list):
                raise ValueError("Supply the logical front-end source region for the relay adapter")
            for op in source:
                if (not isinstance(op, list) or len(op) != 3 or not isinstance(op[0], str)
                    or not isinstance(op[1], list) or any(type(q) is not int for q in op[1])
                    or (op[2] is not None and not finite_parameters([op[2]]))):
                    raise ValueError("Invalid bound logical operation")
            plan = {"native": compile_native_relay(source)}
        policies.append({"binding": key[0], "basis": key[1], "policy": plan})
    return {"interface": kind, "capabilities": capabilities, "settings": policies}
