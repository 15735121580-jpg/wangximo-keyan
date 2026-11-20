"""Test the installed CLI and independently sample its emitted native schedules.

Run in the CLI virtual environment with the four JOINT host variables set.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
CLI = Path(sys.executable).with_name("vqa-joint")


def run(args, cwd, expected=0):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    p = subprocess.run([str(x) for x in args], cwd=cwd, env=env, text=True, capture_output=True)
    if p.returncode != expected:
        raise RuntimeError(f"Unexpected exit {p.returncode} for {args[0]}: {p.stdout} {p.stderr}")
    return p.stdout


def read(path):
    return json.loads(path.read_text())


def host_check(host, root):
    run_dir = root / host
    example = HERE/"example.json"
    run([CLI, "compile", example, "--host", host, "--goal", "peak_data_atoms,duration_us", "--out", run_dir], root)
    result = read(run_dir/"result.json")
    assert result["status"] == "selected"
    binding = result["selected"]["readout"]["theta0"]
    assert binding["readout"]["mask"] == 5 and binding["readout"]["native_width"] == 1
    assert all((run_dir/file).is_file() for file in binding["files"].values())
    path = run_dir/binding["files"]["bound_schedule.json"]
    m = binding["metrics"]
    assert abs(m["unitary_operator_error"]) <= 1e-8 and not m["geometry_findings"]
    keys = ["--stage", "readout", "--binding", "theta0"]
    p = os.environ["JOINT_" + host.upper() + "_PYTHON"]
    native = run([p, HERE/"sample_native.py", run_dir/"result.json", *keys, "--seed", "42"], root).strip()
    decoded = json.loads(run([CLI, "decode", run_dir/"result.json", *keys, "--native", native], root))
    assert decoded["native_outcome"] == 1 and decoded["logical_bitstring"] == "101"
    packet = read(example)
    operations = packet["stages"][0]["bindings"][0]["operations"]
    operations[0][2] = 2.1
    target = root/(host+"-rebound.json")
    target.write_text(json.dumps(packet))
    rebound = root/(host+"-rebound")
    run([CLI, "replay", target, "--strategy", run_dir/"strategy.json", "--out", rebound], root)
    assert read(rebound/"result.json")["status"] == "selected"

    digest = hashlib.sha256((run_dir/"result.json").read_bytes()).hexdigest()
    run([CLI, "compile", example, "--host", host, "--out", run_dir], root, 2)
    assert digest == hashlib.sha256((run_dir/"result.json").read_bytes()).hexdigest()
    operations[1][1] = [2, 1]
    target.write_text(json.dumps(packet))
    denied = root/(host+"-refused")
    run([CLI, "replay", target, "--strategy", run_dir/"strategy.json", "--out", denied], root, 2)
    assert read(denied/"result.json")["status"] == "failed" and not (denied/"strategy.json").exists()

    # User keys are labels only, including keys that look like relative paths.
    packet = read(example)
    packet["stages"][0]["id"] = "../outside-stage"
    packet["stages"][0]["bindings"][0]["id"] = "../outside-binding"
    target.write_text(json.dumps(packet))
    escaped = root/(host+"-labels")
    run([CLI, "compile", target, "--host", host, "--out", escaped], root)
    labeled = read(escaped/"result.json")["selected"]["../outside-stage"]["../outside-binding"]
    for value in labeled["files"].values():
        emitted = (escaped/value).resolve()
        assert emitted.is_relative_to(escaped.resolve()/"native/stage-0/binding-0") and emitted.is_file()
    assert not (root/"outside-stage").exists() and not (root/"outside-binding").exists()

    # Valid measured packet with a completely empty unitary window.
    packet["stages"][0]["bindings"][0]["operations"] = operations[-3:]
    target.write_text(json.dumps(packet))
    empty = root/(host+"-empty")
    run([CLI, "compile", target, "--host", host, "--out", empty], root, 2)
    no_output = read(empty/"result.json")
    # With joint compilation enabled, the empty unitary is refused while
    # establishing the matched original anchor, before candidate extraction.
    assert no_output["status"] == "reference_unavailable" and not no_output["selected"]
    assert "no common admissible action" in no_output["search_detail"]
    assert not (empty/"strategy.json").exists()
    return dict(profile=binding["profile"], implementation=binding["implementation"], search=result["search"],
                training_angle_rad=2.4, held_out_angle_rad=2.1, held_out_status="selected",
                decoder=binding["readout"], sampled=decoded, draw_source="Qiskit native statevector, seed 42",
                native_schedule=read(path), native_gate_count=m["gate_count"],
                operator_error=m["unitary_operator_error"], qiskit=m["qiskit"],
                source_revision_target=m["source_revision_target"], refused_reused_output=True,
                refused_out_of_shape_replay=True, filesystem_safe_stage_and_binding_labels=True, empty_window_status=no_output["status"])


if __name__ == "__main__":
    from importlib.metadata import version
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    opts = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        checks = {h: host_check(h, Path(tmp)) for h in ("zap", "zac")}
    report = dict(cli_version=version("vqa-joint"), python=sys.version.split()[0], numpy=version("numpy"),
                  hosts=checks, execution="installed CLI and native software backends, ideal sampled output")
    data = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if opts.output:
        opts.output.write_text(data)
    else:
        print(data, end="")
