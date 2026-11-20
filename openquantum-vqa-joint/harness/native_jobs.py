"""Run structured native packets through the bundled compiler and its replay CLI."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def check_packet(packet):
    if not isinstance(packet, dict):
        raise ValueError("packet must be a JSON object")
    # Stage and binding names can become part of a run's native file tree.
    for stage in packet.get("stages", []):
        for row in [stage, *stage["bindings"]]:
            name = row["id"]
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", name):
                raise ValueError("Use a plain stage or binding identifier")


def run_native(job, skill: Path, directory: Path) -> int:
    """Write a relative-path result manifest plus original native schedules."""
    host = job.get("host")
    if host not in ("zap", "zac", "zac-encoded"):
        raise ValueError("host must be zap, zac or zac-encoded")
    encoded = host == "zac-encoded"
    allowed = {"mode", "host", "packet", "seed"}
    if not encoded:
        allowed.update({"goals", "profiles", "budget", "original_only", "wire_relabel", "replay_packet"})
    unknown = set(job) - allowed
    if unknown:
        raise ValueError(f"Unknown native fields: {sorted(unknown)}")
    packet = job["packet"]
    check_packet(packet)
    seed = job.get("seed")
    if "seed" in job and (type(seed) is not int or seed < 0 or (not encoded and not packet.get("feedback"))):
        raise ValueError("Use a nonnegative seed on an encoded or feedback job")
    if "replay_packet" in job:
        check_packet(job["replay_packet"])
    source = (skill / "assets" / "native" / "src").resolve()
    if not (source / "vqa_joint" / "product_cli.py").is_file():
        raise ValueError("The selected Skill does not contain the native compiler")
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1")
    entry = [sys.executable, "-B", "-m", "vqa_joint.product_cli"]
    def save(name, value):
        (directory / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    def invoke(argv, receipt=None):
        result = subprocess.run(entry + argv, cwd=directory, env=env, text=True,
                                stdout=subprocess.PIPE if receipt else None, check=False)
        if receipt and result.returncode == 0:
            save(receipt, json.loads(result.stdout))
        return result.returncode

    save("packet.json", packet)
    argv = ["compile-encoded" if encoded else "compile", "packet.json", "--out", "native"]
    if not encoded:
        argv.extend(["--host", host])
        for name, flag in (("profiles", "--profile"), ("goals", "--goal")):
            if name not in job:
                continue
            values = job[name]
            if not isinstance(values, list) or not values or not all(
                    isinstance(v, str) and re.fullmatch(r"[a-z][a-z_-]*", v) for v in values):
                raise ValueError(f"{name} must be a nonempty list of policy keywords")
            if name == "profiles":
                for value in values:
                    argv.extend([flag, value])
            else:
                argv.extend([flag, ",".join(values)])
        if "budget" in job:
            if type(job["budget"]) is not int or job["budget"] < 1:
                raise ValueError("budget must be a positive integer")
            argv.extend(["--budget", str(job["budget"])])
        for name in ("original_only", "wire_relabel"):
            value = job.get(name, False)
            if type(value) is not bool:
                raise ValueError(f"{name} must be boolean")
            if value:
                argv.append("--" + name.replace("_", "-"))
    status = invoke(argv)
    manifest = {"mode": "native", "host": host}
    native_result = directory / "native/result.json"
    if native_result.is_file():
        manifest["native_result"] = "native/result.json"
        manifest["selection_status"] = json.loads(native_result.read_text())["status"]
    if status == 0 and "replay_packet" in job:
        save("replay-packet.json", job["replay_packet"])
        status = invoke(["replay", "replay-packet.json", "--strategy", "native/strategy.json",
                         "--out", "rebound"])
        if status == 0:
            manifest["rebound_result"] = "rebound/result.json"
    if status == 0 and seed is not None:
        status = invoke(["replay-encoded" if encoded else "sample-job", "native/result.json",
                         "--seed", str(seed)], "sample.json")
        if status == 0:
            manifest["seed"] = seed
            manifest["sample"] = "sample.json"
    manifest["exit_code"] = status
    save("result.json", manifest)
    return status
