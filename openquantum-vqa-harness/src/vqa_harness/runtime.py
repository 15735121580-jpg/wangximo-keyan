"""Locate OpenQuantum's public SDK launcher without reading credentials."""
import json
import os
from pathlib import Path
import subprocess
import sys


def openquantum_sdk():
    explicit = os.environ.get("OPENQUANTUM_SDK_ACCESS")
    candidates = [Path(explicit)] if explicit else []
    for host in (".codex", ".claude"):
        candidates.extend(sorted(Path.home().glob(
            host + "/plugins/cache/*/openquantum/*/skills/quantum-sdk-workbench/scripts/sdk-access.mjs"
        ), reverse=True))
    for path in candidates:
        if not path.is_file():
            continue
        proc = subprocess.run(["node", str(path), "inspect", "hamiltonian-simulation"],
                              capture_output=True, text=True, timeout=30, check=False)
        if proc.returncode:
            continue
        info = json.loads(proc.stdout)
        if info.get("environmentState") != "prepared":
            continue
        python = info["run"]["command"]
        if Path(python).is_file():
            return {"backend": "openquantum", "python": python,
                    "sdk": info["id"], "locator": str(path),
                    "manifest": info.get("manifest"), "lock": info.get("lock"),
                    "bootstrap": info.get("bootstrap"),
                    "dependencies": info.get("dependencies", [])}
    raise RuntimeError("Prepared OpenQuantum Qiskit SDK not found. Install OpenQuantum and prepare "
                       "hamiltonian-simulation, or use the standalone simulator extra.")


def execute(job, output_directory, backend="openquantum"):
    """Use the selected SDK in an isolated subprocess and return its real receipt."""
    if backend not in ("openquantum", "standalone"):
        raise ValueError("backend must be openquantum or standalone")
    out = Path(output_directory).expanduser().resolve()
    if out.exists():
        raise FileExistsError("Choose a new output directory: " + str(out))
    runtime = openquantum_sdk() if backend == "openquantum" else {
        "backend": "standalone", "python": sys.executable}
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([runtime["python"], "-B", "-m", "vqa_harness.worker", str(out)],
                          input=json.dumps({"job": job, "runtime": runtime}, allow_nan=False),
                          capture_output=True, text=True, env=env, timeout=180, check=False)
    if proc.returncode:
        raise RuntimeError("VQA execution failed. " + proc.stderr[-6000:])
    result = out / "result.json"
    if not result.is_file():
        raise RuntimeError("Worker completed without its result.json")
    return json.loads(result.read_text())
