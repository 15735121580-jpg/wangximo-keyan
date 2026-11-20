"""Adapter to pinned upstream neutral-atom compilers for bound unitary calls.

Each invocation has a private working directory; the compiler checkout is read only.
The output is an angle-preserving annotated native schedule plus scoped metrics.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).absolute().parent
WORK = HERE.parent.parent
HOSTS = {
    "zap": ("zap-recompile/author", "zap-verified/venv", "ba75e945da6c14ae41ee0495dbc8e26fa4e11f8e"),
    "zac": ("campaign14/ebadi/ZAC", "campaign14/ebadi/zac_venv", "e5083362f99e6915f20c2bd0eaa88b6d1cdcecac"),
}


ZAP_PROFILES = {
    f"{schedule}-{route}": ("asap_" + schedule, route)
    for schedule in ("separate", "joint")
    for route in ("baseline", "always_move", "always_stay")
}
ZAC_PROFILES = {
    "trivial-fixed": 0,
    "trivial-dynamic": 1,
    "trivial-dynamic-reuse": 2,
    "sa-dynamic-reuse": 3,
}
DEFAULT_PROFILE = {"zap": "separate-baseline", "zac": "sa-dynamic-reuse"}


def available_profiles(host):
    """Finite, source-verified control envelope on the same hardware for each host."""
    return tuple({"zac": ZAC_PROFILES, "zap": ZAP_PROFILES}[host])


def compile_qasm(host, qasm, output_dir, profile=None):
    """Compile one bound unitary OpenQASM2 circuit, return per-call metrics.

    Angles, emitted gate semantics and measured logical resources differ from
    architecture resource estimates. See the README for exact interpretation.
    """
    if host not in HOSTS:
        raise ValueError(f"Unknown host {host!r}")
    profile = DEFAULT_PROFILE[host] if profile is None else profile
    if profile not in available_profiles(host):
        raise ValueError(f"Unknown {host} profile {profile!r}; valid: {available_profiles(host)}")
    source_dir, venv, _ = HOSTS[host]
    source = Path(os.environ.get(f"JOINT_{host.upper()}_SOURCE", WORK / source_dir)).absolute()
    python = Path(os.environ.get(f"JOINT_{host.upper()}_PYTHON", WORK / venv / "bin/python")).absolute()
    if not python.is_file() or not (source / host / f"{host}.py").is_file():
        raise FileNotFoundError(f"Install {host} and set JOINT_{host.upper()}_SOURCE and JOINT_{host.upper()}_PYTHON")
    out = Path(output_dir).absolute()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{host}-") as td:
        scratch = Path(td)
        (scratch / "call.qasm").write_text(qasm)
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="1", RAYON_NUM_THREADS="1", QISKIT_PARALLEL="FALSE")
        process = subprocess.run([str(python), str(HERE/"worker.py"), host, str(source), profile],
                                 cwd=scratch, env=env, capture_output=True, text=True)
        if process.returncode or "[Error]" in process.stdout:
            raise RuntimeError(f"{host} compilation failed: {(process.stdout + process.stderr)[-1800:]}")
        artifacts = {}
        for filename in ("call.qasm", "native_trace.json", "bound_schedule.json", "metrics.json"):
            artifacts[filename] = (scratch / filename).read_bytes()
        for filename, data in artifacts.items():
            with tempfile.NamedTemporaryFile(dir=out, delete=False) as f:
                f.write(data)
                staged = f.name
            os.replace(staged, out / filename)
    return json.loads(artifacts["metrics.json"])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("host", choices=HOSTS)
    cli.add_argument("qasm", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--profile", help="native, finite host profile; omit for the default")
    args = cli.parse_args()
    print(json.dumps(compile_qasm(args.host, args.qasm.read_text(), args.output, args.profile), indent=2))


if __name__ == "__main__":
    main()
