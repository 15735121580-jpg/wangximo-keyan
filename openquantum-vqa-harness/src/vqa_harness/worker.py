"""Scientific subprocess launched inside OpenQuantum's pinned SDK environment."""
import importlib.util
import json
from pathlib import Path
import sys


def main():
    packet = json.load(sys.stdin)
    bootstrap = packet["runtime"].get("bootstrap")
    if bootstrap:
        spec = importlib.util.spec_from_file_location("oq_bootstrap", bootstrap["path"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.prepare_sdk("hamiltonian-simulation")
    from .workflow import run_job
    out = Path(sys.argv[1])
    result = run_job(packet["job"], out)
    (out / "runtime.json").write_text(json.dumps(packet["runtime"], indent=2) + "\n")
    print(out / "result.json")


if __name__ == "__main__":
    main()
