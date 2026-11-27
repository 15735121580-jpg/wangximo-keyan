"""Run an archived script with the submitted 0.2.0 host-only selector."""
import argparse
import hashlib
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", action="store_true", help="Interpret the target as a Python module.")
    parser.add_argument("target")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    wheel = Path(__file__).resolve().parents[1] / "dist/openquantum_vqa_harness-0.2.0-py3-none-any.whl"
    expected = "0e3691ebf5a4e330d7082315e6eab3cb96b525a1aea9b6988e1fce8f0a604a1d"
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != expected:
        raise RuntimeError("The archived 0.2.0 wheel failed its recorded digest check")
    sys.path.insert(0, str(wheel))
    import vqa_harness
    if vqa_harness.__version__ != "0.2.0" or not vqa_harness.__file__.startswith(str(wheel)):
        raise RuntimeError("The verified archive was not loaded")
    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(wheel) if not previous else str(wheel) + os.pathsep + previous
    sys.argv = [args.target, *args.arguments]
    if args.module:
        runpy.run_module(args.target, run_name="__main__", alter_sys=True)
    else:
        script = Path(args.target).resolve()
        sys.path.insert(0, str(script.parent))
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
