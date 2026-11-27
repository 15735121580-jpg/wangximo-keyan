"""Host-independent command line interface."""
import argparse
import json
import sys
from . import __version__
from .runtime import execute, openquantum_sdk
from .native_policy import compile_native_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    run = sub.add_parser("run")
    run.add_argument("job")
    run.add_argument("--out", required=True)
    run.add_argument("--backend", choices=["openquantum", "standalone"], default="openquantum")
    native = sub.add_parser("compile-native", help="Emit one policy for a bound native host interface")
    native.add_argument("job")
    native.add_argument("--out", required=True, help="New JSON output file")
    certified = sub.add_parser("compile-joint", help="Compile on a registered platform with a certified objective gap")
    certified.add_argument("job")
    certified.add_argument("--platform", required=True)
    certified.add_argument("--tolerance", help="Absolute objective tolerance, as an exact rational")
    certified.add_argument("--out", required=True, help="New JSON result file")
    guard = sub.add_parser("guard-release", help="Release a phase policy under registered non-regression tests")
    guard.add_argument("job")
    guard.add_argument("--out", required=True, help="New JSON output file")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            print(json.dumps(openquantum_sdk(), indent=2))
        else:
            with open(args.job) as stream:
                job = json.load(stream)
            if args.command in ("compile-native", "guard-release", "compile-joint"):
                if args.command == "compile-joint":
                    from .certified_entry import compile_registered
                    result = compile_registered(job, args.platform, args.tolerance)
                elif args.command == "guard-release":
                    from .phase_release import phase_release
                    result = phase_release(job)
                else:
                    result = compile_native_policy(job)
                payload = json.dumps(result, indent=2, allow_nan=False)
                with open(args.out, "x") as stream:
                    stream.write(payload + "\n")
                print(args.out)
            else:
                print(json.dumps(execute(job, args.out, args.backend), indent=2))
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
