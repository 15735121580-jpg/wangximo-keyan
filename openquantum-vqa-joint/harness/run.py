"""Convert a task description to a checked OpenQuantum compiler run."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SKILL = HERE.parents[1] / ".agents" / "skills" / "vqa-joint-compilation"
if not DEFAULT_SKILL.is_dir():
    DEFAULT_SKILL = HERE.parent / "skills" / "vqa-joint-compilation"
INSTRUCTION = """Extract a two-round Rb–Cs feedback VQE compiler contract.
Return JSON matching the supplied schema. You are a parser. Use no tools.
Transcribe user-supplied quantities, including fractions, with their units.
Classify supplied timings as synthetic, simulator, or supplied-calibration
only as indicated by the description. A valid contract contains every limit,
both species' population in each architecture, branch times in microseconds,
noise and evidence. For incomplete or incompatible descriptions set state to
needs_input, use a null contract and list the missing or incompatible fields.
Do not infer missing numbers, construct an architecture, solve the task or
claim hardware validation. The task description starts after this sentence.
"""


def extract(intent, directory, model):
    output_path = directory / "model-output.json"
    argv = [
        "codex", "exec", "--model", model, "--disable", "memories",
        "--disable", "external_agent_memory_import", "--ephemeral",
        "--skip-git-repo-check", "--sandbox", "read-only",
        "--cd", str(directory), "--output-schema", str(HERE / "contract-schema.json"),
        "--output-last-message", str(output_path), "-",
    ]
    with (directory / "model-run.log").open("w", encoding="utf-8") as log:
        subprocess.run(argv, input=INSTRUCTION + intent, text=True, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=360)
    return json.loads(output_path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intent", type=Path)
    parser.add_argument("--structured", action="store_true", help="Run a typed finite, interval or native job JSON directly")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--skill", type=Path, default=DEFAULT_SKILL)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    intent = args.intent.read_text(encoding="utf-8")
    directory = args.out.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    if args.structured:
        def unique_fields(pairs):
            obj = dict(pairs)
            if len(obj) != len(pairs):
                raise ValueError("Duplicate JSON field in job.")
            return obj

        try:
            job = json.loads(intent, object_pairs_hook=unique_fields)
            if not isinstance(job, dict) or job.get("mode") not in ("finite", "interval", "native"):
                raise ValueError("Structured mode must be finite, interval or native.")
            (directory / "job.json").write_text(intent, encoding="utf-8")
            if job["mode"] == "native":
                from native_jobs import run_native
                return run_native(job, args.skill.resolve(strict=True), directory)
            from typed_jobs import run_typed
            return run_typed(job, args.skill.resolve(strict=True), directory)
        except (KeyError, TypeError, ValueError, ZeroDivisionError, OSError) as error:
            print(f"Job rejected: {error}", file=sys.stderr)
            return 2

    comparator = (args.skill / "scripts" / "compare_two_species.py").resolve(strict=True)
    parsed = extract(intent, directory, args.model)
    provenance = {
        "model": args.model,
        "intent_sha256": hashlib.sha256(intent.encode()).hexdigest(),
        "compiler_sha256": hashlib.sha256(comparator.read_bytes()).hexdigest(),
        "memory_disabled": True,
        "source": "model extraction under a fixed output schema",
    }
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if parsed["state"] == "needs_input":
        for item in parsed["missing_or_incompatible"]:
            print(item)
        return 2
    if parsed["state"] != "ready" or parsed["contract"] is None or parsed["missing_or_incompatible"]:
        raise ValueError("Inconsistent model extraction status.")
    contract = directory / "contract.json"
    contract.write_text(json.dumps(parsed["contract"], indent=2) + "\n")
    command = [sys.executable, "-B", str(comparator), str(contract),
               "--out", str(directory / "result.json")]
    if parsed["seed"] is not None:
        command.extend(["--seed", str(parsed["seed"])])
    completed = subprocess.run(command, check=False)
    if completed.returncode == 0:
        receipt = json.loads((directory / "result.json").read_text())
        if not receipt["certificate_rechecked"]:
            raise RuntimeError("Compiler returned an unchecked result.")
        print(directory / "result.json")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
