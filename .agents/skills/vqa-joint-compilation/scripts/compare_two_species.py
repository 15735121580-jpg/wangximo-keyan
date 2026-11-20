#!/usr/bin/env python3
"""Compare declared complete policies for the bundled two-round feedback VQE."""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
from random import Random
import sys

import z3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "certified"))
from feedback_vqe import Program, leaf_distribution
from optimal_compiler import Diagram, solve, verify


def rational(value):
    """Read an exact fraction, keeping the input independent of float rounding."""
    if type(value) not in (int, str):
        raise ValueError("Use JSON integers or decimal/fraction strings for exact scalars.")
    return Fraction(value)


def real(value):
    return z3.RealVal(str(value))


def probability(value):
    value = rational(value)
    if not 0 <= value <= 1:
        raise ValueError("Probabilities must lie in [0, 1].")
    return value


def count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Atom counts and depths must be nonnegative integers.")
    return value


def duration(value):
    value = rational(value)
    if value < 0:
        raise ValueError("Durations must be nonnegative.")
    return value


def compare(packet, seed):
    if packet["task"] != "two-round-feedback-vqe":
        raise ValueError("This input adapter accepts only two-round-feedback-vqe.")
    hardware = packet["hardware"]
    species = (hardware["data_species"], hardware["service_species"])
    if set(species) != {"Rb", "Cs"} or hardware["time_unit"] != "us":
        raise ValueError("Declare distinct Rb/Cs data and service species and use microseconds.")
    if hardware["evidence"] not in {"synthetic", "simulator", "supplied-calibration"}:
        raise ValueError("Declare the source class of the numerical input.")
    if not isinstance(hardware["source"], str) or not hardware["source"].strip():
        raise ValueError("Provide a citation or a description of the numerical input.")
    bounds = packet["limits"]
    required = {"max_peak_atoms", "max_depth", "max_model_flip_probability",
                "min_record_fidelity"}
    if bounds.keys() != required:
        raise ValueError(f"Supply exactly these model bounds: {', '.join(sorted(required))}.")
    atom_limit, depth_limit = count(bounds["max_peak_atoms"]), count(bounds["max_depth"])
    max_flip = probability(bounds["max_model_flip_probability"])
    min_fidelity = probability(bounds["min_record_fidelity"])
    ideal = leaf_distribution(0)
    policies = {}
    diagrams = []
    for candidate in packet["architectures"]:
        name = candidate["id"]
        if not isinstance(name, str) or not name.strip() or name in policies:
            raise ValueError("Each candidate needs a unique, non-empty string ID.")
        atoms = candidate["peak_atoms_by_species"]
        if set(atoms) != set(species) or not all(count(n) > 0 for n in atoms.values()):
            raise ValueError("Account for the peak inventory of both species.")
        peak, depth = sum(atoms.values()), count(candidate["max_branch_depth"])
        prefix = duration(candidate["prefix_us"])
        tail_plus = duration(candidate["suffix_us"]["+"])
        tail_minus = duration(candidate["suffix_us"]["-"])
        flip = probability(candidate["flip_per_readout"])
        if flip >= Fraction(1, 2):
            raise ValueError("A symmetric readout-flip rate must be less than one half.")
        noisy = leaf_distribution(flip)
        positive = sum(p for (first, _), p in noisy.items() if first == 1)
        expected = prefix + positive * tail_plus + (1 - positive) * tail_minus
        job_flip = 1 - (1 - flip) ** 2
        affinity = sum(z3.Sqrt(real(ideal[h] * noisy[h])) for h in ideal)
        fidelity_check = z3.simplify(affinity ** 2 >= real(min_fidelity))
        if not (z3.is_true(fidelity_check) or z3.is_false(fidelity_check)):
            raise ValueError("Could not simplify the exact record-fidelity constraint.")
        checks = {
            "peak_atoms": peak <= atom_limit,
            "branch_depth": depth <= depth_limit,
            "model_flip_probability": job_flip <= max_flip,
            "record_fidelity": z3.is_true(fidelity_check),
        }
        fidelity = sum(math.sqrt(float(ideal[h] * noisy[h])) for h in ideal) ** 2
        policies[name] = {
            "prefix_us_exact": str(prefix),
            "suffix_us_exact": {"+": str(tail_plus), "-": str(tail_minus)},
            "flip_per_readout_exact": str(flip),
            "expected_us_exact": str(expected),
            "peak_atoms_by_species": atoms,
            "conservative_total_atoms": peak,
            "max_branch_depth": depth,
            "p_any_model_readout_flip_exact": str(job_flip),
            "record_fidelity_approx": fidelity,
            "constraints_met": checks,
        }
        diagrams.append(Diagram(name, (), z3.BoolVal(all(checks.values())), real(expected), name))
    if not diagrams:
        raise ValueError("At least one feasible or infeasible declared architecture is required.")
    result = solve(diagrams)
    if not verify(diagrams, result):
        raise RuntimeError("The independent exact feasibility and lower-bound checks failed.")
    report = {
        "model": packet["task"],
        "input_hardware": {name: hardware[name] for name in
                           ("data_species", "service_species", "time_unit", "evidence", "source")},
        "declared_limits": bounds,
        "prediction_assumption": "independent symmetric flips on bound Z records",
        "scope": "declared finite model catalogue",
        "status": {"optimal": "optimal_in_declared_catalogue",
                   "infeasible": "infeasible_in_declared_catalogue"}[result.status],
        "certificate_rechecked": True,
        "architectures": policies,
        "quantum_process_fidelity": None,
        "device_validation": None,
    }
    if result.diagram is not None:
        winner = result.diagram.name
        report.update({
            "selected": winner,
            "ties": [name for name, entry in policies.items()
                     if all(entry["constraints_met"].values()) and
                     entry["expected_us_exact"] == policies[winner]["expected_us_exact"]],
        })
        if seed is not None:
            outcomes = leaf_distribution(rational(policies[winner]["flip_per_readout_exact"]))
            # Sample the exact categorical distribution with an integer draw.
            scale = math.lcm(*(p.denominator for p in outcomes.values()))
            draw = Random(seed).randrange(scale)
            for outcomes_pair, weight in outcomes.items():
                draw -= int(weight * scale)
                if draw < 0:
                    break
            replay = Program("e").replay(*outcomes_pair)
            report["seeded_model_replay"] = {
                "seed": seed, "readouts": replay["readouts"], "parameters": replay["parameters"],
                "choice": replay["choice"],
                "empirical_energies_exact": {k: str(v) for k, v in replay["empirical_energies"].items()},
            }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, help="Write a new JSON report. Default is stdout.")
    parser.add_argument("--seed", type=int, help="Replay one modeled readout pair deterministically.")
    args = parser.parse_args()
    try:
        payload = compare(json.loads(args.input.read_text()), args.seed)
        rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        if args.out is None:
            print(rendered, end="")
        else:
            with args.out.open("x", encoding="utf-8") as output:
                output.write(rendered)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OSError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0 if "selected" in payload else 2


if __name__ == "__main__":
    raise SystemExit(main())
