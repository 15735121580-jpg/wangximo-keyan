"""Compile measured neutral-atom circuit packets and decode native outcomes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from math import expm1, isfinite, log
import os
from pathlib import Path
import secrets
import shutil
import sys

from vqa_joint.contract import DIGITAL_GATE_INPUTS
from vqa_joint.feedback_job import FEEDBACK_COSTS, FeedbackJob, sample_saved
from vqa_joint.host_policy import Binding, HostPolicy, check_nonregression, extract_policy
from vqa_joint.host_scope import build_scope, matched_contract, search_summary
from vqa_joint.hosts.native import DEFAULT_PROFILE, HOSTS, available_profiles
from vqa_joint.joint import author_host, synthesize
from vqa_joint.ports import WirePort
from vqa_joint.sector_dispatch import Readout


COSTS = ("duration_us", "gate_count", "peak_data_atoms", "gate_dependency_depth", "gate_loss_proxy_sum")
DEFAULT_GOAL = COSTS[:3]
FIDELITIES = {key: bound.low for key, bound in DIGITAL_GATE_INPUTS}
ARITIES = {"h": 1, "x": 1, "y": 1, "z": 1, "s": 1, "sdg": 1, "t": 1, "tdg": 1,
           "rx": 1, "ry": 1, "rz": 1, "cx": 2, "cz": 2, "rzz": 2, "measure": 1}
ROTATIONS = {"rx", "ry", "rz", "rzz"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def named_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Provide at least one entry per stage and binding list")
    seen = set()
    for row in rows:
        key = row["id"]
        if not isinstance(key, str) or not key.strip() or key in seen:
            raise ValueError(f"Expected a unique nonempty stage or binding ID, got {key!r}")
        seen.add(key)
        yield key, row


def parse_packet(packet):
    if packet["schema_version"] != 1 or not isinstance(packet["task_contract"], str) or not packet["task_contract"].strip():
        raise ValueError("Expected schema_version 1 and a nonempty task_contract")
    stages, shots = {}, {}
    for stage, row in named_rows(packet["stages"]):
        bindings = stages[stage] = {}
        for key, body in named_rows(row["bindings"]):
            width, count = body["width"], body.get("shots")
            if type(width) is not int or not 1 <= width <= 8:
                raise ValueError("Each binding needs 1–8 logical wires")
            if count is not None and (type(count) is not int or count < 1):
                raise ValueError("Binding shots must be a positive integer")
            if count is None and packet.get("feedback") is None:
                raise ValueError("A static binding requires a shot count")
            ops = []
            for name, wires, angle in body["operations"]:
                if name not in ARITIES or not isinstance(wires, list) or len(wires) != ARITIES[name]:
                    raise ValueError(f"Unsupported gate or arity: {name!r}")
                if len(set(wires)) != len(wires) or any(type(q) is not int or not 0 <= q < width for q in wires):
                    raise ValueError(f"Invalid wires for {name}")
                if name in ROTATIONS:
                    if type(angle) not in (float, int) or not isfinite(angle):
                        raise ValueError(f"Expected finite numeric angle for {name}")
                elif angle is not None:
                    raise ValueError(f"Expected null angle for {name}")
                ops.append((name, tuple(wires), angle))
            suffix = [("measure", (q,), None) for q in range(width)]
            if ops[-width:] != suffix or any(g == "measure" for g, _, _ in ops[:-width]):
                raise ValueError("End each call with one Z measurement per wire in increasing order")
            bindings[key] = (width, tuple(ops))
            shots[stage, key] = count
    return stages, shots


def costs(selected, shots):
    total = dict.fromkeys(COSTS, 0)
    for stage, row in selected.items():
        for key, call in row.items():
            m = call.lowered.native_metrics
            count = shots[stage, key]
            for metric in ("duration_us", "gate_count", "gate_dependency_depth"):
                total[metric] += count * m[metric]
            total["peak_data_atoms"] = max(total["peak_data_atoms"], m["n_data_atoms"])
            log_weight = m["n_1q"] * log(FIDELITIES["1q_average"]) + m["n_2q"] * log(FIDELITIES["2q_average"])
            total["gate_loss_proxy_sum"] += count * -expm1(log_weight)
    return total


def original_binding(binding):
    port = WirePort(tuple(range(binding.n)))
    calls = tuple(call for call in binding.calls if call.family == "identity" and
                  call.lowered.implementation == "original" and call.lowered.wire_port == port)
    return Binding(binding.n, binding.ops, calls)


def host_environment(host):
    for suffix in ("SOURCE", "PYTHON"):
        key = f"JOINT_{host.upper()}_{suffix}"
        if key not in os.environ or not os.environ[key]:
            raise ValueError(f"Set {key} to the installed host {'checkout' if suffix == 'SOURCE' else 'interpreter'}")
        path = Path(os.environ[key]).expanduser().absolute()
        check = path / host / (host + ".py") if suffix == "SOURCE" else path
        if not check.is_file():
            raise ValueError(f"{key} does not identify an installed {host} {suffix.lower()}")
        os.environ[key] = str(path)


def artifact(call, out, count):
    native = call.lowered
    directory = native.schedule.parent
    files = ("bound_schedule.json", "native_trace.json", "metrics.json", "call.qasm")
    return dict(family=call.family, variant=call.variant, profile=native.profile,
                implementation=native.implementation, shots=count,
                readout=asdict(call.readout), wire_order=list(native.wire_port.physical_of_logical),
                metrics=native.native_metrics,
                files={name: (directory/name).relative_to(out).as_posix() for name in files})


def comparison_scope(packet, host, priorities, envelope):
    spec = packet.get("feedback")
    platform = dict(host_revision=HOSTS[host][2], gate_average_fidelities=FIDELITIES,
                    host_profiles=list(envelope), geometry="pinned host profile default geometry")
    defaults = dict(preparation="Fresh all-zero data for every call in the simulation",
                    readout="Terminal computational-basis readout and logical wire decoding")
    if spec is not None:
        defaults.update(estimation="Packet outcome scores and batch reductions",
                        feedback="Finite measurement-conditioned controller from packet",
                        delivery="Packet terminal outcome and success event")
    return build_scope(host, task=packet, platform=platform, priorities=priorities, conventional=defaults)


def execute(args):
    packet = load(args.packet)
    stages, shots = parse_packet(packet)
    replay = args.cmd == "replay"
    spec = packet.get("feedback")
    job = FeedbackJob(spec, stages) if spec is not None else None
    if job:
        for node in job.graph.nodes.values():
            if node["kind"] == "batch" and shots[node["stage"], node["binding"]] not in (None, node["shots"]):
                raise ValueError("Omit a binding shot count when its feedback batches use different shot counts")
    saved = load(args.strategy) if replay else {}
    if replay and saved.get("feedback_contract") != spec:
        raise ValueError("Replay requires the original feedback and noise contract")
    stored = HostPolicy.from_dict(saved) if replay else None
    host = stored.host if replay else args.host
    if host not in HOSTS:
        raise ValueError("Unknown native host")
    if replay and (set(stored.actions) != set(stages) or stored.task_contract != packet["task_contract"]):
        raise ValueError("Replay requires the extracted stages and task contract")
    goal = () if replay else tuple(args.goal.split(","))
    allowed = FEEDBACK_COSTS if job else COSTS
    if not replay and (not goal or len(set(goal)) != len(goal) or not set(goal) <= set(allowed)):
        raise ValueError(f"Specify distinct costs drawn from {','.join(allowed)}")
    if "modeled_job_time_us" in goal and job.clock is None:
        raise ValueError("Optimizing full modeled time requires an explicit serial clock")
    if "max_circuit_process_infidelity" in goal and not job.spaces:
        raise ValueError("Optimizing process fidelity requires a common coherent port")
    def evaluate(table):
        return job.assess(table)[0] if job else costs(table, shots)
    if not replay and args.budget < 1:
        raise ValueError("Policy evaluation budget must be positive")
    profiles = () if replay else (available_profiles(host) if args.all_profiles else
                                  tuple(args.profile or (DEFAULT_PROFILE[host],)))
    if not replay and (not set(profiles) <= set(available_profiles(host)) or len(set(profiles)) != len(profiles)):
        raise ValueError(f"Profiles for {host}: {', '.join(available_profiles(host))}")
    host_environment(host)
    algorithm = synthesize(author_host(host), enabled=stored.changed if replay else not args.original_only)
    out = args.out.absolute()
    out.mkdir(parents=True, exist_ok=False)
    task = packet["task_contract"]
    pool, selected, rejected = {}, {}, []
    report = dict(schema_version=1, host=host, task_contract=task, source_revision_target=HOSTS[host][2],
                  gate_average_fidelity_inputs=FIDELITIES, cost_scope="serial native unitary windows times declared shots",
                  circuit_process_fidelity=None, decoded_logical_failure=None, whole_job_execution_time=None,
                  scope="unencoded schedule model; readout, reset, application feedback and retry times unspecified",
                  command=args.cmd, status="incomplete", selected={}, strategy_file=None,
                  native_rejections=rejected)
    envelope = saved.get("host_profiles", sorted({a.profile for a in stored.actions.values()})) if replay else profiles
    saved_anchor = saved.get("reference_actions")
    reference_policy = HostPolicy.from_dict(dict(saved, actions=saved_anchor)) if saved_anchor else None
    if reference_policy and (set(reference_policy.actions) != set(stages) or
            any(action.family != "identity" or action.implementation != "original" or
                action.wire_order != tuple(range(len(action.wire_order))) or action.profile not in envelope
                for action in reference_policy.actions.values())):
        raise ValueError("Replay requires a matched original reference action for each stage")
    original_algorithm = synthesize(author_host(host), enabled=False) if replay and algorithm.changed and not reference_policy else None
    references = {}
    report["host_workflow"] = comparison_scope(packet, host, stored.objective_order if replay else goal, envelope)
    report["rewriting_enabled"] = algorithm.changed
    if job:
        report.update(feedback=spec, cost_scope="expected costs over measured training branches and final readout",
                      scope="independent fresh shots; native gate/readout noise as specified; optional clock is a serial model")
    try:
        for i, (stage, bindings) in enumerate(stages.items()):
            pool[stage], selected[stage], references[stage] = {}, {}, {}
            for j, (key, (n, ops)) in enumerate(bindings.items()):
                # Numeric directories keep user-supplied stage and binding IDs out of filesystem paths.
                work = out / "native" / f"stage-{i}" / f"binding-{j}"
                if replay:
                    selected[stage][key] = stored.bind(algorithm, task, stage, ops, n, work)
                    original_work = out / "reference" / f"stage-{i}" / f"binding-{j}"
                    if reference_policy:
                        references[stage][key] = reference_policy.bind(algorithm, task, stage, ops, n, original_work)
                    elif original_algorithm:
                        calls, issues = original_algorithm.lower_instrument(ops, n, original_work, profiles=envelope)
                        pool[stage][key] = original_binding(Binding(n, ops, tuple(calls)))
                        rejected.extend(dict(stage=stage, binding=key, detail=v) for v in issues)
                else:
                    calls, issues = algorithm.lower_instrument(ops, n, work, profiles=profiles,
                                                              allow_wire_relabel=args.wire_relabel)
                    pool[stage][key] = Binding(n, ops, tuple(calls))
                    rejected.extend(dict(stage=stage, binding=key, detail=v) for v in issues)
        if replay:
            policy = stored
            report["scope_of_replay"] = "Recompiled extracted actions at supplied bindings; per-call guards checked"
            if algorithm.changed:
                if reference_policy:
                    original_costs, original_actions = evaluate(references), reference_policy.actions
                    source = "saved original policy rebound at the supplied angles"
                else:
                    anchor = extract_policy(original_algorithm, task, pool, evaluate,
                                            objective_order=stored.objective_order, max_evaluations=256)
                    if anchor.policy is None:
                        raise ValueError("Cannot establish a matched original policy for this legacy strategy")
                    original_costs, original_actions = anchor.score, anchor.policy.actions
                    source = "original policy reselected at supplied angles for a legacy strategy"
                report["nonregression"] = dict(protected_costs=tuple(k for k in allowed if k in original_costs),
                    reference_costs=original_costs, reference_actions={s:asdict(a) for s,a in original_actions.items()},
                    scope=source)
        else:
            reference_actions = None
            protected = ()
            if algorithm.changed:
                reference_pool = {s:{k:original_binding(b)
                    for k,b in row.items()} for s,row in pool.items()}
                anchor = extract_policy(algorithm, task, reference_pool, evaluate, objective_order=goal,
                                        max_evaluations=args.budget)
                if anchor.policy is None:
                    report.update(status="reference_unavailable", search_detail=anchor.status)
                    save(out/"result.json", report)
                    return 2
                reference_actions = anchor.policy.actions
                protected = tuple(k for k in allowed if k in anchor.score)
                report["nonregression"] = dict(protected_costs=protected, reference_costs=anchor.score,
                    reference_actions={s:asdict(a) for s,a in reference_actions.items()},
                    scope="computed model endpoints; unmeasured physical endpoints remain unresolved")
            extraction = extract_policy(algorithm, task, pool, evaluate, objective_order=goal,
                                        success_floor=job.floor if job else 0., max_evaluations=args.budget,
                                        limits=({"max_circuit_process_infidelity":1-job.fidelity_floor}
                                                if job and job.fidelity_floor is not None else None),
                                        reference=reference_actions, protected_costs=protected)
            report["search"] = dict(profiles=profiles, goal=goal, budget=args.budget,
                                    evaluated=extraction.evaluated, space_size=extraction.space_size,
                                    status=extraction.status, exhaustive=extraction.exhaustive,
                                    rejections=[str(x) for x in extraction.rejected])
            policy = extraction.policy
            report["search"]["coverage"] = search_summary(extraction.evaluated, extraction.space_size,
                has_policy=policy is not None, scores_complete=True)
            if policy is None:
                report["status"] = "no_admissible_policy"
                save(out/"result.json", report)
                return 2
            for stage, bindings in pool.items():
                for key, binding in bindings.items():
                    selected[stage][key] = policy.choose(algorithm, task, stage, binding.ops, binding.n, binding.calls)
        if job:
            evaluated, report["feedback_assessment"] = job.assess(selected)
            report["whole_job_model"] = evaluated
            if "max_circuit_process_infidelity" in evaluated:
                report["circuit_process_fidelity"] = 1-evaluated["max_circuit_process_infidelity"]
                report["process_scope"] = "minimum normalized common-port Choi fidelity across all declared bindings before measurement"
            report["native_cost_subtotals"] = {k:evaluated[k] for k in
                ("duration_us", "gate_count", "gate_dependency_depth", "peak_data_atoms")}
        else:
            evaluated = report["native_cost_subtotals"] = costs(selected, shots)
        policy.verify_job(evaluated)
        if replay and "nonregression" in report:
            evidence = report["nonregression"]
            check_nonregression(evaluated, evidence["reference_costs"], evidence["protected_costs"])
        report["selected"] = {s: {k: artifact(call, out, shots[s,k]) for k, call in row.items()}
                              for s, row in selected.items()}
        report.update(status="selected", strategy_file="strategy.json")
        strategy = policy.as_dict()
        strategy["host_profiles"] = list(envelope)
        if "nonregression" in report:
            strategy["reference_actions"] = report["nonregression"]["reference_actions"]
        if job:
            strategy["feedback_contract"] = spec
        save(out/"strategy.json", strategy)
        save(out/"result.json", report)
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        report.update(status="failed", error=str(exc), selected={}, strategy_file=None)
        save(out/"result.json", report)
        raise


def decode(args):
    result = load(args.result)
    if result["status"] != "selected":
        raise ValueError("This run produced no selected native instrument")
    binding = result["selected"][args.stage][args.binding]
    kernel = Readout(**binding["readout"])
    port = WirePort(tuple(binding["wire_order"]))
    slot = port.logical_outcome(args.native)
    free = kernel.sample_index(slot, secrets.randbits) if args.random_bits is None else args.random_bits
    logical = kernel.decode_bits(slot, free)
    print(json.dumps(dict(native_outcome=args.native, native_width=kernel.native_width,
                          random_bits=free, logical_outcome=logical,
                          logical_bitstring=format(logical, f"0{kernel.width}b"), bit_order="q[n-1]..q[0]")))
    return 0


def compare_workflows(off, on):
    if (any(r.get("status") != "selected" for r in (off,on))
            or off.get("rewriting_enabled") is not False or on.get("rewriting_enabled") is not True):
        raise ValueError("Expected selected original and joint results in that order")
    design = matched_contract(off["host_workflow"], on["host_workflow"])
    selected = {}
    for s, row in off["selected"].items():
        selected[s] = {}
        for k, original in row.items():
            joint = on["selected"][s][k]
            fields = ("family", "variant", "profile", "implementation", "readout", "wire_order")
            selected[s][k] = {name:{"off":original[name], "on":joint[name]} for name in fields
                              if original[name] != joint[name]}
    def measured(result):
        return result.get("whole_job_model", result["native_cost_subtotals"])
    return dict(host=off["host"], matched=design["matched"], task_contract=off["task_contract"],
                host_specialties=design["scope"]["active_specialties"], selected_changes=selected,
                off=measured(off), on=measured(on), cost_scope=off["cost_scope"],
                on_search=on.get("search"), observed_hardware_execution=False)


def export_encoded_native(directory, assignment):
    """Emit the selected ZAC traces and portable architecture alongside the policy."""
    root = Path(__file__).resolve().parent
    files = {}
    for theta in (0, 1, -1):
        for basis in ("X", "Z"):
            key = f"{theta}/{basis}"
            arm = assignment[key]
            if arm not in {"verified", "terminal"}:
                raise ValueError("Unrecognized ZAC preparation in encoded plan")
            source = root / "encoded_job" / "runs" / arm / key
            target = directory / "native" / key
            target.mkdir(parents=True)
            for filename in ("bound_schedule.json", "metrics.json", "call.qasm"):
                shutil.copyfile(source / filename, target / filename)
            trace = load(source / "native_trace.json")
            trace["architecture_spec_path"] = "native/full_architecture.json"
            save(target / "native_trace.json", trace)
            files[key] = {"arm": arm, "directory": f"native/{key}"}
    for source, target in (("zac_full_architecture.json", "full_architecture.json"),
                           ("ZAC-LICENSE.txt", "ZAC-LICENSE.txt")):
        shutil.copyfile(root / "packaging" / source, directory / "native" / target)
    save(directory / "native-files.json", {"host": "ZAC", "bindings": files,
         "architecture": "native/full_architecture.json", "trace_path_base": "output directory"})


def encoded_command(args):
    from vqa_joint.encoded_job.select_policy import replay_policy, select_policy

    if args.cmd == "replay-encoded":
        plan = load(args.result)
        if plan.get("status") not in {"optimal_in_catalogue", "best_certified"}:
            raise ValueError("Encoded replay requires a selected plan")
        print(json.dumps(replay_policy(plan, args.seed), indent=2, allow_nan=False))
        return 0

    plan = select_policy(load(args.packet))
    directory = args.out.absolute()
    directory.mkdir(parents=True, exist_ok=False)
    if plan.get("selected") is not None:
        export_encoded_native(directory, plan["selected"]["assignment"])
    save(directory / "result.json", plan)
    print(directory / "result.json")
    return 0 if plan.get("selected") is not None else 2


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="cmd", required=True)
    encoded = commands.add_parser("compile-encoded", help="Select a complete encoded ZAC VQE policy offline")
    encoded.add_argument("packet", type=Path, help="Encoded job and constraint JSON")
    encoded.add_argument("--out", required=True, type=Path, help="Fresh output directory")
    encoded_replay = commands.add_parser("replay-encoded", help="Sample a complete selected encoded VQE job")
    encoded_replay.add_argument("result", type=Path, help="Saved compile-encoded result.json")
    encoded_replay.add_argument("--seed", required=True, type=int)
    compile_cmd = commands.add_parser("compile", help="Generate a host strategy and native schedules")
    replay_cmd = commands.add_parser("replay", help="Compile new bound calls using an extracted strategy")
    for parser in (compile_cmd, replay_cmd):
        parser.add_argument("packet", type=Path)
        parser.add_argument("--out", required=True, type=Path, help="Fresh output directory")
    compile_cmd.add_argument("--host", required=True, choices=tuple(HOSTS))
    group = compile_cmd.add_mutually_exclusive_group()
    group.add_argument("--profile", action="append", help="Repeat to expand the native profile envelope")
    group.add_argument("--all-profiles", action="store_true")
    compile_cmd.add_argument("--goal", default=",".join(DEFAULT_GOAL),
                             help="Ordered costs; feedback packets also support task_failure_probability and modeled_job_time_us")
    compile_cmd.add_argument("--budget", type=int, default=256, help="Maximum whole-policy evaluations after native lowering")
    compile_cmd.add_argument("--original-only", action="store_true", help="Disable joint rewriting")
    compile_cmd.add_argument("--wire-relabel", action="store_true", help="Try the declared wire permutations")
    replay_cmd.add_argument("--strategy", required=True, type=Path)
    pair = commands.add_parser("compare", help="Check two same-host job contracts and show the selected strategies and costs")
    pair.add_argument("off", type=Path)
    pair.add_argument("on", type=Path)
    sample = commands.add_parser("sample-job", help="Run the selected finite feedback job using the declared simulator model")
    sample.add_argument("result", type=Path)
    sample.add_argument("--seed", type=int, required=True)
    shot = commands.add_parser("decode", help="Decode a native measured integer using its selected readout")
    shot.add_argument("result", type=Path)
    shot.add_argument("--stage", required=True)
    shot.add_argument("--binding", required=True)
    shot.add_argument("--native", required=True, type=lambda s: int(s, 0))
    shot.add_argument("--random-bits", type=lambda s: int(s, 0), help="Explicit independent uniform bits for reproducible decoding")
    args = cli.parse_args()
    try:
        if args.cmd in ("compile-encoded", "replay-encoded"):
            return encoded_command(args)
        if args.cmd == "decode":
            return decode(args)
        if args.cmd == "compare":
            print(json.dumps(compare_workflows(load(args.off), load(args.on)), indent=2))
            return 0
        if args.cmd == "sample-job":
            result = load(args.result)
            if result.get("status") != "selected" or "feedback" not in result:
                raise ValueError("Select a feedback strategy before sampling")
            print(json.dumps(sample_saved(result, args.result.parent, args.seed)))
            return 0
        status = execute(args)
        print(args.out / "result.json")
        return status
    except (ValueError, KeyError, OSError, TypeError, RuntimeError) as exc:
        cli.exit(2, f"vqa-joint: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
