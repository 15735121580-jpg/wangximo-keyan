"""Matched VQA workflow manifests. A manifest assigns responsibilities; execution needs run evidence."""
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path


# Stage identifiers denote functions of a job. They are not a required serial schedule.
# In particular factory supply can overlap data preparation and syndrome can repeat.
CONVENTIONAL = {
    "problem": "Bind the specified Hamiltonian or cost, ansatz, input, output and stopping rule.",
    "preparation": "Reset fresh atoms in the computational basis, then apply the given input-preparation circuit.",
    "encoding": "Prepare the chosen logical code block using a compatible, verified encoder.",
    "logical_synthesis": "Bind the declared ansatz gates, use the device basis decompositions and synthesize rotations at the agreed tolerance.",
    "placement": "Assign named blocks and ancillary atoms to distinct available compatible sites in deterministic site order.",
    "transport": "Route each ready entangler in sequence through legal interaction sites, restore the agreed exit layout and check move constraints.",
    "one_qubit_control": "Emit basis one-qubit gates in dependency order using the device addressing masks and native pulse timings.",
    "entangler_schedule": "Schedule dependency-ready, endpoint-disjoint entanglers under the device blockade and occupancy constraints.",
    "syndrome_extraction": "Execute the declared code check rounds with fresh or reset ancillas, then retain syndromes and available loss flags.",
    "decoding": "Apply the code-matched syndrome decoder (for example a CSS lookup table) and update the Pauli frame before interpreting logical outcomes.",
    "magic_supply": "Supply verified encoded T states using compatible preparation and, when required, an agreed distillation block such as 15-to-1.",
    "factory_acceptance": "Apply the factory herald rule and replace rejected attempts under the retry policy.",
    "injection": "Use gate teleportation with the accepted resource, measuring the required parity and applying its conditional Clifford correction.",
    "readout": "Use the observable basis changes followed by computational-basis imaging and the shared readout model.",
    "estimation": "Decode task observables and form the declared estimator from the actual shots.",
    "feedback": "Choose the next binding, shot count and stopping decision from the observed history.",
    "delivery": "Measure independent final shots when required and score the contracted output.",
}

# Scope of the inspected implementation or paper protocol, independent of the current job.
# A host may supply only part of a stage; the named implementation documents that part.
_HOSTS = {
    "zap": (
        "author_code", "hosts/README.md",
        {"placement": "ZAP zoned placement", "transport": "ZAP zoned AOD routing",
         "one_qubit_control": "ZAP emitted single-qubit gate schedules",
         "entangler_schedule": "ZAP native Rydberg layers"},
        "Local author software and discrete AOD audit. Continuous trajectories and hardware timing need separate evidence."),
    "zac": (
        "author_code", "hosts/README.md",
        {"logical_synthesis": "ZAC native frontend resynthesis of bound unitary calls",
         "placement": "ZAC zoned placement and interlayer reuse", "transport": "ZAC dynamic movement routing",
         "one_qubit_control": "ZAC native one-qubit gate scheduling", "entangler_schedule": "ZAC native entangler scheduling"},
        "Author placement/scheduling checks. The zero-entangler single-site extension must be identified separately in its run evidence."),
    "powermove": (
        "author_code_with_project_binding", "new_host/README.md",
        {"placement": "PowerMove single-AOD annealed placement", "transport": "PowerMove grouped single-AOD routing",
         "entangler_schedule": "PowerMove CZ-limited blocks and zoned schedule"},
        "Project binds one-qubit angles. The published run subtotal covers transport; full device timing requires further evidence."),
    "enola": (
        "author_code_with_project_binding", "extended/enola.py",
        {"placement": "Enola native grid placement", "transport": "Enola native movement routing",
         "entangler_schedule": "Enola native CZ scheduling"},
        "The project binds and replays local gates separately. No physical schedule for these gates is certified here."),
    "structure": (
        "project_paper_reconstruction", "extended/structure.py",
        {"logical_synthesis": "Structure-Aware star-forest decomposition of one-layer MaxCut",
         "one_qubit_control": "Structure-Aware one-row addressing masks",
         "entangler_schedule": "Structure-Aware endpoint-exclusive CZ groups",
         "transport": "Structure-Aware one-row interval transport groups"},
        "Local reconstruction of the published one-row model. Author software, continuous routes and full timing are unverified."),
    "grail": (
        "paper_protocol_with_task_specific_reconstruction", "../../outputs/oq-physical-ft/dual_species/README.md",
        {"transport": "GRAIL published addressing primitives moving Rb relative to stationary Cs",
         "one_qubit_control": "GRAIL global microwave control with Rb addressing of Cs",
         "entangler_schedule": "GRAIL species-dependent Rydberg interaction primitives",
         "syndrome_extraction": "GRAIL dual-species ZXXZ syndrome protocols",
         "decoding": "GRAIL published memory/logical-gate decoder prescription",
         "magic_supply": "GRAIL nine-stage cultivation",
         "factory_acceptance": "GRAIL cultivation hard checks and complementary-gap selection"},
        "Requires compatible Cs/Rb controls, noise and measurements. Workspace reconstructions cover named protocols; author full VQA implementation is unverified."),
    "caliper": (
        "paper_protocol", "../../outputs/oq-physical-ft/sources/grail-caliper.md",
        {"factory_acceptance": "Caliper decoding and postselection during magic-state escape"},
        "Evidence covers the published code-family escape protocol and conditional acceptance. It supplies no neutral-atom placement, transport or complete VQA backend."),
    "bridge": (
        "project_paper_reconstruction", "relay_jobs/README.md",
        {"transport": "Dual-species buffer relay under the selected data-motion mode",
         "entangler_schedule": "Schedule the native buffer-mediated interactions"},
        "Four paired small-cell jobs use a reconstructed protocol and a model of pulse windows. Device-wide timing is uncalibrated."),
    "hybrid_remote_cz": (
        "project_paper_reconstruction", "relay_jobs/README.md",
        {"transport": "Direction-constrained neutral-atom transport",
         "entangler_schedule": "Choice of native transported or remote CZ under the shared hardware constraints"},
        "Four paired small-cell jobs use the reconstructed static remote protocol. Assembly, return and device channel calibration are outside the pulse window."),
    "mobius": (
        "project_paper_reconstruction", "single_hosts/mobius_jobs.py",
        {"logical_synthesis": "Diagonal gate synthesis using available native multiatom controlled phases"},
        "Four paired tasks use a common four-site zoned completion and the specified multiatom phase model."),
    "atomique": (
        "author_code_with_project_binding", "structure_jobs/atomique/paired_summary.json",
        {"placement": "Map algorithm qubits to reconfigurable atom arrays",
         "transport": "Coordinate permitted array reconfiguration and atom movement",
         "entangler_schedule": "Emit hardware-supported entangling layers"},
        "Pinned author compiler with angle-bound replay, discrete layout audit and a gate and movement clock model."),
    "pulser": (
        "simulation_with_project_controller", "analog/README.md",
        {"logical_synthesis": "Represent the bound analog controls with a Pulser pulse sequence",
         "placement": "Bind the graph to occupied traps in the supplied register",
         "one_qubit_control": "Execute the declared global drive and detuning waveforms",
         "entangler_schedule": "Evolve under the distance-dependent Rydberg interaction Hamiltonian"},
        "The verified host instance is a fixed-register analog MIS simulator. Pulse metrics have analog units."),
}

# The single-stage Caliper component is available for eligible factory protocols.
BENCHMARK_HOSTS = ("zap", "zac", "grail", "structure", "enola", "powermove", "atomique",
                   "bridge", "hybrid_remote_cz", "mobius", "pulser")

# These implementations have actually been connected for specific workloads. The task
# and platform in a new manifest still have to be matched to a run before execution is claimed.
_COMPONENTS = {
    "encoding": ("encoded_job/circuits.py", "one-block Steane input; verified variant in encoded_job/verified.py"),
    "syndrome_extraction": ("encoded_job/circuits.py", "one-block Steane extraction and terminal checks"),
    "decoding": ("encoded_job/channel.py", "finite syndrome table for the specified Steane instrument"),
    "readout": ("native_noise.py", "modeled terminal measurement of supported bound native calls"),
    "estimation": ("feedback_job.py", "finite-shot scoring for the supported task schema"),
    "feedback": ("feedback_job.py", "finite feedback controller for supported job descriptions"),
    "delivery": ("feedback_job.py", "final output law for the supported terminal task"),
}
_ENCODED = {"encoding", "syndrome_extraction", "decoding"}
_FACTORY = {"magic_supply", "factory_acceptance", "injection"}
HOST_NAMES = tuple(_HOSTS)
SELECTION_RULE = {"domain":"complete_job", "admissible":"no_regression_on_all_applicable_outcomes",
                  "reference":"retain_if_tied_or_no_certified_improvement", "missing":"unresolved"}


def _snapshot(value):
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def build_scope(host, *, task, platform, priorities, encoded=False, requires_magic=False, conventional=None):
    """Build a JSON manifest. Pass distinct task/platform specifications and ordered user priorities.

    requires_magic means this encoded implementation uses injection. For other encoded
    choices, such as transversal Clifford tasks, factory stages remain inapplicable.
    conventional maps applicable nonspecialist stage IDs to common implementation IDs.
    """
    host = _text(host, "host").lower()
    host = {"structure-aware": "structure", "power-move": "powermove"}.get(host, host)
    if host not in _HOSTS:
        raise ValueError(f"Unknown host {host}; choose from {HOST_NAMES}")
    if type(encoded) is not bool or type(requires_magic) is not bool or (requires_magic and not encoded):
        raise ValueError("Magic-state injection requires an encoded job with boolean applicability flags")
    if host == "pulser" and encoded:
        raise ValueError("The current fixed-register analog adapter has no encoded workflow")
    if isinstance(priorities, str):
        raise ValueError("Priorities must be an ordered sequence")
    order = [_text(item, "priority") for item in priorities]
    if not order or len(set(order)) != len(order):
        raise ValueError("Specify distinct priorities in their required order")
    task, platform = _snapshot(task), _snapshot(platform)
    if not all((isinstance(v, str) and v.strip()) or (isinstance(v, dict) and v) for v in (task, platform)):
        raise ValueError("Task and platform require nonempty identifiers or specifications")
    tier, source, specialty, boundary = _HOSTS[host]
    conventional = dict(conventional or {})
    stages = {}
    for stage, default in CONVENTIONAL.items():
        active = (encoded or stage not in _ENCODED) and (requires_magic or stage not in _FACTORY)
        if host == "pulser" and stage == "transport":
            active = False
        role = "specialty" if stage in specialty else "conventional"
        overridden = stage in conventional
        if overridden and not active:
            raise ValueError(f"Conventional override for inactive stage {stage}")
        protocol = specialty[stage] if role == "specialty" else conventional.pop(stage, default)
        component = _COMPONENTS.get(stage) if role == "conventional" and not overridden else None
        if not overridden and stage == "one_qubit_control" and host in ("enola", "powermove"):
            component = (source, "Project binds local gates for replay. Physical one-qubit scheduling is unverified.")
        if role == "specialty" and tier != "paper_protocol":
            component = (source, boundary)
        stages[stage] = {
            "applicable": active, "role": role, "protocol": _text(protocol, stage),
            "basis": tier if role == "specialty" else "conventional_protocol",
            "execution": "not_executed" if active else "not_applicable",
            "available_component": dict(source=component[0], scope=component[1]) if component else None,
        }
    if conventional:
        raise ValueError(f"Unknown or specialty stage overrides: {sorted(conventional)}")
    return dict(version=1, host=host, task=task, platform=platform, priorities=order,
                encoded=encoded, requires_magic=requires_magic,
                host_source=source, host_boundary=boundary, stages=stages,
                active_specialties=[name for name in specialty if stages[name]["applicable"]],
                selection_rule=dict(SELECTION_RULE),
                execution_status="declaration_only")


def _active_records(records, stages, field):
    records = _snapshot(records or {})
    if not isinstance(records, dict):
        raise ValueError(f"{field} must map stage IDs to records")
    for name, value in records.items():
        if name not in stages or not stages[name]["applicable"]:
            raise ValueError(f"{field} mentions unknown or inactive stage {name}")
        if not value:
            raise ValueError(f"Empty {field} for {name}")
    return records


def matched_contract(off_scope, on_scope, *, off_actions=None, on_actions=None, rewrites=(),
                     off_evidence=None, on_evidence=None):
    """Validate the shared comparison and emit only recorded arm differences.

    Actions name actual or planned choices under the common implementations. Rewrites
    contain 'stages', 'off', 'on' and 'relation' (an identity or certificate reference).
    An evidence record has a 'kind' and a 'source' identifying a caller-supplied artifact.
    Evidence is reported as supplied, without promotion to a verified measurement.
    """
    for scope in (off_scope, on_scope):
        if not isinstance(scope, dict) or scope.get("version") != 1:
            raise ValueError("Unsupported workflow scope version")
        if scope.get("execution_status") != "declaration_only" or scope.get("host") not in _HOSTS:
            raise ValueError("Expected a declared comparison scope for a registered host")
        if not isinstance(scope.get("stages"), dict) or set(scope["stages"]) != set(CONVENTIONAL):
            raise ValueError("The declared scope must cover every workflow stage")
    for field in ("host", "task", "platform", "priorities", "encoded", "requires_magic",
                  "stages", "host_source", "host_boundary", "active_specialties", "selection_rule"):
        if field not in off_scope or field not in on_scope or off_scope[field] != on_scope[field]:
            raise ValueError(f"Unmatched comparison field: {field}")
    stages = off_scope["stages"]
    off = _active_records(off_actions, stages, "off_actions")
    on = _active_records(on_actions, stages, "on_actions")
    if set(off) != set(on):
        raise ValueError("Record both arm actions for each compared stage")
    diffs = {s: dict(off=off[s], on=on[s]) for s in off if off[s] != on[s]}
    interfaces = []
    for rule in rewrites:
        rule = _snapshot(rule)
        names = rule.get("stages") if isinstance(rule, dict) else None
        if not isinstance(names, list) or len(names) < 2 or len(set(names)) != len(names):
            raise ValueError("A cross-interface rewrite requires distinct participating stages")
        for name in names:
            if name not in stages or not stages[name]["applicable"]:
                raise ValueError(f"Inactive rewrite stage {name}")
        for key in ("off", "on", "relation"):
            _text(rule.get(key), key)
        if rule["off"] == rule["on"]:
            raise ValueError("A rewrite must name the changed interface")
        interfaces.append(rule)
    evidence = {}
    for arm, rec in (("off", off_evidence), ("on", on_evidence)):
        rec = _active_records(rec, stages, arm+"_evidence")
        for value in rec.values():
            if not isinstance(value, dict):
                raise ValueError("Stage evidence needs a kind and source")
            _text(value.get("kind"), "evidence kind")
            source = _text(value.get("source"), "evidence source")
            if value["kind"] == "native_schedule":
                schedule = json.loads(Path(source).read_text())
                gates = schedule.get("gates") if isinstance(schedule, dict) else None
                width = schedule.get("num_qubits") if isinstance(schedule, dict) else None
                if not isinstance(gates, list) or type(width) is not int or width < 1:
                    raise ValueError("Native schedule must contain a width and replayable gate list")
                if not all(isinstance(g, dict) and isinstance(g.get("gate"), str) and g["gate"]
                           and isinstance(g.get("qubits"), list) and g["qubits"]
                           and all(type(q) is int and 0 <= q < width for q in g["qubits"])
                           and len(set(g["qubits"])) == len(g["qubits"]) for g in gates):
                    raise ValueError("Malformed native gate record")
                value["observed_record"] = {"num_qubits": schedule["num_qubits"], "gate_count": len(gates)}
            else:
                value["observed_record"] = None
        evidence[arm] = rec
    return dict(matched=True, scope=_snapshot(off_scope), action_differences=diffs,
                interface_rewrites=interfaces, evidence=evidence,
                execution_status="stage_evidence_supplied" if any(evidence.values()) else "declaration_only")


def search_summary(evaluated, space_size, *, has_policy, scores_complete=False):
    """Describe coverage of the supplied finite pool for the declared objective order.

    scores_complete means every visited policy has a comparable score or certified
    exclusion. The caller supplies this from its evaluator, not from a stage manifest.
    """
    if type(evaluated) is not int or type(space_size) is not int or not 0 <= evaluated <= space_size:
        raise ValueError("Invalid finite search counts")
    if type(has_policy) is not bool or type(scores_complete) is not bool:
        raise ValueError("Search evidence requires boolean status flags")
    if has_policy and evaluated == 0:
        raise ValueError("An unevaluated pool cannot yield a selected policy")
    exhaustive = evaluated == space_size and space_size > 0
    if not has_policy:
        status = "no_selected_policy"
    elif exhaustive and scores_complete:
        status = "optimal_in_declared_finite_pool"
    elif not exhaustive:
        status = "partial_search_in_declared_pool"
    else:
        status = "exhaustive_with_missing_scores"
    return dict(status=status, evaluated=evaluated, space_size=space_size,
                exhaustive=exhaustive, scores_complete=scores_complete, has_policy=has_policy)
