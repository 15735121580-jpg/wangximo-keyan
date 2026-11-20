"""Synthesize a callable compilation algorithm from a typed host workflow."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable

from vqa_joint.contract import Assessment, OBJECTIVES, compare
from vqa_joint.hosts.native import available_profiles, compile_qasm
from vqa_joint.rewrite import normalize, standard, variants, qasm
from vqa_joint.ports import WirePort, port_candidates

@dataclass(frozen=True)
class Arrow:
    name: str
    source: str
    target: str

@dataclass(frozen=True)
class HostGraph:
    name: str
    arrows: tuple[Arrow, ...]
    profiles: tuple[str, ...]
    native_compile: Callable
    joint_rules: Callable = variants

    def check(self):
        if not self.arrows or not self.profiles:
            raise ValueError("A host needs a workflow and at least one legal control profile")
        for left, right in zip(self.arrows,self.arrows[1:]):
            if left.target != right.source:
                raise ValueError(f"Incompatible ports: {left.target}, {right.source}")

def author_host(name):
    return HostGraph(name, (
        Arrow("generate", "LogicalCall", "EquivalentCircuits"),
        Arrow("commit", "EquivalentCircuits", "Circuit"),
        Arrow("native", "Circuit", "Schedule"),
        Arrow("apply", "Schedule", "Instrument"),
    ), available_profiles(name), compile_qasm)

@dataclass(frozen=True)
class LoweredCall:
    implementation: str
    profile: str
    schedule: Path
    native_metrics: dict
    wire_port: WirePort

@dataclass(frozen=True)
class GeneratedAlgorithm:
    host: HostGraph
    diagram: tuple[Arrow,...]
    changed: bool

    def lower_call(self, ops, n, directory, profiles=None, mode_filter=None,
                   allow_wire_relabel=False):
        """Lower candidates at the rewritten interface, preserve the original.

        Returns the pool and all rejected outcomes. Native schedule certification
        here covers operator identity and the adapter's declared geometry checks;
        whole-task correctness, Choi fidelity and FT claims are evaluated later.
        """
        checked_profiles = self.host.profiles if profiles is None else tuple(profiles)
        if not checked_profiles or not set(checked_profiles) <= set(self.host.profiles):
            raise ValueError("Unknown or empty host profile set")
        choices = (self.host.joint_rules(ops, n) if self.changed else
                   {"original": standard(normalize(ops))})
        if mode_filter is not None:
            choices = {k:v for k,v in choices.items() if k in mode_filter or k=="original"}
        pool, rejected = [], []
        identity = WirePort(tuple(range(n)))
        ports = tuple(port_candidates(n)) if self.changed and allow_wire_relabel else (identity,)
        for (mode, instructions), profile, port in product(choices.items(), checked_profiles, ports):
            name = mode if port == identity else mode+"__port_"+"-".join(map(str,port.physical_of_logical))
            out = Path(directory)/profile/name
            try:
                metrics = self.host.native_compile(self.host.name, qasm(port.native_word(instructions),n),
                                                   out, profile=profile)
                if metrics["geometry_findings"] or metrics["unitary_operator_error"] is None or metrics["unitary_operator_error"] > 1e-8:
                    rejected.append((name,profile,"schedule certificate",metrics))
                    continue
            except (ValueError, RuntimeError, OSError) as exc:
                rejected.append((name,profile,"native compilation",str(exc)))
                continue
            pool.append(LoweredCall(name,profile,out/"bound_schedule.json",metrics,port))
        return pool,rejected

    def select_complete(self, candidates, reference, success_floor, objective_order,
                        minimum_process_fidelity=None, maximum_logical_error=None):
        """Choose a complete controller; candidates are scored by a job evaluator.

        This certified route retains the reference unless a task-qualified joint
        candidate is demonstrably non-worse on all six complete-job outputs.
        No missing physical measurement is inferred by the controller.
        """
        base_id, base = reference
        if not isinstance(base,Assessment):
            raise TypeError("Expected whole-job assessments")
        if sorted(objective_order) != sorted((*base.resource_objectives,'logical_error')):
            raise ValueError("Specify the platform resource priorities and logical-error coordinate exactly once")
        if minimum_process_fidelity is not None and not 0 <= minimum_process_fidelity <= 1:
            raise ValueError("Invalid required circuit process fidelity")
        if maximum_logical_error is not None and not 0 <= maximum_logical_error <= 1:
            raise ValueError("Invalid maximum logical-error probability")
        def passes_floor(a):
            if a.success_probability is None or a.success_probability.low < success_floor:
                return False
            if maximum_logical_error is not None:
                p = a.objectives['logical_error']
                if p is None or p.high > maximum_logical_error:
                    return False
            if minimum_process_fidelity is None:
                return True
            bound = a.objectives["circuit_process_infidelity"]
            return bound is not None and bound.high <= 1 - minimum_process_fidelity
        qualified = [(key,assess) for key,assess in candidates
                     if compare(assess,base,success_floor)["certified_pareto"] and passes_floor(assess)]
        if not qualified:
            if passes_floor(base):
                return base_id, "retained reference; no certified improvement"
            return None, "no certified paired decision; reference does not meet the evidenced thresholds"
        relevant = tuple(k for k in objective_order if base.encoding == 'encoded' or k != 'logical_error')
        key, _ = min(qualified, key=lambda row: tuple(row[1].objectives[k].high for k in relevant))
        return key, "task threshold and dominance on the applicable outputs certified under the supplied evidence"

    def lower_instrument(self, ops, n, directory, profiles=None, mode_filter=None,
                         allow_wire_relabel=False):
        """Generate and lower monitored alternatives with an exact per-shot readout."""
        from vqa_joint.sector_dispatch import lower_instrument
        return lower_instrument(self, ops, n, directory, profiles, mode_filter, allow_wire_relabel)

def synthesize(host, enabled=True):
    """Push native lowering through the family of logically equivalent choices.

    Native interpretation acts on each candidate; the final control policy
    can use the resulting physical schedule and the complete VQA acceptance rule.
    """
    host.check()
    graph = list(host.arrows)
    changed = False
    if enabled:
        for i in range(len(graph)-1):
            if graph[i] == Arrow("commit","EquivalentCircuits","Circuit") and graph[i+1] == Arrow("native","Circuit","Schedule"):
                graph[i:i+2] = [
                    Arrow("lift_native","EquivalentCircuits","EquivalentSchedules"),
                    Arrow("extract_policy","EquivalentSchedules","Schedule"),
                ]
                changed = True
    return GeneratedAlgorithm(host, tuple(graph), changed)
