"""Run the four public compiler entry points on a two-round feedback job."""
from dataclasses import dataclass, replace
from fractions import Fraction
import json
import random

from vqa_harness.diagram_context import OpenDiagram, Port, Stage, replace_region
from vqa_harness.diagram_search import Match, direct_diagram, search_diagram
from vqa_harness.guarded_release import Assessment, ReleaseContract
from vqa_harness.host_compilation import Branch, HostJob, compile_bounded, compile_direct


@dataclass(frozen=True)
class Device:
    # Independent bit flip immediately after each physical X.
    flip_probability: Fraction = Fraction(1, 100)

    def __post_init__(self):
        if not isinstance(self.flip_probability, Fraction) or not 0 <= self.flip_probability <= 1:
            raise ValueError("Register a rational probability between zero and one")


def feedback_job(protect_chip=False):
    wires = tuple(Port(name, typ) for name, typ in [
        ("zero", "basis bit"), ("first", "basis bit"), ("record", "measured bit"),
        ("next", "prepared basis bit"), ("last", "basis bit"), ("answer", "two-bit record"),
    ])
    zero, first, record, next_input, last, answer = wires
    graph = OpenDiagram((zero,), (
        Stage("first-drive", "pulse", (zero,), (first,), "XXX"),
        Stage("feedback", "controller", (first,), (record, next_input), "measure-and-prepare"),
        Stage("second-drive", "pulse", (next_input,), (last,), "XXX"),
        Stage("return", "controller", (record, last), (answer,), "measure-and-return"),
    ), (answer,))
    losses = ("gates", "modeled_answer_failure")
    if protect_chip:
        losses += ("chip_answer_failure",)
    return HostJob((1, 0), Device(), graph, ("pulse",), ReleaseContract("two-round-bit-flip", losses))


def single_x(job, graph):
    """XXX and X have the same ideal action, with distinct native noise costs."""
    for stage in graph.stages:
        if stage.role == "pulse" and stage.implementation == "XXX":
            rewritten = replace(stage, implementation="X")
            yield Match((stage.id,), OpenDiagram(stage.inputs, (rewritten,), stage.outputs))


class FeedbackAdapter:
    def __init__(self, job):
        self.job = job
        self.model = "two-round-bit-flip"

    def verifier(self, job):
        if job != self.job:
            raise ValueError("Verifier was registered for a different job")
        return self

    def construct(self, job, branch):
        chosen = {"first": ("first-drive",), "both": ("first-drive", "second-drive")}[branch.name]
        graph = job.host
        for match in single_x(job, job.host):
            if match.selected[0] in chosen:
                graph = replace_region(graph, match.selected, match.replacement, branch.stages)
        return graph

    def assess(self, contract, graph):
        # The client checks the named physical and classical boundary explicitly.
        original = self.job.host
        if (not isinstance(graph, OpenDiagram) or graph.inputs != original.inputs
                or graph.outputs != original.outputs or len(graph.stages) != len(original.stages)):
            return Assessment("infeasible")
        pulse_counts = []
        for expected, stage in zip(original.stages, graph.stages):
            if stage.role == "pulse":
                if (replace(stage, implementation=expected.implementation) != expected
                        or stage.implementation not in ("X", "XXX")):
                    return Assessment("infeasible")
                pulse_counts.append(len(stage.implementation))
            elif stage != expected:
                return Assessment("infeasible")
        if self.job.task != (1, 0) or contract.caps:
            return Assessment("missing")
        # Success needs the first bit to be 1 and the second to equal 0.
        # Each block must incur an even number of independent error flips.
        probability = Fraction(1)
        for count in pulse_counts:
            probability *= (1 + (1 - 2 * self.job.hardware.flip_probability) ** count) / 2
        return Assessment("feasible", {"gates": sum(pulse_counts), "modeled_answer_failure": 1 - probability})

    def direct_branch(self, job):
        return Branch("first", ("pulse",))

    def upper_bounds(self, contract, original, candidate):
        # Both entries are exact under the fixed error model. Chip risk is unavailable.
        return {key: value - original.evidence[key] for key, value in candidate.evidence.items()}


    # Exact complete-program losses can be compared to any verified policy.
    upper_bounds_against = upper_bounds


def execute(graph, device, seed):
    rng = random.Random(seed)
    bit = 0
    for stage in graph.stages:
        if stage.role == "pulse":
            for _ in stage.implementation:
                bit ^= 1
                bit ^= rng.random() < device.flip_probability
        elif stage.implementation == "measure-and-prepare":
            first = bit
        elif stage.implementation == "measure-and-return":
            return first, bit
    raise ValueError("Incomplete feedback controller")


def compile_all(job):
    adapter = FeedbackAdapter(job)
    rules = (("replace XXX with X", single_x),)
    first, both = Branch("first", ("pulse",)), Branch("both", ("pulse",))
    direct = compile_direct(job, first, adapter)
    derivation = direct_diagram(job, rules, adapter, max_rewrites=2)
    return {
        "compile_direct": direct,
        "compile_bounded": compile_bounded(job, (first, both), 2, adapter, incumbent=direct),
        "direct_diagram": derivation,
        "search_diagram": search_diagram(job, rules, adapter, max_rewrites=2, max_candidates=3,
                                         incumbent=derivation),
    }


def demo():
    results = {}
    for protect_chip in (False, True):
        job = feedback_job(protect_chip)
        mode = "require_chip_bound" if protect_chip else "specified_noise_model"
        results[mode] = {}
        for name, receipt in compile_all(job).items():
            program = receipt["policy"]
            assessment = FeedbackAdapter(job).assess(job.contract, program)
            results[mode][name] = {
                "status": receipt["status"], "branch_decisions": [r["decision"] for r in receipt["branches"]],
                "same_host": program is job.host, "native_x": assessment.evidence["gates"],
                "modeled_failure": str(assessment.evidence["modeled_answer_failure"]),
                "sampled_after_release": execute(program, job.hardware, seed=18),
            }
    return results


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))
