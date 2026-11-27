import unittest
from dataclasses import replace

from vqa_harness import (
    Assessment, HostJob, Match, OpenDiagram, Port, ReleaseContract, Stage,
    direct_diagram, search_diagram,
)


class ParallelIdleVerifier:
    model = "parallel-ideal-idles"
    duration = {"long idle": 2, "short idle": 1}

    def assess(self, contract, graph):
        return Assessment("feasible", max(self.duration[s.implementation] for s in graph.stages))

    def upper_bounds(self, contract, reference, candidate):
        return {"duration": candidate.evidence - reference.evidence}

    upper_bounds_against = upper_bounds


class IdentityBudgetTest(unittest.TestCase):
    def test_putting_back_a_parallel_stage_leaves_the_budget_for_a_rewrite(self):
        x, y, u, v = [Port(name, "qubit") for name in "xyuv"]
        left = Stage("left", "idle", (x,), (u,), "long idle")
        right = Stage("right", "idle", (y,), (v,), "short idle")
        host = OpenDiagram((x, y), (left, right), (u, v))
        verifier = ParallelIdleVerifier()
        job = HostJob("preserve each wire", "two independent ideal qubits", host,
                      ("idle",), ReleaseContract(verifier.model, ("duration",)))

        def identity(job, diagram):
            node = next(s for s in diagram.stages if s.id == "left")
            yield Match((node.id,), OpenDiagram(node.inputs, (node,), node.outputs))

        def shorten(job, diagram):
            node = next(s for s in diagram.stages if s.id == "left")
            updated = replace(node, implementation="short idle")
            yield Match((node.id,), OpenDiagram(node.inputs, (updated,), node.outputs))

        keep = (("preserve idle", identity),)
        rules = (*keep, ("shorten idle", shorten))
        host_receipt = direct_diagram(job, keep, verifier, max_rewrites=1)
        self.assertIs(host_receipt["policy"], host)
        self.assertEqual(host_receipt["proposed_steps"], [])
        self.assertEqual(host_receipt["branches"], [])

        direct = direct_diagram(job, rules, verifier, max_rewrites=1)
        search = search_diagram(job, rules, verifier, max_rewrites=1, max_candidates=1,
                                incumbent=host_receipt)
        for result in (direct, search):
            self.assertEqual(result["status"], "released")
            self.assertEqual([s.id for s in result["policy"].stages], ["left", "right"])
            self.assertIs(result["policy"].stages[1], right)
            self.assertEqual(verifier.assess(job.contract, result["policy"]).evidence, 1)
            self.assertEqual(len(result["branches"]), 1)
        self.assertTrue(search["exhausted_to_depth"])
