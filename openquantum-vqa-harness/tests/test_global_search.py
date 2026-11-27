"""Exact single-qubit regression for search through a costlier intermediate."""
import unittest
import numpy as np

from vqa_harness.diagram_context import OpenDiagram, Port, Stage
from vqa_harness.diagram_search import Match, direct_diagram, search_diagram
from vqa_harness.guarded_release import Assessment, ReleaseContract
from vqa_harness.host_compilation import HostJob

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]])
Z = np.diag([1, -1])
MATRICES = {"H": H, "X": X, "Z": Z, "wire": np.eye(2)}
P = [Port(f"q{i}", "qubit in the lab frame") for i in range(4)]


def gate(name, op, left, right):
    return Stage(name, "native evolution", (left,), (right,), op)


HOST = OpenDiagram((P[0],), (gate("a", "H", P[0], P[1]),
                            gate("b", "X", P[1], P[2]),
                            gate("c", "H", P[2], P[3])), (P[3],))


def expand_x(job, graph):
    for node in graph.stages:
        if node.implementation == "X":
            left, right = node.inputs[0], node.outputs[0]
            p, q = Port("tmp1", left.space), Port("tmp2", left.space)
            inside = (gate("e1", "H", left, p), gate("e2", "Z", p, q),
                      gate("e3", "H", q, right))
            yield Match((node.id,), OpenDiagram((left,), inside, (right,)))


def cancel_hh(job, graph):
    for a, b in zip(graph.stages, graph.stages[1:]):
        if a.implementation == b.implementation == "H" and a.outputs == b.inputs:
            wire = gate("pair_" + a.id + b.id, "wire", a.inputs[0], b.outputs[0])
            yield Match((a.id, b.id), OpenDiagram(a.inputs, (wire,), b.outputs))


class Verifier:
    model = "ideal-qubit-native-count"

    def assess(self, contract, graph):
        u = np.eye(2)
        n = 0
        for node in graph.stages:
            u = MATRICES[node.implementation] @ u
            n += int(node.implementation != "wire")
        if not np.allclose(u, Z, atol=1e-12, rtol=0):
            return Assessment("infeasible")
        return Assessment("feasible", n)

    def upper_bounds(self, contract, original, candidate):
        return {"gates": candidate.evidence - original.evidence}


class DiagramSearchTest(unittest.TestCase):
    def setUp(self):
        self.job = HostJob("implement Z on any input", "one ideal qubit", HOST,
                           ("native evolution",), ReleaseContract(Verifier.model, ("gates",)))
        self.rules = (("expand X", expand_x), ("cancel H pair", cancel_hh))

    def test_full_program_gain_through_an_expensive_intermediate(self):
        direct = direct_diagram(self.job, self.rules, Verifier(), 3)
        full = search_diagram(self.job, self.rules, Verifier(), 3, 20, legacy_host_only=True)
        for result in direct, full:
            self.assertEqual(result["status"], "released")
            self.assertEqual(Verifier().assess(self.job.contract, result["policy"]).evidence, 1)
        self.assertEqual(len(direct["branches"]), 1)
        self.assertEqual(len(full["selected_steps"]), 3)
        self.assertTrue(full["exhausted_to_depth"])
        self.assertEqual(full["branches"][0]["decision"], "non_regression_unproved")

    def test_budget_truncation_and_finite_depth_are_distinct(self):
        truncated = search_diagram(self.job, self.rules, Verifier(), 3, 1, legacy_host_only=True)
        self.assertIs(truncated["policy"], HOST)
        self.assertFalse(truncated["exhausted_to_depth"])
        covered = search_diagram(self.job, self.rules, Verifier(), 2, 20, legacy_host_only=True)
        self.assertIs(covered["policy"], HOST)
        self.assertTrue(covered["exhausted_to_depth"])

    def test_fixed_point_receipt_distinguishes_a_cap_from_a_closed_domain(self):
        def search(cap):
            return search_diagram(self.job, self.rules, Verifier(), None, cap, legacy_host_only=True)

        full = search(20)
        self.assertTrue(full["registered_closure_complete"])
        self.assertEqual(Verifier().assess(self.job.contract, full["policy"]).evidence, 1)
        self.assertTrue(search(full["constructed"])["registered_closure_complete"])
        self.assertFalse(search(full["constructed"] - 1)["registered_closure_complete"])
