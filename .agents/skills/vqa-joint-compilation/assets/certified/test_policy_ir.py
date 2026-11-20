import unittest

import z3

from feedback_vqe import (FeedbackBackend, analytic_optimum, build_diagrams,
                          solve_feedback_vqe, symbolic_fixture)
from optimal_compiler import Diagram, solve
from policy_ir import (Action, Decision, Outcome, PublicPort, RecordedFailure,
                       Unreachable, compile_diagrams, dead_leaves, enumerate_policies,
                       synthesize)


class BranchBackend:
    def is_terminal(self, port, history):
        return port.type_name == "done"

    def actions(self, port, history):
        if port.type_name == "start":
            return (Action("measure", "start", tuple(
                Outcome(o, PublicPort("same-port")) for o in ("+", "-"))),)
        if port.type_name == "same-port":
            return tuple(Action(a, "same-port", (Outcome("ok", PublicPort("done")),))
                         for a in ("left", "right"))
        return ()

    def lower(self, tree):
        # Explicit duration and full-output semantics are a backend responsibility.
        # Here the lowering deliberately models an abstract zero-time check.
        return Diagram("policy", (), z3.BoolVal(True), z3.RealVal(0), "abstract")


class PolicyTests(unittest.TestCase):
    def test_distinct_choices_even_when_successor_ports_coincide(self):
        trees = list(enumerate_policies(BranchBackend(), PublicPort("start"), 2))
        choices = {tuple(branch.action.name for _, branch in t.branches) for t in trees}
        self.assertEqual(choices, {("left", "left"), ("left", "right"),
                                   ("right", "left"), ("right", "right")})
        for t in trees:
            self.assertFalse(list(dead_leaves(t)))
            h = tuple(child.history for _, child in t.branches)
            self.assertNotEqual(h[0], h[1])

    def test_horizon_is_explicit_and_guarded_in_both_models(self):
        backend, root = BranchBackend(), PublicPort("start")
        tree, = enumerate_policies(backend, root, 1)
        self.assertEqual([e.kind for e in dead_leaves(tree)], ["dead_horizon"] * 2)
        with self.assertRaisesRegex(ValueError, "unresolved dead leaf"):
            compile_diagrams(backend, root, 1)
        bad = lambda tree, leaf, d: Unreachable(z3.RealVal(0), z3.RealVal(1) / 2)
        self.assertEqual(synthesize(backend, root, 1, bad).status, "infeasible")
        recorded = lambda tree, leaf, d: RecordedFailure(
            z3.BoolVal(True), *([z3.RealVal(1) / 2] * 4))
        self.assertEqual(synthesize(backend, root, 1, recorded).status, "optimal")
        undercredited = lambda tree, leaf, d: RecordedFailure(
            z3.BoolVal(True), z3.RealVal("1/2"), z3.RealVal("1/2"),
            z3.RealVal("1/2"), z3.RealVal(0))
        self.assertEqual(synthesize(backend, root, 1, undercredited).status, "infeasible")

    def test_dead_end_and_broken_lowering(self):
        backend = BranchBackend()
        tree, = enumerate_policies(backend, PublicPort("no-actions"), 1)
        self.assertEqual(tree.kind, "dead_end")
        class EmptyLower(BranchBackend):
            def lower(self, tree):
                return ()
        with self.assertRaisesRegex(ValueError, "nonempty tuple"):
            compile_diagrams(EmptyLower(), PublicPort("done"), 1)

    def test_full_vqe_witness(self):
        catalog, budgets = symbolic_fixture()
        prefix = PublicPort("start")
        trees = list(enumerate_policies(FeedbackBackend(catalog, budgets), prefix, 2))
        self.assertEqual(len(trees), 2)
        self.assertTrue(all(isinstance(t, Decision) and not tuple(dead_leaves(t))
                            for t in trees))
        ds = build_diagrams(catalog, budgets)
        self.assertEqual([d.program.tree for d in ds], trees)
        result = solve_feedback_vqe(catalog, budgets)
        self.assertEqual(result.time.as_fraction(), analytic_optimum(catalog, budgets)[0])
        witness = result.diagram.program
        self.assertIn(witness.tree, trees)
        self.assertEqual(witness.native_program[1][0], "branch_on_visible_m1")


if __name__ == "__main__":
    unittest.main()
