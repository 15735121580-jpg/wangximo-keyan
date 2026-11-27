import unittest
from fractions import Fraction

from vqa_harness.optimality import INFEASIBLE, optimize_physical_domain


class ThreeCosts:
    root = "root"
    costs = {"known": 10, "exact": 5}

    def lower(self, cell):
        return {"root": 1, "known": 10, "maybe": 4, "exact": 5, "invalid": INFEASIBLE}[cell]

    def proposals(self, cell):
        return [cell] if cell in self.costs else []

    def split(self, cell):
        return ("known", "maybe", "exact", "invalid") if cell == self.root else None


class OptimalityTest(unittest.TestCase):
    def run_model(self, model, **kw):
        return optimize_physical_domain(model, model.costs.get, initial=["known"], **kw)

    def test_unresolved_leaf_survives_an_interruption(self):
        result = self.run_model(ThreeCosts(), max_expansions=0)
        self.assertEqual((result["status"], result["lower"], result["upper"]), ("open", 1, 10))
        result = self.run_model(ThreeCosts())
        self.assertEqual((result["status"], result["policy"], result["gap"]), ("open", "exact", 1))

    def test_tolerance_preserves_the_actual_gap(self):
        result = self.run_model(ThreeCosts(), tolerance=Fraction(3, 2))
        self.assertEqual((result["status"], result["gap"]), ("epsilon_optimal", 1))

    def test_optimal_witness_and_infeasibility(self):
        model = ThreeCosts()
        old_lower = model.lower
        model.lower = lambda cell: INFEASIBLE if cell == "maybe" else old_lower(cell)
        result = self.run_model(model)
        self.assertEqual((result["status"], result["policy"], result["gap"]), ("optimal", "exact", 0))
        model.lower = lambda cell: INFEASIBLE
        with self.assertRaises(ValueError):
            self.run_model(model)
        result = optimize_physical_domain(model, model.costs.get)
        self.assertEqual((result["status"], result["policy"]), ("infeasible", None))

    def test_contradictory_lower_bound_and_uncertified_float(self):
        model = ThreeCosts()
        for bound in (100, 0.3):
            with self.subTest(bound=bound), self.assertRaises(ValueError):
                model.lower = lambda cell: bound
                self.run_model(model)


    def test_joint_compiler_prioritizes_an_attainable_exact_solution(self):
        from types import SimpleNamespace
        from vqa_harness import compile_joint

        model = ThreeCosts()
        previous = model.lower
        model.lower = lambda cell: INFEASIBLE if cell == "maybe" else previous(cell)
        problem = SimpleNamespace(scope="finite native controls", domain=model, verify_upper=model.costs.get,
                                  host="known", tolerance=10, max_expansions=10)
        platform = SimpleNamespace(prepare=lambda task: problem)
        exact = compile_joint("observable", platform)
        self.assertEqual((exact["status"], exact["policy"], exact["gap"]), ("optimal", "exact", 0))
        problem.max_expansions = 0
        near = compile_joint("observable", platform)
        self.assertEqual((near["status"], near["policy"], near["gap"]), ("epsilon_optimal", "known", 9))


    def test_model_proposals_go_through_the_same_verifier(self):
        from types import SimpleNamespace
        from vqa_harness import compile_joint

        model = ThreeCosts()
        model.lower = lambda cell: 5
        problem = SimpleNamespace(scope="native controls", domain=model, verify_upper=model.costs.get,
                                  host="known", tolerance=0, max_expansions=1)
        calls = []
        def propose(task, host):
            calls.append((task, host))
            return ("invalid_hint", "exact")
        platform = SimpleNamespace(prepare=lambda task: problem, propose_programs=propose)
        self.assertEqual(compile_joint("observable", platform)["policy"], "exact")
        self.assertEqual(calls, [("observable", "known")])
        problem.host = "exact"
        self.assertEqual(compile_joint("observable", platform)["policy"], "exact")
        self.assertEqual(len(calls), 1)

    def test_delivery_requires_the_callers_tolerance(self):
        from types import SimpleNamespace
        from vqa_harness import compile_joint

        model = ThreeCosts()
        problem = SimpleNamespace(scope="native controls", domain=model, verify_upper=model.costs.get,
                                  host="known", tolerance=2, max_expansions=None)
        platform = SimpleNamespace(prepare=lambda task: problem)
        open_result = compile_joint("task", platform, tolerance=Fraction(1, 2))
        self.assertEqual((open_result["status"], open_result["policy"], open_result["incumbent"]),
                         ("open", None, "exact"))
        self.assertEqual((open_result["lower"], open_result["upper"], open_result["gap"]),
                         (4, 5, 1))
        near = compile_joint("task", platform, tolerance=1)
        self.assertEqual((near["status"], near["policy"], near["gap"]), ("epsilon_optimal", "exact", 1))
        problem.host = "unknown"
        problem.max_expansions = 0
        result = compile_joint("task", platform, tolerance=100)
        self.assertEqual((result["status"], result["policy"], result["incumbent"], result["upper"]),
                         ("open", None, None, None))
