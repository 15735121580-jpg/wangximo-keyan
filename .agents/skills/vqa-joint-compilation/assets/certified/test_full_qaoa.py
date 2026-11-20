from dataclasses import replace
from fractions import Fraction as F
import unittest

import z3

from full_qaoa import (Backend, BUDGET, RESIDUAL, STRINGS, analytic_result,
                       compile_job, deliver, law, score, update)
from policy_ir import Decision, Leaf, PublicPort, compile_diagrams


class CompleteQAOATests(unittest.TestCase):
    def test_qaoa_cost_and_mixer_give_actual_four_string_laws(self):
        # In the even/odd Bell basis the amplitude of odd parity equals
        # (cos(2 beta)*cos(gamma) - i[cos(2 beta)*sin(gamma)+sin(2 beta)])/sqrt(2).
        c = z3.Sqrt(2)/2  # cos(2 beta)=sin(2 beta)
        for label, sine in (("A", z3.RealVal("-1/2")), ("B", z3.RealVal("1/2"))):
            real, imag = c*z3.Sqrt(3)/2, -c*(sine+1)
            cut = z3.simplify((real**2 + imag**2)/2)
            self.assertEqual(cut.as_fraction(), F(1, 4) if label == "A" else F(3, 4))
            for w, p in law(label, F(0)).items():
                expected = cut.as_fraction() if score(w) else 1-cut.as_fraction()
                self.assertEqual(p, expected/2)
        self.assertEqual(RESIDUAL, F(7, 250))
        self.assertEqual(law("A", RESIDUAL)["01"], F(33, 250))

    def test_root_solver_compares_two_full_policies_and_keeps_16_branches(self):
        diagrams = compile_diagrams(Backend(), PublicPort("start"), 2)
        times = {d.program.native_program.first_layout: d.time.as_fraction() for d in diagrams}
        self.assertEqual(times, {"e": F(1033, 125), "s": F(1243, 125)})
        solution = compile_job()
        self.assertEqual(solution.status, "optimal")
        prog = solution.diagram.program.native_program
        self.assertEqual((solution.time.as_fraction(), prog.first_layout), analytic_result())
        self.assertEqual(prog.first_layout, "e")
        path_count = 0
        for first, child in solution.diagram.program.tree.branches:
            self.assertIsInstance(child, Decision)
            self.assertEqual(child.action.name, f"{update(first.label)}-e")
            for second, leaf in child.branches:
                self.assertIsInstance(leaf, Leaf)
                self.assertEqual(leaf.kind, "delivered")
                out = prog.replay(first.label, second.label)
                self.assertEqual(out["delivered"], deliver(first.label, second.label))
                path_count += 1
        self.assertEqual(path_count, 16)
        self.assertEqual(prog.replay("10", "01")["delivered"], ("10", 1))
        self.assertEqual(prog.replay("00", "01")["angles"], ("-pi/6", "+pi/6"))

    def test_best_sample_fidelity_and_whole_job_failure_are_distinct_budgets(self):
        for r in (F(0), RESIDUAL):
            output = {w: F(0) for w in STRINGS}
            for first, p in law("A", r).items():
                for second, q in law(update(first), r).items():
                    emitted, _ = deliver(first, second)
                    output[emitted] += p*q
            self.assertEqual(sum(output.values()), 1)
            f = F(3, 4) - r/2
            self.assertEqual(output["00"], f*(1-f)/2)
            self.assertEqual(output["11"], output["00"])
            self.assertEqual(output["10"], (1-f*(1-f))/2)
            self.assertEqual(output["01"], output["10"])
        for budget in (replace(BUDGET, error=F(1, 20)), replace(BUDGET, fidelity=1),
                       replace(BUDGET, atoms=7), replace(BUDGET, depth=14)):
            self.assertIsNone(analytic_result(budget))
            self.assertEqual(compile_job(budget).status, "infeasible")


if __name__ == "__main__":
    unittest.main()
