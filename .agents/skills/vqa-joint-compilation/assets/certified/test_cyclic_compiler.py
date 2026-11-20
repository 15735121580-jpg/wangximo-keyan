from dataclasses import replace
from fractions import Fraction as F
import unittest

import z3

from cyclic_compiler import Action, Job, Outcome, compile_stopping_job, replay


def out(p, label, fail=False):
    return Outcome(p, output=label, failed=fail)


class UnboundedVariationalControlTests(unittest.TestCase):
    def test_geometric_reprepare_exact_mean_and_nonzero_tail(self):
        # Complete shot can finish or signal retry; no finite round horizon.
        shot = Action("prepare+measure", 2, 1, 1, (Outcome(F(2,3), next_state="try"),
                                                   out(F(1,3), "estimate")))
        j = Job("try", {"try": (shot,)}, {"estimate": 1}, 1, None, 0, 1)
        s = compile_stopping_job(j)
        self.assertEqual(s.result.time.as_fraction(), 6)
        self.assertEqual(s.statistics.visits["try",None].as_fraction(), 3)
        self.assertEqual(s.statistics.failure.as_fraction(), 0)
        # At any finite positive-depth cap a positive-probability path would
        # still need more shots. It cannot be removed or silently delivered.
        self.assertEqual(compile_stopping_job(replace(j, max_depth=3)).result.status, "infeasible")
        exhausted = Action("report exhaustion", 1, 0, 0, (out(1, "exhausted", True),))
        jj = replace(j, actions={"try": (shot, exhausted)}, max_depth=2,
                     ideal={"estimate":1, "exhausted":0},
                     error_limit=F(4,9), fidelity_min=F(5,9))
        capped = compile_stopping_job(jj)
        self.assertEqual(capped.statistics.failure.as_fraction(), F(4,9))
        self.assertEqual(capped.result.time.as_fraction(), F(34,9))

    def test_closed_nonterminating_component_not_deliverable(self):
        loop = Action("loop", 1, 0, 0, (Outcome(1, next_state="loop"),))
        jj = Job("loop", {"loop":(loop,)}, {"y":1}, 1, None, 0, 1)
        self.assertEqual(compile_stopping_job(jj).result.status, "infeasible")
        with self.assertRaisesRegex(ValueError,"closed recurrent"):
            replay(jj, {("loop",None): {"loop":z3.RealVal(1)}})
        useful = Action("finish", 4, 1, 0, (out(1, "y"),))
        enter = Action("enter", 1, 0, 0, (Outcome(1, next_state="loop"),))
        both = replace(jj, start="start", actions={"start": (useful, enter), "loop":(loop,)})
        s = compile_stopping_job(both)
        self.assertEqual(s.result.time.as_fraction(), 4)
        self.assertNotIn(("loop",None), s.policy)
        self.assertEqual(compile_stopping_job(replace(both,max_atoms=0)).result.status,"infeasible")

    def test_randomization_strictly_improves_feasible_minimum(self):
        fast = Action("fast", 1, 1, 1, (out(1,"bad",True),))
        slow = Action("protected", 5, 1, 1, (out(1,"ok"),))
        jj = Job("s", {"s":(fast,slow)}, {"ok":1,"bad":0}, 1, 1, F(1,4), F(3,4))
        s = compile_stopping_job(jj)
        self.assertEqual(s.result.time.as_fraction(), 4)
        self.assertEqual(s.policy["s",0]["fast"].as_fraction(),F(1,4))
        self.assertEqual(s.statistics.output["ok"].as_fraction(),F(3,4))
        self.assertEqual(s.statistics.fidelity.as_fraction(),F(3,4))

    def test_randomization_needed_for_any_feasible_policy(self):
        zero = Action("zero", 1, 1, 1, (out(1,"0"),))
        one = Action("one", 5, 1, 1, (out(1,"1"),))
        jj = Job("s", {"s":(zero,one)}, {"0":F(1,2), "1":F(1,2)}, 1, 1, 0, 1)
        exact = compile_stopping_job(jj)
        self.assertEqual(exact.result.time.as_fraction(),3)
        for op in (zero,one):
            self.assertEqual(compile_stopping_job(replace(jj,actions={"s":(op,)})).result.status,
                             "infeasible")
        # With F=3/4 one needs at least x=(2-sqrt(3))/4 on the slow arm.
        # Design and replay retain the irrational optimal probability exactly.
        q = compile_stopping_job(replace(jj,fidelity_min=F(3,4)))
        self.assertTrue(z3.is_true(z3.simplify(q.result.time == 3-z3.Sqrt(3))))
        self.assertTrue(z3.is_true(z3.simplify(q.statistics.fidelity == z3.RealVal("3/4"))))

    def test_general_output_law_fidelity_not_probability_of_single_event(self):
        op = Action("draw", 1, 1, 1, (out(F(1,2),"0"),out(F(1,2),"1")))
        jj = Job("s", {"s":(op,)}, {"0":F(9,10), "1":F(1,10)},
                 1, 1, 0, F(4,5))
        # (sqrt(.45)+sqrt(.05))^2=.8 exactly.
        s = compile_stopping_job(jj)
        self.assertEqual(s.statistics.fidelity.as_fraction(),F(4,5))
        self.assertEqual(compile_stopping_job(replace(jj, fidelity_min=F(81,100))).result.status,
                         "infeasible")
        three = Action("draw3",1,1,1,(out(F(1,4),"0"),out(F(1,4),"1"),out(F(1,2),"2")))
        jj3 = replace(jj, actions={"s":(three,)}, ideal={"0":F(1,2),"1":F(1,2),"2":0},
                      fidelity_min=F(1,2))
        self.assertEqual(compile_stopping_job(jj3).statistics.fidelity.as_fraction(),F(1,2))


if __name__ == "__main__":
    unittest.main()
