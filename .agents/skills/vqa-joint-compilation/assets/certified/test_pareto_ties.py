from dataclasses import replace
from fractions import Fraction as F
import unittest

import z3

from optimal_compiler import Diagram
from pareto_ties import DEFAULT, Endpoint, compile_lex, verify_lex


def point(name, time, f, depth):
    d = Diagram(name, (), z3.BoolVal(True), z3.RealVal(time), name)
    return Endpoint(d, f, F(1,16), depth, 10, 5)


class ParetoTiesTests(unittest.TestCase):
    def test_remove_dominated_time_tie_and_keep_time_priority(self):
        slow = point("slow-pure", 11, 0, 1)
        good = point("good", 10, F(1,8), 3)
        dominated = point("dominated", 10, F(1,8), 4)
        ds = (dominated,slow,good)
        ans = compile_lex(ds)
        self.assertTrue(verify_lex(ds, ans))
        self.assertIs(ans.diagram, good.diagram)
        fake = replace(ans, diagram=dominated.diagram, values=ans.values[:3]+(z3.RealVal(4),)+ans.values[4:])
        self.assertFalse(verify_lex(ds,fake))

    def test_continuous_tradeoff_no_combining_incompatible_assignments(self):
        x = z3.Real("tradeoff_x")
        d = Diagram("same-job", (x,), z3.And(x >= 0, x <= 1), z3.RealVal(10), "program with x")
        e = Endpoint(d,x,0,2,10-x,3)
        for order, expected_x in ((DEFAULT,0), (("time","gates","peak_atoms","depth",
                                                  "infidelity","logical_error"),1)):
            ans = compile_lex([e],order)
            self.assertTrue(verify_lex([e],ans))
            self.assertTrue(z3.is_true(z3.simplify(dict(ans.assignment)[x] == expected_x)))
            mixed = dict(zip(order,ans.values))
            mixed.update(infidelity=z3.RealVal(0),gates=z3.RealVal(9))
            self.assertFalse(verify_lex([e],replace(ans,values=tuple(mixed[k] for k in order))))
            self.assertFalse(verify_lex([e],replace(ans,assignment=((x,z3.RealVal(1-expected_x)),))))

    def test_secondary_nonattainment_and_empty(self):
        x = z3.Real("open_cost")
        d = Diagram("open", (x,), z3.And(x > 0, x <= 1), z3.RealVal(2), "job")
        e = Endpoint(d,x,0,1,1,1)
        ans = compile_lex([e])
        self.assertEqual(ans.status,"infimum")
        self.assertEqual(len(ans.values),1)
        self.assertIsNone(ans.diagram)
        self.assertTrue(verify_lex([e],ans))
        self.assertFalse(verify_lex([e],replace(ans,infimum=z3.RealVal(1))))
        self.assertTrue(verify_lex([], compile_lex([])))

    def test_invalid_cost_and_order(self):
        e = point("bad_quality",1,2,0)
        with self.assertRaises(ValueError):
            compile_lex([e])
        with self.assertRaises(ValueError):
            compile_lex([], tuple(reversed(DEFAULT)))


if __name__ == "__main__":
    unittest.main()
