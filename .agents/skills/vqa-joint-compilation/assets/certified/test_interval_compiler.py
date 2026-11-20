"""Symbolic rational checks, including adversarial coverage and witness claims."""
from dataclasses import replace
from fractions import Fraction as F
import unittest

from interval_compiler import Candidate, Domain, Evaluation, Interval, Located, solve, verify


def I(a, b=None):
    return Interval(a, a if b is None else b)


def absolute(a, b):
    return I(0 if a <= 0 <= b else min(abs(a), abs(b)), max(abs(a), abs(b)))


def constrained_box():
    def bound(box, p):
        x, = box
        return Evaluation(I(1+x.lo, 1+x.hi), (I(F(1,3)-x.hi, F(1,3)-x.lo),))
    return Domain("linear", (I(0,1),), 1, bound)


class IntervalCompilerTests(unittest.TestCase):
    def test_time_gap_across_every_layout_and_original_budget(self):
        a = constrained_box()
        other = Domain("slower", (I(0,2), I(-1,3)), 1,
                       lambda box,p: Evaluation(I(2,3), (I(-1,0),)))
        margin = F(1,40)
        cert = solve((other,a), margin, max_rounds=9)
        self.assertEqual(cert.status, "delta_optimal")
        self.assertTrue(verify((a,other), margin, cert))
        self.assertEqual(cert.witness.candidate.domain, a.name)
        x, = cert.witness.candidate.point
        self.assertGreaterEqual(x, F(1,3))
        self.assertLessEqual(cert.lower, F(4,3))
        self.assertGreaterEqual(cert.upper, 1+x)
        self.assertLessEqual(cert.upper-cert.lower, margin)
        self.assertEqual(set(l.domain for l in cert.cover), {a.name, other.name})
        self.assertFalse(verify((a,other), 0, cert))

    def test_missing_half_duplicate_parent_and_forged_bounds(self):
        d, delta = constrained_box(), F(1,32)
        c = solve((d,), delta)
        self.assertTrue(verify((d,), delta, c))
        self.assertGreater(len(c.cover), 1)
        first, *rest = c.cover
        for fake in (replace(c, cover=tuple(rest)), replace(c, cover=c.cover+(first,)),
                     replace(c, cover=(replace(first,path=()),)+tuple(rest)),
                     replace(c, lower=c.lower+1), replace(c, upper=c.upper-1)):
            self.assertFalse(verify((d,), delta, fake))
        fake_values = replace(first.values,time=I(2000))
        self.assertFalse(verify((d,), delta, replace(c,
            cover=(replace(first, values=fake_values), *rest))))
        fake_point = replace(c.witness,candidate=Candidate(d.name, (0,)))
        self.assertFalse(verify((d,), delta, replace(c,witness=fake_point)))

    def test_no_tolerance_on_budgets_and_exact_boundary_seed(self):
        # Feasible iff x=1/3, while every sampled midpoint is dyadic.
        def bound(box,p):
            x, = box
            return Evaluation(I(7), (absolute(x.lo-F(1,3),x.hi-F(1,3)),))
        d = Domain("boundary", (I(0,1),), 1, bound)
        pending = solve((d,), 100, max_rounds=6)
        self.assertEqual(pending.status, "undecided")
        self.assertIsNone(pending.witness)
        self.assertTrue(verify((d,), 100, pending))
        done = solve((d,), 0, seeds=(Candidate(d.name,(F(1,3),)),))
        self.assertEqual((done.status,done.lower,done.upper), ("delta_optimal",7,7))
        self.assertTrue(verify((d,), 0, done))

    def test_named_irrational_boundary_seed_must_prove_location(self):
        # Actual residual is abs(x^2-2), and x>=0. Named witness means +sqrt(2).
        def bound(box,p):
            x, = box
            return Evaluation(I(9), (absolute(x.lo*x.lo-2,x.hi*x.hi-2),))
        def named(key,p):
            if key != "positive-root":
                raise KeyError(key)
            return Located((I(1,2),), Evaluation(I(9),(I(0),)))
        d = Domain("algebraic", (I(1,2),), 1, bound, named)
        seed = Candidate(d.name,"positive-root")
        c = solve((d,), 0, seeds=(seed,))
        self.assertEqual(c.status,"delta_optimal")
        self.assertTrue(verify((d,), 0, c))
        bad = replace(d,named_point=lambda k,p: Located((I(-2,2),), Evaluation(I(9),(I(0),))))
        self.assertFalse(verify((bad,), 0, c))
        false_point = replace(d,named_point=lambda k,p: Located((I(1,2),), Evaluation(I(9),(I(1),))))
        self.assertFalse(verify((false_point,), 0, c))

    def test_empty_space_needs_complete_proof_and_unknown_is_preserved(self):
        # g1=x, g2=1/2-x require simultaneously x<=0 and x>=1/2.
        def bound(box,p):
            x, = box
            return Evaluation(I(0), (x,I(F(1,2)-x.hi,F(1,2)-x.lo)))
        d = Domain("empty", (I(0,1),), 2, bound)
        c0 = solve((d,),0,max_rounds=0)
        self.assertEqual(c0.status,"undecided")
        c = solve((d,),0,max_rounds=3)
        self.assertEqual(c.status,"infeasible")
        self.assertIsNone(c.upper)
        self.assertTrue(verify((d,),0,c))
        unknown = Domain("missing", (), 0, lambda box,p: None)
        pending = solve((d,unknown), 100, max_rounds=4)
        self.assertEqual(pending.status,"undecided")
        self.assertIsNone(pending.lower)
        self.assertTrue(verify((d,unknown),100,pending))
        self.assertFalse(verify((d,unknown),100,replace(pending,status="infeasible")))

    def test_multiple_axes_and_precision_retries_on_fixed_point(self):
        def bound(box,p):
            x,y = box
            gap = (x.hi-x.lo)+(y.hi-y.lo)
            return Evaluation(I(10-gap,10), (I(-1),))
        d = Domain("2d", (I(0,1),I(0,1)), 1,bound)
        c = solve((d,), F(1,4),max_rounds=6)
        self.assertEqual(c.status,"delta_optimal")
        self.assertTrue(verify((d,),F(1,4),c))
        self.assertEqual(len(c.cover), 64)
        def singleton(box,p):
            return Evaluation(I(1-F(1,2**p),1+F(1,2**p)),(I(-1,-1+F(2,2**p)),))
        fixed = Domain("fixed",(),1,singleton)
        c = solve((fixed,),F(1,8),max_rounds=5)
        self.assertEqual(c.status,"delta_optimal")
        self.assertEqual(c.witness.precision,4)
        self.assertTrue(verify((fixed,),F(1,8),c))

    def test_false_seed_and_float_inputs_rejected(self):
        d = constrained_box()
        with self.assertRaises(ValueError):
            solve((d,),1,seeds=(Candidate(d.name, (2,)),))
        with self.assertRaises(TypeError):
            solve((d,),0.01)
        with self.assertRaises(TypeError):
            I(0.1,0.2)
        with self.assertRaises(ValueError):
            solve((d,d),0)
        c = solve((),0)
        self.assertEqual(c.status,"infeasible")
        self.assertTrue(verify((),0,c))

    def test_unbounded_search_preserves_state_past_default_cap(self):
        seen = []
        def enclose(box, p):
            seen.append(p)
            err = F(1, 2**p)
            return Evaluation(I(1-err, 1+err), (I(-1),))

        fixed = Domain("fixed", (), 1, enclose)
        delta = F(1, 4096)
        self.assertEqual(solve((fixed,), delta).status, "undecided")
        seen.clear()
        cert = solve((fixed,), delta, max_rounds=None)
        self.assertEqual(cert.status, "delta_optimal")
        self.assertEqual(cert.witness.precision, 13)
        # One box and one witness query per round; the partition never restarts.
        self.assertEqual(seen, [p for p in range(14) for _ in range(2)])
        self.assertTrue(verify((fixed,), delta, cert))


if __name__ == "__main__":
    unittest.main()
