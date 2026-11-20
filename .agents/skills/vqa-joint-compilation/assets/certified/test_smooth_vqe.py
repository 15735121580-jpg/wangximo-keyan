"""Exact analytical/algebraic checks; all four adaptive branches are retained."""
from dataclasses import replace
from fractions import Fraction as F
import unittest

import z3

from feedback_vqe import leaf_distribution
from interval_compiler import Interval, verify
from smooth_vqe import (CATALOG, DELTA, LIMITS, TARGET_FLIP, ControlledProgram,
                        _enclose, _fidelity_at_r, _time, analytic_optimum,
                        certified_job, domains, exp_minus)


def exact_z3(x):
    return z3.RealVal(f"{x.numerator}/{x.denominator}")


class SmoothCompilationTests(unittest.TestCase):
    def test_two_quality_thresholds_equal_true_full_delivery_at_analytic_boundary(self):
        ideal = leaf_distribution(0)
        real = leaf_distribution(TARGET_FLIP)
        prog = ControlledProgram("e", F(3, 4))
        self.assertEqual(set(ideal), {(-1, -1), (-1, 1), (1, -1), (1, 1)})
        self.assertEqual(set(real), set(ideal))
        leaves = [prog.replay(*record) for record in ideal]
        # The readouts are themselves delivered, so different leaves stay distinct.
        self.assertEqual({c["readouts"] for c in leaves}, set(ideal))
        self.assertEqual(sum(real.values()), 1)
        complete_affinity = sum(z3.Sqrt(exact_z3(ideal[h] * real[h])) for h in ideal)
        self.assertTrue(z3.is_true(z3.simplify(complete_affinity ** 2 == exact_z3(LIMITS.fidelity))))
        self.assertEqual(1 - (1 - TARGET_FLIP) ** 2, LIMITS.error)
        for r in (F(1, 20), TARGET_FLIP, F(1, 10)):
            real = leaf_distribution(r)
            affinity = sum(z3.Sqrt(exact_z3(ideal[h] * real[h])) for h in ideal)
            enclosure = _fidelity_at_r(r, 32)
            self.assertTrue(z3.is_true(z3.simplify(z3.And(
                exact_z3(enclosure.lo) <= affinity ** 2, affinity ** 2 <= exact_z3(enclosure.hi)))))
        self.assertEqual(_time("s", F(0), TARGET_FLIP) - _time("e", F(0), TARGET_FLIP), F(267, 169))

    def test_interval_witness_replay_and_independent_analytic_solution(self):
        ds = domains()
        cert, prog = certified_job()
        self.assertEqual(cert.status, "delta_optimal")
        self.assertTrue(verify(ds, DELTA, cert))
        self.assertLessEqual(cert.upper - cert.lower, DELTA)
        analytic = analytic_optimum()
        self.assertLessEqual(cert.lower, analytic.lo)
        self.assertLessEqual(analytic.hi, cert.upper)
        self.assertLess(analytic.hi - analytic.lo, F(1, 10**18))
        self.assertEqual(prog.layout, "e")
        self.assertEqual(prog.instructions()[0][1], prog.control)
        exact_point = (Interval(prog.control, prog.control),)
        self.assertTrue(all(c.hi <= 0 for c in _enclose("e", exact_point, 40).constraints))
        self.assertEqual(prog.replay(1, -1)["choice"], "B")
        self.assertEqual(prog.replay(-1, 1)["parameters"], ("A", "A"))
        self.assertFalse(verify(ds, DELTA, replace(cert, lower=cert.upper)))

    def test_all_box_oracles_and_costing_on_all_four_branches(self):
        for layout in ("e", "s"):
            whole = _enclose(layout, (Interval(0, 1),), 24)
            for x in (F(0), F(1, 3), F(1)):
                point = _enclose(layout, (Interval(x, x),), 30)
                self.assertTrue(whole.time.contains(point.time))
                self.assertTrue(all(g.contains(q) for g, q in zip(whole.constraints, point.constraints)))
            # Exact conditional times from the input physical instrument, summed
            # using the four leaf masses (never ideal weights for real time).
            x, r = F(2, 7), TARGET_FLIP
            expected = sum(mass * (
                CATALOG.a + CATALOG.b + 2 * x + (CATALOG.delta if layout == "e" else 0)
                + (CATALOG.kappa if layout == "e" and m1 == -1 else 0)
                + (CATALOG.eta if layout == "s" and m1 == 1 else 0)
            ) for (m1, _), mass in leaf_distribution(r).items())
            self.assertEqual(expected, _time(layout, x, r))
        self.assertEqual(exp_minus(0, 32), Interval(1, 1))
        self.assertEqual(_enclose("s", (Interval(1, 1),), 30).constraints[:2],
                         (Interval(0, 0), Interval(0, 0)))
        self.assertGreater(_enclose("e", (Interval(0, 0),), 30).constraints[2].lo, 0)
        with self.assertRaises(ValueError):
            exp_minus(F(11, 10), 12)

    def test_resource_stop_does_not_turn_a_feasible_witness_into_a_global_claim(self):
        cert, prog = certified_job(max_rounds=0)
        self.assertEqual(cert.status, "undecided")
        self.assertIsNone(prog)
        self.assertTrue(verify(domains(), DELTA, cert))


if __name__ == "__main__":
    unittest.main()
