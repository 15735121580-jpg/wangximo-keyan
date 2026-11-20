"""Exact symbolic regression suite for the real-field optimizer and verifier."""
from fractions import Fraction
from unittest import TestCase, main

import z3

from optimal_compiler import Diagram, Result, solve, verify


def diagram(name, controls, feasible, cost):
    return Diagram(name, controls, feasible, cost, ("native-proof", name))


class RealSynthesisTests(TestCase):
    def test_continuous_rational_interior_and_competing_diagrams(self):
        x = z3.Real("interior_control")
        adaptive = diagram("rational", (x,), z3.And(x >= 0, x <= 1), 1 + (3*x-1)**2)
        slower = diagram("slow", (), z3.BoolVal(True), z3.RealVal("5/4"))
        catalog = (slower, adaptive)
        result = solve(catalog)
        self.assertEqual(result.status, "optimal")
        self.assertIs(result.diagram, adaptive)
        self.assertEqual(result.time.as_fraction(), Fraction(1))
        self.assertEqual(result.assignment[0][1].as_fraction(), Fraction(1, 3))
        self.assertTrue(verify(catalog, result))
        self.assertFalse(verify(catalog, Result("optimal", z3.RealVal("5/4"), slower)))

    def test_algebraic_minimum_closed(self):
        x = z3.Real("root_control")
        task = diagram("positive-root", (x,), z3.And(x >= 0, x*x >= 2, x <= 3), x)
        result = solve((task,))
        root = z3.simplify(z3.Sqrt(2))
        self.assertEqual(result.status, "optimal")
        self.assertTrue(z3.is_true(z3.simplify(result.time == root)))
        self.assertTrue(z3.is_true(z3.simplify(result.assignment[0][1] == root)))
        self.assertTrue(verify((task,), result))

    def test_open_infima_rational_and_algebraic(self):
        x = z3.Real("open_control")
        for lower in (z3.RealVal(0), z3.simplify(z3.Sqrt(2))):
            with self.subTest(lower=lower):
                task = diagram("open", (x,), z3.And(x > lower, x < 3), x)
                result = solve((task,))
                self.assertEqual(result.status, "infimum")
                self.assertIsNone(result.diagram)
                self.assertTrue(z3.is_true(z3.simplify(result.time == lower)))
                self.assertTrue(verify((task,), result))
                fake = Result("optimal", result.time, task, ((x, result.time),))
                self.assertFalse(verify((task,), fake))
                self.assertFalse(verify((task,), Result("infimum", z3.RealVal(-1))))

    def test_infeasible_catalog_and_empty_language(self):
        x = z3.Real("infeasible_control")
        task = diagram("impossible", (x,), z3.And(x > 0, x < 0), x)
        for catalog in ((), (task,)):
            result = solve(catalog)
            self.assertEqual(result.status, "infeasible")
            self.assertTrue(verify(catalog, result))
            self.assertFalse(verify(catalog, Result("infimum", z3.RealVal(0))))

    def test_quantified_nested_real_feasibility(self):
        x, y = z3.Reals("public_control verifier_input")
        # A single public controller x valid for every hidden y in [-1,1].
        spec = z3.ForAll(y, z3.Implies(z3.And(y >= -1, y <= 1), x >= y*y))
        task = diagram("robust", (x,), z3.And(spec, x <= 2), x)
        result = solve((task,))
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.time.as_fraction(), Fraction(1))
        self.assertTrue(verify((task,), result))

    def test_input_contract(self):
        x, y = z3.Reals("x y")
        permits = z3.BoolVal(True)
        bad = (
            diagram("free", (x,), y == 0, x),
            diagram("loose-index", (), z3.Var(0, z3.RealSort()) > 0, z3.RealVal(1)),
            diagram("integer", (), z3.ForAll([z3.Int("k")], z3.Int("k") > 0), z3.RealVal(1)),
            diagram("repeat", (x, x), permits, x),
            diagram("inverse", (x,), x > 0, 1/x),
            diagram("poison", (x,), x > 0, x/z3.RealVal(0)),
            diagram("unknown-fun", (x,), permits,
                    z3.Function("oracle", z3.RealSort(), z3.RealSort())(x)),
            diagram("fractional-power", (x,), x > 0, x**z3.RealVal("1/2")),
            diagram("negative-duration", (x,), x >= 1, -x),
        )
        for task in bad:
            with self.subTest(name=task.name), self.assertRaises(ValueError):
                solve((task,))
        one = diagram("duplicate-name", (), permits, z3.RealVal(1))
        with self.assertRaises(ValueError):
            solve((one, one))

    def test_certificate_must_carry_exact_closed_values(self):
        x, free = z3.Reals("witness fake_param")
        task = diagram("witness", (x,), x >= 0, x)
        fake = Result("optimal", z3.RealVal(0), task, ((x, free - free),))
        self.assertFalse(verify((task,), fake))
        self.assertFalse(verify((task,), Result("optimal", free, task, ((x, free),))))
        self.assertFalse(verify((task,), Result("optimal")))
        self.assertFalse(verify((task,), Result("infeasible")))


if __name__ == "__main__":
    main()
