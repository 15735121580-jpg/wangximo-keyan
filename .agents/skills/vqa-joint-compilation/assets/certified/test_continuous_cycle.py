"""Exact symbolic checks for continuous-control absorbing jobs."""
from dataclasses import replace
from fractions import Fraction as F
import unittest

import z3

from continuous_cycle import (
    Backend, Choice, Transfer, _formulate, compile_cycle, replay, verify_solution,
)
from optimal_compiler import Diagram, solve


def exact(test, predicate):
    test.assertTrue(z3.is_true(z3.simplify(predicate)), str(predicate))


def choose(p, *control):
    number = lambda t: z3.RealVal(str(t))
    return Choice(number(p), tuple(map(number, control)))


class ContinuousCycleTests(unittest.TestCase):
    def test_algebraic_continuous_optimum_with_unbounded_retry(self):
        def step(s, u):
            a, = u
            return Transfer(z3.And(a >= 0, a <= 1), (1-a,), (), F(1,8)+a*a)

        backend = Backend(0, (1,), 0, step, lambda moments: True)
        sol = compile_cycle(backend)
        self.assertEqual(sol.result.status, "optimal")
        (atom,) = sol.policy[0]
        a, = atom.control
        # T(a)=1/(8a)+a for a>0. Equality requires 8a^2=1.
        exact(self, z3.And(8*a*a == 1, a > 0, a < 1, atom.probability == 1))
        exact(self, sol.stats.visits[0] * a == 1)
        exact(self, z3.And(2*sol.result.time**2 == 1, sol.result.time > 0))
        self.assertTrue(verify_solution(backend, sol))
        worse = {0: (choose(1, 1),)}
        self.assertFalse(verify_solution(backend, replace(sol, policy=worse)))
        self.assertFalse(verify_solution(backend, replace(sol, policy=worse,
            stats=replay(backend, worse), result=replace(sol.result, time=z3.RealVal("9/8")))))

    def test_infimum_not_attained_even_with_closed_action_interval(self):
        def step(s, u):
            a, = u
            return Transfer(z3.And(a >= 0, a <= 1), (1-a,), (), a*(1+a))

        backend = Backend(0, (1,), 0, step, lambda moments: True)
        sol = compile_cycle(backend)
        self.assertEqual(sol.result.status, "infimum")
        exact(self, sol.result.time == 1)
        self.assertIsNone(sol.policy)
        self.assertTrue(verify_solution(backend, sol))
        with self.assertRaises(ValueError):
            replay(backend, {0: (choose(1, 0),)})

    def test_general_support_requires_randomized_endpoint_choices(self):
        def step(s, u):
            a, = u
            return Transfer(z3.Or(a == 0, a == 1), (F(1,2),), (a/2,), 1)

        backend = Backend(0, (1,), 1, step, lambda z: z[0] == F(1,2))
        sol = compile_cycle(backend)
        self.assertEqual(sol.result.status, "optimal")
        self.assertTrue(verify_solution(backend, sol))
        exact(self, sol.stats.time == 2)
        exact(self, sol.stats.visits[0] == 2)
        for bit in (0, 1):
            weight = sum((atom.probability for atom in sol.policy[0]
                          if z3.is_true(z3.simplify(atom.control[0] == bit))), z3.RealVal(0))
            exact(self, weight == F(1,2))
        with self.assertRaises(ValueError):
            replay(backend, {0: (choose(1, 1),)})

    def test_unreachable_closed_circulation_is_excluded(self):
        def step(s, u):
            return Transfer(True, (0, s), (), int(s == 0))

        backend = Backend(0, (0, 0), 0, step, lambda z: True)
        sol = compile_cycle(backend)
        self.assertEqual(set(sol.policy), {0})
        exact(self, sol.stats.time == 1)
        d = _formulate(backend)
        (unreachable_mass, _), = d.program[1]
        forced = Diagram("forced-circulation", d.variables,
                         z3.And(d.feasible, unreachable_mass > 0), d.time, d.program)
        self.assertEqual(solve((forced,)).status, "infeasible")

    def test_improper_start_or_invalid_transition(self):
        for p in (1, -1, 2):
            backend = Backend(0, (0,), 0, lambda s, u: Transfer(True, (p,), (), 0),
                              lambda z: True)
            sol = compile_cycle(backend)
            self.assertEqual(sol.result.status, "infeasible")
            self.assertTrue(verify_solution(backend, sol))

    def test_resource_monitor_charges_each_quantum_retry(self):
        # State d is the number of quantum attempts already used (d<=1).
        # A failed first attempt enters d=1. Its only possible guard is false
        # for the same quantum retry, so no complete policy can promise success.
        def step(d, u):
            return Transfer(d < 1, (0, F(1,2)), (), 1)

        backend = Backend(0, (0, 0), 0, step, lambda z: True)
        self.assertEqual(compile_cycle(backend).result.status, "infeasible")


if __name__ == "__main__":
    unittest.main()
