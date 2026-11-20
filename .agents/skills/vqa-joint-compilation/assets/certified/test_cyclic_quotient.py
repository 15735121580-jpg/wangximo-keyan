from dataclasses import replace
from fractions import Fraction as F
import unittest

import z3

from cyclic_compiler import Action, Job, Outcome, compile_stopping_job, replay
from cyclic_quotient import bisimulation, compile_quotient, verify_square


def go(probability, target):
    return Outcome(probability, next_state=target)


def finish(probability, label="ok", failed=False):
    return Outcome(probability, output=label, failed=failed)


def instrument_job():
    # Different physical kernels at the two ports. After identifying A and B,
    # both instruments give the same repeat-or-deliver channel.
    prepare = Action("prepare", 1, 1, 1, (go(F(1, 2), "A"), go(F(1, 2), "B")))
    a = Action("measure", 2, 2, 1, (go(F(1, 4), "A"), go(F(1, 4), "B"), finish(F(1, 2))))
    b = Action("measure", 2, 2, 1, (go(F(1, 2), "B"), finish(F(1, 2))))
    return Job("root", {"root": (prepare,), "A": (a,), "B": (b,)}, {"ok": 1}, 2, None, 0, 1)


class QuotientTests(unittest.TestCase):
    def test_genuine_merge_with_different_implementations_and_geometric_retry(self):
        job = instrument_job()
        sol = compile_quotient(job)
        cert = sol.certificate
        self.assertTrue(verify_square(cert))
        self.assertEqual(len(cert.quotient.actions), 2)
        self.assertEqual(cert.projection["A"], cert.projection["B"])
        self.assertIsNot(cert.local_actions["A", "measure"], cert.local_actions["B", "measure"])
        self.assertIs(cert.local_actions["B", "measure"], job.actions["B"][0])
        self.assertEqual(sol.original_statistics.expected_time.as_fraction(), 5)
        self.assertEqual(sol.quotient_solution.result.time.as_fraction(),
                         compile_stopping_job(job).result.time.as_fraction())
        self.assertEqual(sol.original_statistics.visits["A", None].as_fraction(), F(2, 3))
        self.assertEqual(sol.original_statistics.visits["B", None].as_fraction(), F(4, 3))
        self.assertEqual(sol.original_statistics.fidelity.as_fraction(), 1)
        colored = bisimulation(job, {"A": "species1", "B": "species2", "root": "input"})
        self.assertNotEqual(colored.projection["A"], colored.projection["B"])
        with self.assertRaises(ValueError):
            bisimulation(job, {"A": "species1"})

    def test_refinement_propagates_and_preserves_noise_cost_resource_and_alphabet(self):
        correlated = Action("read", 3, 2, 1, (finish(F(1, 2), "x", True), finish(F(1, 2), "y")))
        anticorrelated = replace(correlated, outcomes=(finish(F(1, 2), "x"), finish(F(1, 2), "y", True)))
        ops = {
            "same": (correlated,),
            "replica": (replace(correlated, outcomes=tuple(reversed(correlated.outcomes))),),
            "noise": (anticorrelated,),
            "slow": (replace(correlated, duration=4),),
            "deep": (replace(correlated, depth=2),),
            "wide": (replace(correlated, peak_atoms=3),),
            "other": (replace(correlated, name="read2"),),
            "extra": (correlated, replace(correlated, name="read2")),
        }
        # Disagreement propagates backwards through two equal-looking layers.
        for name in ("same", "replica", "noise"):
            ops["pre_" + name] = (Action("step", 1, 1, 1, (go(1, name),)),)
            ops["prepre_" + name] = (Action("step", 1, 1, 1, (go(1, "pre_" + name),)),)
        job = Job("prepre_same", ops, {"x": F(1, 2), "y": F(1, 2)}, 5, None, 1, 0)
        cert = bisimulation(job)
        p = cert.projection
        distinct = ("same", "noise", "slow", "deep", "wide", "other", "extra")
        self.assertEqual(len({p[s] for s in distinct}), len(distinct))
        for prefix in ("", "pre_", "prepre_"):
            self.assertEqual(p[prefix + "same"], p[prefix + "replica"])
            self.assertNotEqual(p[prefix + "same"], p[prefix + "noise"])
        coarse_ops = dict(cert.quotient.actions)
        coarse_ops[p["same"]] = (anticorrelated,)
        broken = replace(cert, quotient=replace(cert.quotient, actions=coarse_ops))
        self.assertFalse(verify_square(broken))

    def test_depth_product_and_fault_are_preserved(self):
        j = instrument_job()
        abort = Action("report exhaustion", 1, 0, 0, (finish(1, "exhausted", True),))
        ops = {**j.actions, **{s: j.actions[s] + (abort,) for s in ("A", "B")}}
        job = replace(j, actions=ops, max_depth=2, ideal={"ok": 1, "exhausted": 0},
                      error_limit=F(1, 2), fidelity_min=F(1, 2))
        sol = compile_quotient(job)
        self.assertEqual(sol.quotient_solution.result.time.as_fraction(), F(7, 2))
        self.assertEqual(sol.original_statistics.failure.as_fraction(), F(1, 2))
        for s in ("A", "B"):
            self.assertEqual(set(sol.policy[s, 1]), {"measure"})
            self.assertEqual(set(sol.policy[s, 2]), {"report exhaustion"})
        for reduced in (replace(job, max_atoms=1), replace(job, error_limit=F(49, 100))):
            self.assertEqual(compile_quotient(reduced).quotient_solution.result.status, "infeasible")

    def test_fine_random_observation_can_be_recovered_by_private_randomization(self):
        start = Action("expose", 1, 1, 0, (go(F(1, 2), "A"), go(F(1, 2), "B")))
        low = Action("zero", 1, 1, 1, (finish(1, "0"),))
        high = Action("one", 5, 1, 1, (finish(1, "1"),))
        j = Job("r", {"r": (start,), "A": (low, high), "B": (high, low)},
                {"0": F(1, 2), "1": F(1, 2)}, 1, 1, 0, 1)
        fine = {("r", 0): {"expose": z3.RealVal(1)}, ("A", 0): {"zero": z3.RealVal(1)},
                ("B", 0): {"one": z3.RealVal(1)}}
        self.assertEqual(replay(j, fine).expected_time.as_fraction(), 4)
        sol = compile_quotient(j)
        block = sol.certificate.projection["A"]
        self.assertEqual(block, sol.certificate.projection["B"])
        self.assertEqual(sol.quotient_solution.policy[block, 0]["one"].as_fraction(), F(1, 2))
        self.assertEqual(sol.original_statistics.expected_time.as_fraction(), 4)
        self.assertEqual(sol.original_statistics.fidelity.as_fraction(), 1)


if __name__ == "__main__":
    unittest.main()
