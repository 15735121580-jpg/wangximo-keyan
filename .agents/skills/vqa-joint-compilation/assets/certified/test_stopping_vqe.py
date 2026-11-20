"""Independent exact audit of the VQE fixture; no sampling or device execution."""
from collections import defaultdict
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
from unittest import TestCase, main

import z3

from cyclic_compiler import compile_stopping_job, replay
from stopping_vqe import make_job


def direct_law():
    """Enumerate true outcomes and two independent hidden flips algebraically."""
    law = defaultdict(F)
    for true1, err1, true2, err2 in product((1, -1), repeat=4):
        visible1 = true1 * err1
        prob1 = F(3, 4) if true1 == 1 else F(1, 4)
        # Shot 2 is trained on the *visible*, not latent, first sign.
        prob_pos2 = F(1, 4) if visible1 == 1 else F(3, 4)
        prob2 = prob_pos2 if true2 == 1 else 1 - prob_pos2
        noise = (F(9, 10) if err1 == 1 else F(1, 10)) * (
            F(9, 10) if err2 == 1 else F(1, 10))
        law[visible1, true2*err2, (err1 == -1 or err2 == -1)] += prob1*prob2*noise
    return law


def policy(layout, barrier):
    """Explicit public policy; all initial, loop and last-decision depths are known."""
    public = {("start", 0): {"encoded-A/"+layout: z3.RealVal(1)}}
    for first in (1, -1):
        tag = f"{layout}/{first}"
        public["barrier/"+tag, 5] = {barrier: z3.RealVal(1)}
        public["round-2/"+tag, 5] = {"encoded-"+("B" if first == 1 else "A"): z3.RealVal(1)}
    return public


class StoppingVQEAudit(TestCase):
    def test_every_hidden_branch_against_joint_law(self):
        job = make_job()
        joint = direct_law()
        for root_action in job.actions["start"]:
            layout = root_action.name.split("/")[1]
            for root_outcome in root_action.outcomes:
                barrier_state = root_outcome.next_state
                first = int(barrier_state.rsplit("/", 1)[1])
                marginal = sum(p for (y, _, _), p in joint.items() if y == first)
                self.assertEqual(root_outcome.probability, marginal)
                poll, sync = job.actions[barrier_state]
                self.assertEqual((poll.depth, sync.depth, poll.peak_atoms, sync.peak_atoms), (0,0,0,0))
                self.assertEqual({o.next_state: o.probability for o in poll.outcomes},
                                 {barrier_state: F(1,2), f"round-2/{layout}/{first}": F(1,2)})
                final_action, = job.actions[sync.outcomes[0].next_state]
                self.assertEqual((root_action.depth + final_action.depth,
                                  max(root_action.peak_atoms, final_action.peak_atoms)), (13, 12))
                for leaf in final_action.outcomes:
                    observed1, observed2 = map(int, leaf.output.split(","))
                    self.assertEqual(observed1, first)
                    self.assertEqual(leaf.probability*marginal, joint[first, observed2, leaf.failed])
                conditional_no_fault = sum(o.probability for o in final_action.outcomes if not o.failed)
                posterior1 = conditional_no_fault / F(9,10)
                self.assertEqual(posterior1, F(27,28) if first == 1 else F(3,4))

    def test_synthesis_independent_lower_bound_and_fidelity(self):
        job = make_job()
        solution = compile_stopping_job(job)
        stats = solution.statistics
        self.assertEqual(solution.result.status, "optimal")
        # Every proper barrier policy obeys E[polls]/2 + E[syncs] == 1;
        # its expected duration is 2 + E[syncs]. Remaining control chooses
        # the first orientation *before* learning the measured first sign.
        e = F(4) + F(7,10)*4 + F(3,10)*5
        s = F(3) + F(7,10)*8 + F(3,10)*4
        self.assertEqual(2 + min(e, s), F(103,10))
        self.assertEqual(stats.expected_time.as_fraction(), 2 + min(e,s))
        self.assertEqual(stats.failure.as_fraction(), F(19,100))
        self.assertEqual({y: p.as_fraction() for y,p in stats.output.items()},
                         {"1,1": F(21,100), "1,-1": F(49,100),
                          "-1,1": F(21,100), "-1,-1": F(9,100)})
        self.assertEqual(job.ideal, {"1,1": F(3,16), "1,-1": F(9,16),
                                     "-1,1": F(3,16), "-1,-1": F(1,16)})
        expected_fidelity = (207 + 72*z3.Sqrt(7))/400
        self.assertTrue(z3.is_true(z3.simplify(stats.fidelity == expected_fidelity)))
        self.assertEqual(solution.policy, policy("e","poll"))
        self.assertEqual(replay(job,policy("e","synchronous-result")).expected_time.as_fraction(), e+3)
        self.assertEqual(replay(job,policy("s","poll")).expected_time.as_fraction(), s+2)

    def test_hard_resource_and_quality_thresholds(self):
        job = make_job()
        # Poll adds real time; decreasing a required *quantum* budget by one
        # cannot be repaired by a different public schedule.
        for change in (dict(max_depth=12), dict(max_atoms=11),
                       dict(error_limit=F(18,100)), dict(fidelity_min=F(1))):
            with self.subTest(change=change):
                self.assertEqual(compile_stopping_job(replace(job, **change)).result.status,
                                 "infeasible")


if __name__ == "__main__":
    main()
