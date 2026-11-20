"""Invariants for early terminal reduction and matched complete-job evidence."""
import unittest

from vqa_joint.contract import Assessment, Interval, Platform, PULSE_OBJECTIVES, RESOURCE_OBJECTIVES, large_gain
from vqa_joint.host_policy import PolicyRefusal, check_nonregression
from vqa_joint.terminal_controller import best_of_shots


class ReleaseBehaviorTests(unittest.TestCase):
    def test_terminal_never_submits_a_later_block_after_certificate(self):
        requests = []
        batches = iter((["10", "00"], ["11", "01"]))
        def draw(count):
            requests.append(count)
            return next(batches)
        # 01 and 10 achieve equal score; 01 also satisfies the tie rule.
        ranks = {"01": (1, 1), "10": (1, 0), "00": (0, 0)}
        result = best_of_shots(draw, ranks.get, lambda x: x == "01", shot_cap=11, block_size=2)
        self.assertEqual((result.answer, result.shots, result.decision_checks), ("01", 4, 2))
        self.assertEqual(requests, [2, 2])

    def test_partial_final_block_and_missing_draw(self):
        requests = []
        def empty_answers(count):
            requests.append(count)
            return ["infeasible"] * count
        result = best_of_shots(empty_answers, lambda x: None, lambda x: True, shot_cap=5, block_size=2)
        self.assertEqual(requests, [2, 2, 1])
        self.assertEqual((result.answer, result.shots), (None, 5))
        with self.assertRaisesRegex(ValueError, "incomplete block"):
            best_of_shots(lambda count: [], str, lambda x: True, shot_cap=3, block_size=1)

    def test_success_regression_rejected_even_above_required_floor(self):
        platform = Platform.neutral_atom_study("A", calibration="model", code_decoder_round="unencoded",
                                              channel_model="local depolarization")
        def digital(time, success):
            costs = {key: Interval(1, 1) for key in RESOURCE_OBJECTIVES}
            costs["execution_seconds"] = Interval(time, time)
            costs["circuit_process_infidelity"] = Interval(.001, .001)
            return Assessment("same job", platform, Interval(success, success), costs, encoding="unencoded")
        decision = large_gain(digital(4, .95), digital(10, .99), "execution_seconds", .9)
        self.assertTrue(decision["both_pass_task_threshold"])
        self.assertIn("task_success", decision["worse"])
        self.assertFalse(decision["certified_pareto"])
        self.assertFalse(decision["qualifies_large_gain"])
        with self.assertRaisesRegex(PolicyRefusal, "task_failure_probability"):
            check_nonregression({"time": 4, "task_failure_probability": .05},
                                {"time": 10, "task_failure_probability": .01},
                                ("time", "task_failure_probability"))

    def test_analog_simulation_with_unknown_clock_cannot_certify_joint_gain(self):
        platform = Platform("array", "pulse model", "unencoded", (), None, "coherent pulse dynamics", "analog")
        def analog(pulses):
            costs = {key: Interval(1, 1) for key in PULSE_OBJECTIVES}
            costs.update(physical_pulses=Interval(pulses, pulses), execution_seconds=None,
                         circuit_process_infidelity=Interval(.001, .001))
            return Assessment("same pulse job", platform, Interval(.99, .99), costs, encoding="unencoded")
        decision = large_gain(analog(4), analog(10), "physical_pulses", .9)
        self.assertAlmostEqual(decision["gain_lower"], .6)
        self.assertIn("execution_seconds", decision["unresolved"])
        self.assertFalse(decision["certified_pareto"])
        self.assertFalse(decision["qualifies_large_gain"])


if __name__ == "__main__":
    unittest.main()
