"""Independent finite error-path oracle and frozen noisy-contract regressions."""
from fractions import Fraction
from itertools import product
import unittest

from vqa_harness.diagram_search import direct_diagram, search_diagram
from vqa_harness.examples.feedback import FeedbackAdapter, compile_all, feedback_job
from vqa_harness.guarded_release import Assessment


def failure_by_error_paths(graph, probability):
    first, second = (len(stage.implementation) for stage in graph.stages if stage.role == "pulse")
    count = first + second
    failed = Fraction(0)
    for errors in product((0, 1), repeat=count):
        observed = ((first + sum(errors[:first])) % 2, (count + sum(errors)) % 2)
        if observed != (1, 0):
            failed += probability ** sum(errors) * (1 - probability) ** (count - sum(errors))
    return failed


class FeedbackClientTests(unittest.TestCase):
    def test_closed_job_channel_against_all_physical_error_paths(self):
        job = feedback_job()
        programs = [job.host] + [r["policy"] for r in compile_all(job).values()]
        for program in programs:
            with self.subTest(native=[s.implementation for s in program.stages]):
                assessed = FeedbackAdapter(job).assess(job.contract, program)
                self.assertEqual(assessed.evidence["modeled_answer_failure"],
                                 failure_by_error_paths(program, job.hardware.flip_probability))
        self.assertEqual([FeedbackAdapter(job).assess(job.contract, p).evidence["gates"]
                          for p in programs], [6, 4, 2, 2, 2])

    def test_every_entry_retains_original_for_missing_noisy_bound(self):
        job = feedback_job(protect_chip=True)
        for name, result in compile_all(job).items():
            with self.subTest(name=name):
                self.assertIs(result["policy"], job.host)
                self.assertEqual(result["status"], "retained_host")
                self.assertTrue(result["branches"])
                self.assertTrue(all(r["decision"] == "missing_bounds" for r in result["branches"]))

    def test_unassessable_host_skips_diagram_matchers(self):
        job = feedback_job()

        class Unavailable:
            model = job.contract.model

            def assess(self, contract, graph):
                return Assessment("missing")

        def fail_if_called(job, graph):
            raise AssertionError("Matcher called on an unresolved original")

        rules = (("unreachable", fail_if_called),)
        direct = direct_diagram(job, rules, Unavailable(), 2)
        searched = search_diagram(job, rules, Unavailable(), 2, 4)
        for result in (direct, searched):
            self.assertEqual(result["status"], "unresolved_host")
            self.assertIsNone(result["policy"])
            self.assertFalse(result["branches"])
        self.assertEqual(searched["constructed"], 0)
        self.assertFalse(searched["exhausted_to_depth"])
