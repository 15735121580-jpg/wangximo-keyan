"""The independent deployment boundary is checked without invoking a quantum device."""
from dataclasses import replace
from fractions import Fraction as F
import hashlib
import unittest

from vqa_harness import (DeploymentContract, PhysicalEvidence, admit_deployment,
                         audit_physical_evidence, dispatch_qualified, freeze_deployment, qualify_saved)


class Registry:
    def __init__(self):
        self.records = {}
        self.seals = {}
        self.order = 15

    def subject(self, program):
        return hashlib.sha256(program.encode()).hexdigest()

    def seal(self, contract, subject, standby):
        self.order += 1
        self.seals[self.order] = (contract, subject, standby)
        return self.order

    def verify_seal(self, contract, subject, standby, order):
        return self.seals.get(order) == (contract, subject, standby)

    def read_evidence(self, contract, program, reference):
        return self.records.get(reference)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.c = DeploymentContract("task-v1/device-epoch-7", F(95,100), F(95,100), (("physical_error", F(1,10)),))
        self.r = Registry()
        self.candidate = "frozen-native-words-plus-controller"
        self.standby = "saved-native-words-plus-controller"
        self.r.records["r1"] = PhysicalEvidence(
            self.c.scope, self.r.subject(self.candidate), "new", 30, 31, True,
            "binomial", 59, 59, (("physical_error", F(1,20)),), F(1))
        self.r.records["prior"] = replace(self.r.records["r1"], subject=self.r.subject(self.standby),
                                          batch="prior", start_order=5, end_order=6)

    def test_exact_one_sided_threshold_and_observed_failure(self):
        frozen = freeze_deployment(self.c, self.candidate, self.r)
        self.assertEqual(admit_deployment(frozen, "r1", self.r)["status"], "qualified_candidate")
        for n, k, reason in [(58,58,"answer_probability_unproved"), (1000,999,"observed_answer_failure")]:
            self.r.records["test"] = replace(self.r.records["r1"], trials=n, successes=k)
            self.assertIn(reason, admit_deployment(frozen, "test", self.r)["candidate_reasons"])

    def test_perfect_target_demands_verified_universal_proof(self):
        perfect = replace(self.c, minimum_success=F(1))
        frozen = freeze_deployment(perfect, self.candidate, self.r)
        self.assertEqual(admit_deployment(frozen, "r1", self.r)["status"], "not_authorized")
        self.r.records["proof"] = replace(self.r.records["r1"], answer_method="deterministic", successes=None, trials=None)
        self.assertEqual(admit_deployment(frozen, "proof", self.r)["status"], "qualified_candidate")
        self.r.records["claim"] = replace(self.r.records["r1"], answer_method="deterministic")
        self.assertEqual(admit_deployment(frozen,"claim",self.r)["status"],"not_authorized")

    def test_bounds_missing_stale_or_failed_control_withholds(self):
        frozen = freeze_deployment(self.c,self.candidate,self.r)
        controls = [{"scope":"other"}, {"subject":"different"}, {"physical_upper":()},
                    {"start_order":frozen.order}, {"physical_gates_passed":False},
                    {"physical_upper":(("physical_error",F(1,5)),)}, {"bounds_confidence":F(96,100)},
                    {"answer_method":"technical_blocks"}, {"bounds_confidence":None}]
        for control in controls:
            with self.subTest(**control):
                self.r.records["check"] = replace(self.r.records["r1"],**control)
                self.assertIsNone(admit_deployment(frozen,"check",self.r)["policy"])
        self.assertIsNone(admit_deployment(frozen,"absent",self.r)["policy"])
        self.assertIsNone(admit_deployment(replace(frozen,candidate="swapped"),"r1",self.r)["policy"])
        self.assertIsNone(admit_deployment(replace(frozen,order=999),"r1",self.r)["policy"])
        self.r.verify_seal = lambda *args: "unresolved"
        self.assertIsNone(admit_deployment(frozen,"r1",self.r)["policy"])

    def test_saved_standby_reverified_before_release(self):
        saved = qualify_saved(self.c,self.standby,"prior",self.r)["saved"]
        frozen = freeze_deployment(self.c,self.candidate,self.r,saved)
        self.r.records["fail"] = replace(self.r.records["r1"],physical_gates_passed=False)
        recovered = admit_deployment(frozen,"fail",self.r)
        self.assertEqual((recovered["status"],recovered["policy"]),("qualified_standby",self.standby))
        self.r.records["prior"] = replace(self.r.records["prior"],physical_upper=())
        self.assertIsNone(admit_deployment(frozen,"fail",self.r)["policy"])

    def test_same_batch_or_postfreeze_controls_cannot_rescue(self):
        late = replace(self.r.records["prior"],start_order=30,end_order=31,batch="new")
        self.r.records["late"] = late
        saved = qualify_saved(self.c,self.standby,"late",self.r)["saved"]
        with self.assertRaises(ValueError):
            freeze_deployment(self.c,self.candidate,self.r,saved)
        prior = qualify_saved(self.c,self.standby,"prior",self.r)["saved"]
        frozen = freeze_deployment(self.c,self.candidate,self.r,prior)
        self.r.records["fail"] = replace(self.r.records["r1"],batch="prior",physical_gates_passed=False)
        self.assertIsNone(admit_deployment(frozen,"fail",self.r)["policy"])

    def test_hubbard_zero_of_three_with_unknown_standby_is_withheld(self):
        self.r.records["technical"] = replace(self.r.records["r1"], physical_gates_passed=False,
                                              answer_method="technical_blocks",physical_upper=(),successes=0,trials=3)
        self.assertIsNone(qualify_saved(self.c,self.standby,"absent",self.r)["saved"])
        frozen = freeze_deployment(self.c,self.candidate,self.r)
        decision = admit_deployment(frozen,"technical",self.r)
        self.assertEqual(decision["status"],"not_authorized")
        self.assertIn("physical_gate_failed_or_missing",decision["candidate_reasons"])
        self.assertIsNone(decision["policy"])
        self.assertEqual(audit_physical_evidence(self.c,self.candidate,"technical",self.r)["status"],"rejected_evidence")

    def test_device_submitter_receives_only_a_currently_qualified_policy(self):
        frozen = freeze_deployment(self.c, self.candidate, self.r)
        sent = []
        self.assertEqual(dispatch_qualified(frozen, "unknown", self.r, sent.append)["status"], "not_authorized")
        self.assertEqual(sent, [])
        dispatch_qualified(frozen, "r1", self.r, sent.append)
        self.assertEqual(sent, [self.candidate])
        prior = qualify_saved(self.c, self.standby, "prior", self.r)["saved"]
        with_standby = freeze_deployment(self.c, self.candidate, self.r, prior)
        dispatch_qualified(with_standby, "unknown", self.r, sent.append)
        self.assertEqual(sent, [self.candidate, self.standby])
