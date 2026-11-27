"""Default A2 must protect B, the delivered direct policy, and the registered reward."""
from dataclasses import dataclass, replace
import unittest

from vqa_harness import (
    Assessment, Branch, HostJob, ReleaseContract, compile_bounded, compile_direct,
    direct_diagram, search_diagram,
)
from vqa_harness.examples.feedback import FeedbackAdapter, feedback_job, single_x


@dataclass(frozen=True)
class Cost:
    time: int
    risk: int
    reward: int


class VerifiedCosts:
    model = "complete-model-job"

    def __init__(self):
        self.policies = {
            "one": Cost(80, 8, 50),
            "risk_regresses": Cost(40, 9, 60),
            "reward_regresses": Cost(50, 7, 35),
            "safe": Cost(60, 7, 60),
            "equal_safe": Cost(60, 7, 60),
            "equal_direct": Cost(80, 8, 50),
        }
        self.constructed = []

    def verifier(self, job):
        return self

    def direct_branch(self, job):
        return Branch("one", ("controller",))

    def construct(self, job, branch):
        self.constructed.append(branch.name)
        return self.policies[branch.name]

    def assess(self, contract, program):
        return Assessment("feasible", program)

    def upper_bounds(self, contract, base, candidate):
        a, b = base.evidence, candidate.evidence
        return {"time": b.time-a.time, "risk": b.risk-a.risk, "-J": a.reward-b.reward}

    upper_bounds_against = upper_bounds


class ProtectedIncumbentTests(unittest.TestCase):
    def setUp(self):
        self.adapter = VerifiedCosts()
        self.job = HostJob("answer-v1", "device-v1", Cost(100,10,20), ("controller",),
                           ReleaseContract(self.adapter.model, ("time","risk"), reward="J"),
                           execution_budget=("shots", 256))
        self.direct = compile_direct(self.job, self.adapter.direct_branch(self.job), self.adapter)
        self.one = self.direct["policy"]

    def choose(self, *names, **kw):
        return compile_bounded(self.job, tuple(Branch(n,("controller",)) for n in names),
                               len(names), self.adapter, **kw)

    def test_tradeoffs_pass_old_release_and_fail_default_dual_guard(self):
        names = ("risk_regresses","reward_regresses","equal_direct")
        old = self.choose(*names,legacy_host_only=True)
        self.assertIs(old["policy"],self.adapter.policies["risk_regresses"])
        new = self.choose(*names,incumbent=self.direct)
        self.assertIs(new["policy"],self.one)
        self.assertEqual(new["status"],"retained_incumbent")
        self.assertEqual([r["decision"] for r in new["branches"]],
                         ["non_regression_unproved","non_regression_unproved","no_certified_gain"])
        self.assertEqual(new["branches"][0]["upper_vs_incumbent"]["risk"],1)
        self.assertEqual(new["branches"][1]["upper_vs_incumbent"]["-J"],15)

    def test_rank_certified_survivors_by_original_objective_then_registered_order(self):
        r=self.choose("risk_regresses","safe","equal_safe",incumbent=self.direct)
        self.assertEqual((r["selected"],r["status"]),(1,"released"))
        self.assertIs(r["policy"],self.adapter.policies["safe"])
        self.assertEqual(r["branches"][1]["upper_vs_host"],{"time":-40,"risk":-3,"-J":-40})
        self.assertEqual(r["branches"][1]["upper_vs_incumbent"],{"time":-20,"risk":-1,"-J":-10})
        contract=ReleaseContract("complete-model-job",("-J","time","risk"),reward="J")
        self.assertEqual(contract.losses,("-J","time","risk"))

    def test_prescribed_default_and_zero_new_search_budget(self):
        self.adapter.constructed.clear()
        automatic=self.choose("risk_regresses")
        self.assertIs(automatic["policy"],self.one)
        self.assertEqual(self.adapter.constructed,["one","risk_regresses"])
        self.adapter.constructed.clear()
        cached=self.choose(incumbent=self.direct)
        self.assertIs(cached["policy"],self.one)
        self.assertEqual(self.adapter.constructed,[])

    def test_unknown_pairwise_bound_cannot_improve_a_known_incumbent(self):
        self.adapter.upper_bounds_against=None
        r=self.choose("safe",incumbent=self.direct)
        self.assertEqual(r["branches"][0]["decision"],"missing_bounds")
        self.assertIs(r["policy"],self.one)
        self.adapter.upper_bounds_against=lambda *args:{"time":-20,"risk":-1}
        r=self.choose("safe",incumbent=self.direct)
        self.assertIsNone(r["branches"][0]["upper_vs_incumbent"]["-J"])
        self.assertIs(r["policy"],self.one)

    def test_weak_host_bound_still_blocks_an_apparently_good_incumbent_difference(self):
        original=self.adapter.upper_bounds
        def incomplete(contract, base, proposed):
            values=original(contract,base,proposed)
            if proposed.evidence is self.adapter.policies["safe"]:
                values["risk"]=None
            return values
        self.adapter.upper_bounds=incomplete
        r=self.choose("safe",incumbent=self.direct)
        self.assertIs(r["policy"],self.one)
        self.assertEqual(r["branches"][0]["decision"],"missing_bounds")

    def test_wrong_job_and_unregistered_direct_are_unresolved(self):
        for changes in ({"execution_budget":("shots",64)}, {"hardware":"other"}, {"task":"other"}):
            with self.subTest(changes=changes):
                other=replace(self.job,**changes)
                r=compile_bounded(other,(),0,self.adapter,incumbent=self.direct)
                self.assertIsNone(r["policy"])
                self.assertEqual(r["status"],"unresolved_incumbent")
        self.adapter.direct_branch=None
        self.assertIsNone(self.choose("safe")["policy"])
        with self.assertRaises(ValueError):
            self.choose("safe",incumbent=self.direct,legacy_host_only=True)

    def test_direct_fallback_to_b_uses_original_guard_for_incumbent_too(self):
        self.adapter.policies["one"]=Cost(80,11,50)
        base=compile_direct(self.job,self.adapter.direct_branch(self.job),self.adapter)
        self.assertIs(base["policy"],self.job.host)
        self.adapter.upper_bounds_against=None
        r=self.choose("safe",incumbent=base)
        self.assertIs(r["policy"],self.adapter.policies["safe"])
        self.assertEqual(r["branches"][0]["upper_vs_host"],r["branches"][0]["upper_vs_incumbent"])

    def test_diagram_keeps_an_already_delivered_derivation_outside_bounded_frontier(self):
        job=feedback_job(); verifier=FeedbackAdapter(job);relations=(("remove two X",single_x),)
        direct=direct_diagram(job,relations,verifier,2)
        self.assertEqual(verifier.assess(job.contract,direct["policy"]).evidence["gates"],2)
        for r in (search_diagram(job,relations,verifier,2,0,incumbent=direct),
                  search_diagram(job,relations,verifier,2,0)):
            self.assertEqual(r["status"],"retained_incumbent")
            self.assertEqual(r["selected_steps"],direct["proposed_steps"])
            self.assertEqual(r["constructed"],0)
        old=search_diagram(job,relations,verifier,2,0,legacy_host_only=True)
        self.assertIs(old["policy"],job.host)

    def test_mutation_of_scope_after_a1_release_does_not_relabel_the_old_policy(self):
        mutable = replace(self.job, execution_budget={"shots": 256})
        prior=compile_direct(mutable,self.adapter.direct_branch(mutable),self.adapter)
        mutable.execution_budget["shots"]=128
        r=compile_bounded(mutable,(),0,self.adapter,incumbent=prior)
        self.assertEqual(r["status"],"unresolved_incumbent")
        restored=replace(self.job)
        self.assertIs(compile_bounded(restored,(),0,self.adapter,incumbent=self.direct)["policy"],self.one)

    def test_opaque_sdk_state_requires_an_explicit_trusted_scope_digest(self):
        job=replace(self.job,hardware=object())
        direct=compile_direct(job,self.adapter.direct_branch(job),self.adapter)
        self.assertIsNone(direct["incumbent"])
        self.adapter.incumbent_key=lambda j,p:"calibrated-job-budget-policy-digest"
        sealed=compile_direct(job,self.adapter.direct_branch(job),self.adapter)
        self.assertIs(compile_bounded(job,(),0,self.adapter,incumbent=sealed)["policy"],self.one)
        self.adapter.incumbent_key=lambda j,p:"different-calibration-or-controller"
        self.assertIsNone(compile_bounded(job,(),0,self.adapter,incumbent=sealed)["policy"])
