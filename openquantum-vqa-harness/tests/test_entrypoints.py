"""Public-API release, admission, validation and early-exit mechanics."""
import unittest
from vqa_harness.guarded_release import Assessment, ReleaseContract
from vqa_harness.host_compilation import Branch, HostJob, compile_bounded, compile_direct


class BookkeepingAdapter:
    """A constructed control with an exact integer whole-job loss."""
    model = "integer-test"

    def __init__(self, status="feasible"):
        self.status = status
        self.inputs = []

    def verifier(self, job):
        return self

    def construct(self, job, branch):
        self.inputs.append(job.host)
        return int(branch.name)

    def assess(self, contract, policy):
        return Assessment(self.status, policy)

    def upper_bounds(self, contract, host, proposal):
        return {"job_loss":proposal.evidence-host.evidence}


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.job = HostJob("task","device",10,("readout","schedule"),
                           ReleaseContract("integer-test",("job_loss",)))
        self.branches = (Branch("8",("readout",)),Branch("6",("readout","schedule")),
                         Branch("2",("schedule",)))

    def test_budgeted_selection_from_one_baseline(self):
        adapter = BookkeepingAdapter()
        result = compile_bounded(self.job,self.branches,2,adapter,legacy_host_only=True)
        self.assertEqual((result["policy"],result["selected"],result["evaluated"]), (6,1,2))
        self.assertEqual(adapter.inputs,[self.job.host]*2)
        self.assertEqual(compile_direct(self.job,self.branches[0],adapter)["policy"],8)

    def test_unavailable_host_and_out_of_scope_stage(self):
        adapter = BookkeepingAdapter("missing")
        self.assertEqual(compile_direct(self.job,self.branches[0],adapter)["status"],"unresolved_host")
        self.assertFalse(adapter.inputs)
        with self.assertRaises(ValueError):
            compile_direct(self.job,Branch("2",("undeclared",)),adapter)
        self.assertFalse(adapter.inputs)

    def test_zero_budget_and_malformed_registration(self):
        adapter = BookkeepingAdapter()
        self.assertIs(compile_bounded(self.job,self.branches,0,adapter,legacy_host_only=True)["policy"],self.job.host)
        with self.assertRaises(ValueError):
            compile_bounded(self.job,(self.branches[0],self.branches[0]),2,adapter)
        with self.assertRaises(ValueError):
            compile_bounded(self.job,self.branches,-1,adapter)
        self.assertFalse(adapter.inputs)
