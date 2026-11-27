import json
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from vqa_harness.guarded_release import Assessment, ReleaseContract, guarded_release
from vqa_harness.phase_release import phase_release


class Verifier:
    model = "test"

    def assess(self, contract, policy):
        return Assessment(policy.get("feasibility", "feasible"), policy)

    def upper_bounds(self, contract, original, proposal):
        assert original.evidence["id"] == "host"
        return proposal.evidence.get("verified")


class GuardTests(unittest.TestCase):
    def run_guard(self, branches, host=None):
        if host is None:
            host = {"id": "host"}
        return guarded_release(ReleaseContract("test", ("gates", "risk")), host, branches, Verifier())

    def test_comparator_and_order(self):
        a = dict(verified={"gates": -2, "risk": Fraction(0)})
        b = dict(verified={"gates": -1, "risk": -10})
        c = dict(verified={"gates": -3, "risk": .25})
        r = self.run_guard([a, b, c, a.copy()])
        self.assertIs(r["policy"], a)
        self.assertEqual(r["selected"], 0)
        self.assertEqual([x["decision"] for x in r["branches"]],
                         ["eligible", "eligible", "non_regression_unproved", "eligible"])

    def test_missing_invalid_tied_and_failed(self):
        host = {"id": "host", "controller": object()}
        cases = [dict(verified={"gates": -1}), dict(verified={"gates": True, "risk": 0}),
                 dict(verified={"gates": 0, "risk": 0}), dict(feasibility="infeasible"),
                 dict(verified={"gates": float("nan"), "risk": -1}), dict(feasibility="missing")]
        r = self.run_guard(cases, host)
        self.assertIs(r["policy"], host)
        self.assertEqual([x["decision"] for x in r["branches"]], [
            "missing_bounds", "invalid_bounds", "no_certified_gain", "infeasible", "invalid_bounds", "missing"])

    def test_host_dispositions(self):
        b = {"id": "feasible"}
        self.assertEqual(self.run_guard([b], dict(feasibility="infeasible"))["status"], "recovery")
        self.assertEqual(self.run_guard([], dict(feasibility="infeasible"))["status"], "no_certified_recovery")
        for status in ("missing", "invalid", "unrecognised"):
            self.assertIsNone(self.run_guard([b], dict(feasibility=status))["policy"])

    def test_contract_and_verifier_registration(self):
        for losses in ((), ("",), ("x", "x")):
            with self.assertRaises(ValueError):
                ReleaseContract("test", losses)
        with self.assertRaises(ValueError):
            guarded_release(ReleaseContract("other", ("x",)), {}, [], Verifier())

    def test_real_cli_and_untrusted_fields(self):
        gate = lambda pos: dict(position=pos, layer=pos, gate="Z", qubits=[0])
        host = dict(wire_count=1, gates=[gate(1), gate(2), gate(3)])
        proposal = dict(wire_count=1, gates=[gate(1)])
        packet = dict(model="native-phase-anf-v1", target=dict(wire_count=1, supports=[[0]]),
                      losses=["gates", "depth", "active_wires"], caps={}, host=host, candidates=[proposal])
        with tempfile.TemporaryDirectory() as folder:
            job, out = Path(folder)/"in.json", Path(folder)/"out.json"
            job.write_text(json.dumps(packet))
            subprocess.run([sys.executable, "-m", "vqa_harness.cli", "guard-release", str(job),
                            "--out", str(out)], check=True, capture_output=True)
            receipt = json.loads(out.read_text())
            self.assertEqual(receipt["status"], "released")
            self.assertEqual(receipt["policy"], proposal)
        with self.assertRaises(ValueError):
            phase_release(dict(packet, validation_outcomes=[1, 1]))
        forged = dict(proposal, upper_differences={"gates": -1000})
        self.assertEqual(phase_release(dict(packet, candidates=[forged]))["branches"][0]["decision"], "invalid")
        self.assertEqual(phase_release(dict(packet, losses=["fidelity"]))["status"], "retained_host")
        self.assertEqual(phase_release(dict(packet, caps={"depth": 2}))["status"], "recovery")
        self.assertEqual(phase_release(dict(packet, caps={"latency": 2}))["status"], "unresolved_host")
        self.assertEqual(phase_release(dict(packet, candidates=[dict(wire_count=1, gates=[])]))
                         ["branches"][0]["feasibility"], "infeasible")


if __name__ == "__main__":
    unittest.main()
