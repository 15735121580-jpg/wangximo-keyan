"""Independent numerical and accounting checks for complete local VQE jobs."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from qiskit import qasm2

from vqa_harness.workflow import _angle, run_job


def fixture():
    return {
        "task": {"n_qubits": 2, "initial_bits": [0, 0],
                 "gates": [{"op": "ry", "qubits": [0], "angle": "theta"},
                           {"op": "cx", "qubits": [0, 1]}],
                 "observables": [{"label": "energy", "terms": [
                     {"coeff": -.7, "pauli": "ZI"}, {"coeff": -.4, "pauli": "XX"},
                     {"coeff": -.12, "pauli": "ZZ"}]}]},
        "device": {"n_qubits": 2, "edges": [[0, 1]], "native": ["rz", "sx", "x", "cz"],
                   "gate_times_us": {"rz": 0, "sx": .04, "x": .04, "cz": .3},
                   "reset_us": 1, "readout_us": 1,
                   "gate_fidelities": {"one_qubit": .9999, "two_qubit": .999}},
        "vqa": {"parameter_grid": [{"theta": 0}, {"theta": math.pi/4}, {"theta": math.pi/2}],
                "shots_per_term": 32, "terminal_shots_per_term": 64, "seed": 41},
        "platform": {"name": "explicit synthetic two-site local model"}}


class WorkflowTests(unittest.TestCase):
    def test_full_job_equivalence_answer_resources_and_replay(self):
        job = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            result = run_job(job, out)
            again = run_job(job, Path(temporary) / "again")
            self.assertEqual(result["arms"], again["arms"])
            self.assertTrue(result["verification"]["all_ideal_pauli_checks_passed"])
            for actual, params in zip(result["reference"]["ideal_grid_energies"], job["vqa"]["parameter_grid"]):
                theta = params["theta"]
                self.assertAlmostEqual(actual, -.7*math.cos(theta)-.4*math.sin(theta)-.12)
            self.assertAlmostEqual(result["reference"]["exact_ground_energy"], -.12-math.hypot(.7,.4))
            self.assertEqual(result["arms"]["host"]["resources"]["requested_pauli_samples"], 3*3*32+3*64)
            self.assertEqual(result["arms"]["joint"]["resources"]["requested_pauli_samples"], 3*3*32+3*64)
            self.assertGreater(result["arms"]["joint"]["resources"]["deterministic_samples"], 0)
            self.assertGreater(result["arms"]["host"]["resources"]["duration_us"], 2*(3*3*32+3*64))
            self.assertGreater(result["arms"]["host"]["resources"]["two_qubit_count"], 0)
            self.assertEqual(result["arms"]["joint"]["resources"]["two_qubit_count"], 0)
            self.assertEqual(json.loads((out/"result.json").read_text())["arms"], result["arms"])
            self.assertTrue((out/"replay.py").is_file())
            for path in (out/"host"/"terminal").glob("*.qasm"):
                self.assertGreater(qasm2.load(str(path)).num_clbits, 0)
            with self.assertRaises(FileExistsError):
                run_job(job, out)

    def test_noise_is_evaluated_after_basis_changes(self):
        job = fixture()
        job["device"]["gate_fidelities"] = {"one_qubit": 1, "two_qubit": 1}
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)/"ideal"
            run_job(job, out)
            circuits = json.loads((out/"circuits.json").read_text())
            for arm in circuits.values():
                for settings in arm:
                    for setting in settings:
                        self.assertAlmostEqual(setting["ideal_expectation"], setting["noisy_expectation"])
                        if setting["state_fidelity_to_own_ideal"] is not None:
                            self.assertAlmostEqual(setting["state_fidelity_to_own_ideal"], 1)
            noisy = copy.deepcopy(job)
            noisy["device"]["gate_fidelities"]["one_qubit"] = .98
            run_job(noisy, Path(temporary)/"noisy")
            settings = json.loads((Path(temporary)/"noisy"/"circuits.json").read_text())
            # At theta=pi/2 the transformed X readout must include noisy basis gates.
            self.assertLess(abs(settings["joint"][2][1]["noisy_expectation"]), .999)

    def test_arithmetic_and_invalid_jobs_fail(self):
        self.assertAlmostEqual(_angle("-(theta + pi/4)", {"theta": .2}), -(.2+math.pi/4))
        for expression in ("__import__('os').system('x')", "theta[0]", "missing", "1/0", "1e309"):
            with self.assertRaises(ValueError):
                _angle(expression, {"theta": .2})
        with tempfile.TemporaryDirectory() as temporary:
            for mutate in (lambda j: j["vqa"].update(shots_per_term=0),
                           lambda j: j["device"]["gate_fidelities"].update(one_qubit=1.1),
                           lambda j: j["vqa"]["parameter_grid"].append({}),
                           lambda j: j["device"]["gate_times_us"].update(cz=float("nan"))):
                job = fixture()
                mutate(job)
                with self.assertRaises(ValueError):
                    run_job(job, Path(temporary)/"invalid")
            self.assertFalse((Path(temporary)/"invalid").exists())


if __name__ == "__main__":
    unittest.main()
