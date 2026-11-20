"""Installed CLI regressions for archived, complete encoded jobs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CLI = Path(sys.executable).with_name("vqa-joint")


class EncodedCLI(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def request(self, constraints, name, *, expected):
        packet = dict(host="ZAC", task="steane_grid_vqe", train_shots_per_basis=64, constraints=constraints)
        filename = self.root / (name + ".json")
        filename.write_text(json.dumps(packet))
        out = self.root/name
        cp = subprocess.run([str(CLI), "compile-encoded", str(filename), "--out", str(out)], capture_output=True)
        self.assertEqual(cp.returncode, expected, cp.stderr.decode())
        return out, json.loads((out/"result.json").read_text())

    def test_hybrid_closed_loop(self):
        out, plan = self.request({"min_application_success":.91146}, "hybrid", expected=0)
        self.assertEqual(plan["status"], "optimal_in_catalogue")
        selected = plan["selected"]["assignment"]
        self.assertEqual([k for k in selected if selected[k] == "verified"], ["0/Z"])
        self.assertEqual(len(plan["candidates"]), 64)
        native = json.loads((out/"native-files.json").read_text())
        self.assertTrue((out/native["architecture"]).is_file())
        for key, files in native["bindings"].items():
            folder = out/files["directory"]
            self.assertEqual(files["arm"], selected[key])
            emitted = json.loads((folder/"bound_schedule.json").read_text())
            circuit = plan["program"]["bindings"][key]
            self.assertEqual(emitted, circuit["native_schedule"])
            self.assertEqual(len(emitted["gates"]), circuit["gate_count"])
            self.assertTrue((out/json.loads((folder/"native_trace.json").read_text())["architecture_spec_path"]).is_file())
            self.assertTrue((folder/"call.qasm").is_file())
        replay = subprocess.run([str(CLI), "replay-encoded", str(out/"result.json"), "--seed", "42"],
                                check=True, capture_output=True)
        draw = json.loads(replay.stdout)
        self.assertGreaterEqual(draw["attempts"], draw["accepted"])
        self.assertEqual(draw["accepted"], 2*64*len(draw["history"]))
        self.assertEqual(draw["peak_atoms"], 14)
        self.assertTrue(all(sum(r["raw_atom_bitstrings"].values()) == r["attempts"]
                            for h in draw["history"] for r in h["measurement"]))

    def test_unknown_vs_infeasible(self):
        for cs, status in [({"max_peak_atoms":6}, "infeasible_in_catalogue"),
                           ({"min_circuit_process_fidelity":.8}, "uncertified"),
                           ({"max_round_logical_error":1e-5}, "uncertified")]:
            out, plan = self.request(cs, next(iter(cs)), expected=2)
            self.assertEqual(plan["status"], status)
            self.assertIsNone(plan["selected"])
            self.assertFalse((out/"native-files.json").exists())
