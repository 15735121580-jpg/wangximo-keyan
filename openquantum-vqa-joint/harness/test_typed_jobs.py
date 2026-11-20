"""Exercise publication-root discovery, complete CLI runs and rechecked certificates."""

from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


class StructuredJobs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.job = json.loads((HERE / 'validation/typed-job.json').read_text())

    def call(self, expected_code):
        input_path = self.directory / 'input.json'
        input_path.write_text(json.dumps(self.job))
        completed = subprocess.run([sys.executable, '-B', str(HERE / 'run.py'), str(input_path),
                                    '--structured', '--out', 'run'], cwd=self.directory,
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, expected_code, completed.stderr)
        receipt = self.directory / 'run/result.json'
        return json.loads(receipt.read_text()) if receipt.exists() else None

    def test_exact_feedback_control(self):
        result = self.call(0)
        self.assertTrue(result['certificate_rechecked'])
        self.assertEqual(result['status'], 'optimal')
        self.assertEqual(result['selected'], 'deferred')
        self.assertEqual(Fraction(result['time']), 7)
        self.assertEqual(Fraction(result['controls']['x']), 1)

    def test_interval(self):
        self.job.update(mode='interval', tolerance='1/32', max_rounds=8)
        result = self.call(0)
        self.assertTrue(result['certificate_rechecked'])
        certificate = result['certificate']
        self.assertEqual(certificate['status'], 'delta_optimal')
        self.assertEqual(result['selected'], 'deferred')
        low, high = Fraction(certificate['lower']), Fraction(certificate['upper'])
        self.assertLessEqual(low, 7)
        self.assertGreaterEqual(high, 7)
        self.assertLessEqual(high-low, Fraction(1, 32))

    def test_uncertified_gap(self):
        self.job.update(mode='interval', tolerance=0, max_rounds=0)
        result = self.call(2)
        self.assertTrue(result['certificate_rechecked'])
        self.assertEqual(result['status'], 'undecided')

    def test_infeasible(self):
        for policy in self.job['policies']:
            policy['residuals'] = [1]
        result = self.call(2)
        self.assertTrue(result['certificate_rechecked'])
        self.assertEqual(result['status'], 'infeasible')
        self.assertIsNone(result['selected'])

    def test_incompatible_ports(self):
        self.job['policies'][1]['nodes'][0]['output'] = 'different'
        self.assertIsNone(self.call(2))

    def test_invalid_branch_kernel(self):
        self.job['policies'][1]['nodes'][0]['outcomes']['+'][0] = 1
        self.assertIsNone(self.call(2))


if __name__ == '__main__':
    unittest.main()
