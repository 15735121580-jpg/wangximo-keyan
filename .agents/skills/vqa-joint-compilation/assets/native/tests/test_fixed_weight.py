"""Regression checks for arbitrary fixed-cardinality ports and coherent inputs."""
import unittest

import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile

from vqa_joint.sector_qaoa.fixed_weight import CostMixerLayer, FixedWeightPort


def qasm_for(port, prep, layers):
    """Build the independent full-wire program without using restricted matrices."""
    qc = QuantumCircuit(port.n)
    for i in range(port.n):
        if (port.seed_mask >> i) & 1:
            qc.x(i)
    def exchanges(seq):
        for i, j, beta in seq:
            qc.rxx(beta, i, j)
            qc.ryy(beta, i, j)
    exchanges(prep)
    for layer in layers:
        # No assumptions about the energies of infeasible masks.
        diag = [np.exp(-1j * layer.gamma * layer.energy(mask)) for mask in range(1 << port.n)]
        qc.unitary(np.diag(diag), list(range(port.n)))
        exchanges(layer.exchanges)
    return qasm2.dumps(transpile(qc, basis_gates=["u3", "cz"], optimization_level=1, seed_transpiler=0))


class FixedWeightTests(unittest.TestCase):
    def test_catalog_and_padding(self):
        for n in range(1, 9):
            for k in range(n+1):
                port = FixedWeightPort(n, k, max_sector_dimension=100)
                expected = tuple(m for m in range(1<<n) if m.bit_count() == k)
                self.assertEqual(port.labels, expected)
                for i, mask in enumerate(expected):
                    self.assertEqual(port.encode(mask), i)
                    self.assertEqual(port.decode(i), mask)
                for i in range(len(expected), 1 << port.width):
                    self.assertIsNone(port.decode(i))

    def test_two_layer_general_graph_nonminimal_seed(self):
        port = FixedWeightPort(4, 2, seed_mask=10)
        prep = ((1, 3, -.401), (1, 2, .27))
        def scores(m):
            return .19 * (m & 2) - .031 * (m & 4) * (m & 1)
        layers = (CostMixerLayer(.73, scores, ((0,3,.12),(0,1,-.31))),
                  CostMixerLayer(-.29, lambda m: .083*m*m, ((2,1,.66),(1,3,.31))))
        compact = port.bind(prep, layers)
        c = port.implementations(qasm_for(port, prep, layers), compact)
        self.assertEqual(list(c), ["original", "original_o3", "joint"])
        self.assertEqual([c[key].width for key in c], [4, 4, 3])
        self.assertTrue(all(plan.verified_residual < 1e-10 for plan in c.values()))
        seed = port.encode(port.seed_mask)
        for name, plan in c.items():
            self.assertEqual(np.argmax(plan.entry[:, seed]), 0, name)
            self.assertIsNotNone(plan.qasm)
        self.assertEqual(c["joint"].readout[-2:], (None, None))

    def test_constant_sector_has_no_quantum_instructions(self):
        for k in (0,3):
            port = FixedWeightPort(3,k)
            layers = [CostMixerLayer(.77, lambda m: m+.5, ((0,1,-.4),))]
            c = port.implementations(qasm_for(port, (), layers), port.bind((), layers))
            self.assertEqual(c["joint"].width,0)
            self.assertEqual(c["joint"].readout,(port.seed_mask,))
            self.assertIsNone(c["joint"].qasm)

    def test_reject_wrong_coherent_continuation_with_correct_initial_output(self):
        port = FixedWeightPort(3,1)
        reference = qasm_for(port, (), [])
        bad = np.diag([1., -1., 1.])
        with self.assertRaisesRegex(ValueError,"Coherent task-port"):
            port.implementations(reference, bad)

    def test_nonfinite_wrong_wires_and_budget(self):
        with self.assertRaisesRegex(ValueError,"audit budget"):
            FixedWeightPort(200,1)
        with self.assertRaisesRegex(ValueError,"synthesis budget"):
            FixedWeightPort(8,4)
        with self.assertRaisesRegex(ValueError,"declared weight"):
            FixedWeightPort(4,2,seed_mask=1)
        port = FixedWeightPort(3,1)
        with self.assertRaises(ValueError):
            port.bind([(1,1,.1)],[])
        with self.assertRaises(ValueError):
            port.bind([], [CostMixerLayer(np.inf, lambda m: m, ())])
        with self.assertRaises(ValueError):
            port.encode(0)
        with self.assertRaises(ValueError):
            port.decode(7)


if __name__ == "__main__":
    unittest.main()
