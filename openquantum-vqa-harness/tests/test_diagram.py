import unittest
from dataclasses import replace
from vqa_harness.readout_diagram import bound_readout, rewrite_diagram, ParityPort, QuantumPort


class DiagramTests(unittest.TestCase):
    def test_measuring_z_after_x_keeps_sign_and_score_key(self):
        diagram = bound_readout((5,), [("x",(0,),())], 2, "Z", -0.8)
        revised = rewrite_diagram(diagram,"P")
        self.assertEqual(revised.native_input,((),"Z",-1))
        self.assertEqual(revised.score,diagram.score)
        self.assertEqual(revised.measurement.output,revised.score.input)

    def test_reject_mismatched_quantum_and_classical_joins(self):
        diagram = bound_readout((5,), [], 1, "X", -0.6)
        with self.assertRaises(ValueError):
            replace(diagram,score=replace(diagram.score,input=ParityPort(0,"X")))
        with self.assertRaises(ValueError):
            replace(diagram,measurement=replace(diagram.measurement,input=QuantumPort((11,))))

    def test_commuting_and_disjoint_rules(self):
        diagram = bound_readout((5,11), [("ry",(1,),(0.3,)),("rz",(0,),(0.4,))],0,"ZI",1.)
        self.assertEqual(rewrite_diagram(diagram,"RD").native_input,((),"ZI",1))
        self.assertEqual(rewrite_diagram(diagram,"P"),diagram)
