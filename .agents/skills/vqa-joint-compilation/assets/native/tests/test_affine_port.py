"""Check every local basis for every ordered pair of distinct three-bit words."""
from itertools import permutations, product
import unittest

import numpy as np

from vqa_joint.coherent_port import source_target
from vqa_joint.sector_dispatch import Readout, instrument_plans
from vqa_joint.sector_vqe.affine_port import BASIS, readout


class AffinePortTests(unittest.TestCase):
    def test_all_three_wire_effects(self):
        for axes in product('XYZ',repeat=3):
            basis = np.ones((1,1))
            for axis in reversed(axes):
                basis = np.kron(basis,BASIS[axis])
            for words in permutations(range(8),2):
                low,rows,offset,r = readout(words,axes)
                kernel = Readout('affine_two_word',3,1,r,decode_rows=rows,xor_offset=offset)
                columns = basis[:,words]
                source = np.einsum('xi,xj->xij',columns.conj(),columns)
                z = BASIS[low]
                lowered = np.array([kernel.lift_distribution(np.eye(2)[bit]) for bit in (0,1)])
                actual = np.einsum('bx,bi,bj->xij',lowered,z.conj(),z)
                self.assertLess(abs(source-actual).max(),1e-12,(words,axes))

    def test_terminal_prefix_and_fanout_precedence(self):
        meas = tuple(('measure',(q,),None) for q in range(3))
        prep = (('ry',(1,),.71),('s',(1,),None))
        enc = (('cx',(1,0),None),('x',(2,),None))
        tail = (('sdg',(1,),None),('h',(1,),None),('h',(2,),None))
        ops = prep+enc+tail+meas
        identity, compressed = instrument_plans(ops,3)
        self.assertEqual((identity.variant,compressed.variant),('identity','affine_two_word'))
        self.assertEqual(instrument_plans(ops,3,False),[identity])
        self.assertEqual(compressed.readout.random_width,2)
        space = source_target(ops,'affine_two_word')
        self.assertEqual(space.port.words,(4,7))
        self.assertEqual(space.port.axes,('Z','Y','X'))
        self.assertTrue(np.allclose(space.entry.conj().T@space.entry,np.eye(2)))
        legacy = prep+(('cx',(1,0),None),)+meas
        self.assertEqual([p.variant for p in instrument_plans(legacy,3)],['identity','fanout_Z'])
        invalid = prep+enc+(('rx',(0,),.21),)+meas
        self.assertEqual([p.variant for p in instrument_plans(invalid,3)],['identity'])
        with self.assertRaises(ValueError):
            source_target(invalid,'affine_two_word')


if __name__ == '__main__':
    unittest.main()
