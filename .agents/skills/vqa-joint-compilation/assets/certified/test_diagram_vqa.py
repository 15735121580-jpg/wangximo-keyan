"""Exact symbolic checks; run 'python3 -m unittest test_diagram_vqa' here."""
from itertools import product
from unittest import TestCase, main

from diagram_vqa import (Model, Op, Step, example_kernel_catalogue,
                         native_word, relabel, replay_trace, saturate)


def cut_value(edges, bits):
    return sum(w for i, j, w in edges if bits[i] != bits[j])


def flip(bits, i):
    return bits[:i] + (1 - bits[i],) + bits[i + 1:]


class QAOARewriteTests(TestCase):
    def setUp(self):
        self.model, self.seed, self.options = example_kernel_catalogue()

    def test_path_and_proofs(self):
        self.assertEqual([tuple(g.kind for g in c.program) for c in self.options],
                         [("Q", "Layer", "M", "Fixed"),
                          ("Layer", "Q", "M", "Fixed"),
                          ("Layer", "M", "C", "Fixed")])
        for c in self.options:
            self.assertEqual(replay_trace(self.model, self.seed, c.trace), c.program)
            self.assertEqual(native_word(self.model, c.program), c.native)
            self.assertEqual(c.native[-1], ("VISIBLE_FEEDBACK", self.model.feedback))
        last = self.options[-1]
        inverse = (Step("MC->QM", 1), Step("LQ->QL", 0))
        self.assertEqual(replay_trace(self.model, last.program, inverse), self.seed)

    def test_weighted_cost_and_mixer_generators(self):
        # The cut Hamiltonian is diagonal. The mixer sends a basis state to
        # the sum of its one-bit flips. Compare integer labels, not samples.
        g = self.seed[0].arg
        for bits in product((0, 1), repeat=self.model.n):
            after = relabel(bits, g)
            self.assertEqual(cut_value(self.model.edges, bits),
                             cut_value(self.model.edges, after))
            for i in range(self.model.n):
                self.assertEqual(relabel(flip(bits, i), g), flip(after, g[i]))
            for c in self.options:
                ports = list(bits)
                for instruction in c.native:
                    if instruction[0] == "SWAP":
                        _, u, v = instruction
                        ports[u], ports[v] = ports[v], ports[u]
                    if instruction[0] == "RELABEL_RECORD":
                        ports = list(relabel(tuple(ports), instruction[1]))
                self.assertEqual(tuple(ports), after)

    def test_illegal_and_truncated(self):
        m = self.model
        unequal = Model(m.n, ((0, 1, 1), (0, 2, 2), (1, 2, 1)), m.layers, m.feedback)
        self.assertEqual(len(saturate(unequal, self.seed)), 1)
        self.assertEqual(len(saturate(m, self.seed, max_steps=1)), 2)
        with self.assertRaises(ValueError):
            replay_trace(unequal, self.seed, self.options[-1].trace)
        with self.assertRaises(ValueError):
            replay_trace(m, self.seed, (Step("QM->MC", 2),))
        with self.assertRaises(ValueError):
            saturate(Model(m.n, m.edges, m.layers, m.feedback, (0,)), self.seed)
        with self.assertRaises(ValueError):
            saturate(m, self.seed[:-1] + (Op("Fixed", "read_hidden_fault"),))


if __name__ == "__main__":
    main()
