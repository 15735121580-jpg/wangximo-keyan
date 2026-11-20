"""Cost-certificate regression on an exact symbolic fixture, no device sampling."""
from fractions import Fraction
from unittest import TestCase, main

from compile_qaoa_diagram import Limits, PrimitiveCost, compile_kernel


def table(pre=3, post=2, classical=1, classical_failure=0):
    return {
        "RZZ": PrimitiveCost(1, 1, 0), "RX": PrimitiveCost(1, 1, 0),
        "MEASURE_Z_ALL": PrimitiveCost(2, 1, 0),
        "VISIBLE_FEEDBACK": PrimitiveCost(1, 0, 0),
        "SWAP_BEFORE_LAYER": PrimitiveCost(pre, 1, 2, Fraction(1, 10)),
        "SWAP_AFTER_LAYER": PrimitiveCost(post, 1, 1, Fraction(1, 10)),
        "RELABEL_RECORD": PrimitiveCost(classical, 0, 0, classical_failure),
    }


class KernelExtractionTests(TestCase):
    def test_switch_and_certificate(self):
        limits = Limits(5, 9, 1, 0)
        for prices, expected_time, key in (
                (table(), 10, "RELABEL_RECORD"),
                (table(classical=8), 13, "SWAP_AFTER_LAYER"),
                (table(pre=1, classical=8), 11, "SWAP_BEFORE_LAYER")):
            result = compile_kernel(prices, limits, data_atoms=3)
            self.assertEqual(result.status, "optimal")
            w = result.diagram.program
            self.assertEqual(result.time.as_fraction(), expected_time)
            self.assertEqual(result.time.as_fraction(), w.time)
            self.assertIn(key, w.price_keys)
            self.assertTrue(w.candidate.native)

    def test_independent_budgets_gate_selection(self):
        # Even the fastest realignment is invalid if its error budget fails.
        prices = table(classical_failure=Fraction(1, 4))
        result = compile_kernel(prices, Limits(4, 9, Fraction(1,5), Fraction(4,5)),
                                data_atoms=3)
        word = result.diagram.program
        self.assertIn("SWAP_AFTER_LAYER", word.price_keys)
        self.assertEqual(word.fidelity, Fraction(81, 100))
        self.assertEqual(word.failure, Fraction(19, 100))
        self.assertEqual(compile_kernel(prices, Limits(3, 7, 0, 1),
                                        data_atoms=3).status, "infeasible")

    def test_reject_floating_calibration(self):
        with self.assertRaises(ValueError):
            PrimitiveCost(1, 0, 0, 0.2)


if __name__ == "__main__":
    main()
