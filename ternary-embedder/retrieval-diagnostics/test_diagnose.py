import unittest

import numpy as np
import torch

from diagnose import FrozenLinear, TernLinear, evaluate, paired


class DiagnosisTests(unittest.TestCase):
    def test_frozen_arithmetic_matches_training_module(self):
        torch.manual_seed(5)
        module = TernLinear(1536, 384)
        x = torch.randn(4, 1536)
        with torch.no_grad():
            for mode in (False, True):
                module.quant = mode
                torch.testing.assert_close(FrozenLinear(module, mode, mode)(x), module(x), atol=0, rtol=0)

    def test_graded_ndcg_and_margin(self):
        data = {"cids": ["a", "b", "c"], "qids": ["q"], "qrels": {"q": {"a": 2, "b": 1}}}
        summary, rows = evaluate(np.array([[0.8, 0.9, 0.85]]), data)
        dcg = 1 + 2 / np.log2(4)
        ideal = 2 + 1 / np.log2(3)
        self.assertAlmostEqual(summary["ndcg10"], 100 * dcg / ideal)
        self.assertEqual(rows[0]["best_relevant_id"], "b")
        self.assertEqual(rows[0]["best_competitor_id"], "c")
        self.assertAlmostEqual(rows[0]["best_relevant_margin"], 0.05)

    def test_ties_self_match_and_paired_direction(self):
        data = {"cids": ["q", "a", "b"], "qids": ["q"], "qrels": {"q": {"a": 1}}}
        _, bad = evaluate(np.array([[1.0, 0.5, 0.5]]), data)
        _, good = evaluate(np.array([[1.0, 0.6, 0.5]]), data)
        self.assertEqual(bad[0]["first_relevant_rank"], 2)
        self.assertEqual(good[0]["first_relevant_rank"], 1)
        delta = paired(bad, good)
        self.assertEqual(delta["top1_gained"], 1)
        self.assertGreater(delta["ndcg10_delta_right_minus_left"], 0)


if __name__ == "__main__":
    unittest.main()
