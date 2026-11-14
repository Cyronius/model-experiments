import copy
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tern-embed'))
from model import Student


class ProjectionFoldTests(unittest.TestCase):
    def test_fold_matches_actual_ternary_model_and_keeps_backbone_identical(self):
        torch.manual_seed(71)
        torch.set_num_threads(2)
        model = Student(vocab=64, emb=8, d=8, heads=2, ffn=16, layers=3,
                        max_pos=12, out_dim=8).eval()
        model.set_quant(True)
        with torch.no_grad():
            model.out.weight.copy_(torch.randn(8, 8))
            matrix = torch.eye(8) + .02 * torch.randn(8, 8)
            ids = torch.randint(0, 64, (7, 12))
            mask = torch.arange(12)[None, :] < torch.tensor([12, 7, 4, 8, 5, 9, 2])[:, None]
            expected = F.normalize(model(ids, mask)[2] @ matrix.T, dim=-1)
            folded = copy.deepcopy(model)
            folded.out.weight.copy_(matrix @ model.out.weight)
            actual = folded(ids, mask)[2]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        self.assertTrue(folded.quantized)
        for name, value in model.state_dict().items():
            if name != 'out.weight':
                self.assertTrue(torch.equal(value, folded.state_dict()[name]), name)


if __name__ == '__main__':
    unittest.main()
