import unittest

import torch

from export_pair_precision import pack_weight, unpack_weight


class Q4MseTests(unittest.TestCase):
    def test_smaller_group_metadata_roundtrips_actual_four_bit_codes(self):
        weight = torch.full((2, 128), .1)
        weight[:, ::64] = 1.
        original = pack_weight(weight, 4)
        smaller = pack_weight(weight, 4, group_size=16)
        self.assertEqual(smaller['group'], 16)
        self.assertEqual(smaller['packed'].numel(), weight.numel() // 2)
        self.assertEqual(smaller['scales'].numel(), original['scales'].numel() * 4)
        self.assertLess(float((unpack_weight(smaller) - weight).square().mean()),
                        float((unpack_weight(original) - weight).square().mean()))

    def test_actual_packed_error_never_exceeds_included_maxabs_candidate(self):
        weight = torch.full((2, 128), .1)
        weight[:, ::64] = 1.
        original = pack_weight(weight, 4)
        optimized = pack_weight(weight, 4, 'mse-grid')
        old_error = (unpack_weight(original) - weight).square().mean()
        new_error = (unpack_weight(optimized) - weight).square().mean()
        self.assertLess(float(new_error), float(old_error))
        self.assertEqual(optimized['packed'].numel(), weight.numel() // 2)
        self.assertEqual(optimized['scales'].dtype, torch.bfloat16)
        self.assertEqual(optimized['group'], 64)


if __name__ == '__main__':
    unittest.main()
