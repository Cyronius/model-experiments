"""Regression checks for the ABI and scale boundaries that matter on hardware."""
import unittest

import numpy as np

from arithmetic import (dequantize, group_width, integer_products, pack_activations,
                        pack_weights, quantize, unpack_weights)


class ArithmeticTests(unittest.TestCase):
    def test_asymmetric_pack_roundtrip_and_reserved_code(self):
        rng = np.random.default_rng(71)
        weights = rng.integers(-1, 2, (64, 384), dtype=np.int8)
        packed = pack_weights(weights)
        self.assertEqual(packed.shape, (2, 24, 128))
        np.testing.assert_array_equal(unpack_weights(packed), weights)
        # Decode a hand-constructed byte to test all four code positions.
        packed = np.zeros((1, 1, 128), dtype=np.uint8)
        packed[0, 0, 0] = 0b11010011
        decoded = unpack_weights(packed)
        self.assertEqual([int(decoded[n, 0]) for n in (0, 16, 1, 17)], [-1, 0, 1, -1])
        packed[0, 0, 0] = 2
        with self.assertRaises(ValueError):
            unpack_weights(packed)

    def test_scale_threshold_and_ties_to_even(self):
        x = np.zeros((4, 384), dtype=np.float32)
        x[:, 0] = [127.0, np.nextafter(np.float32(127), np.float32(128)), 0, 1e-8]
        x[0, 1:7] = [0.5, 1.5, 2.5, -0.5, -1.5, -2.5]
        codes, scales = quantize(x)
        self.assertEqual(scales.shape, (4, 1))
        self.assertEqual(scales[0, 0], 1)
        self.assertEqual(scales[1, 0], 2)
        np.testing.assert_array_equal(codes[0, 1:7], [0, 2, 2, 0, -2, -2])
        self.assertTrue(np.isfinite(scales).all())
        self.assertTrue((scales > 0).all())
        self.assertFalse(codes[2:].any())

    def test_six_group_down_projection_cannot_use_one_scale(self):
        rng = np.random.default_rng(1536)
        x = rng.standard_normal((4, 1536)).astype(np.float32)
        x *= np.repeat(np.array([0.01, 0.1, 1, 10, 100, 1000], np.float32), 256)
        weights = rng.integers(-1, 2, (64, 1536), dtype=np.int8)
        aq, scales = quantize(x)
        self.assertEqual(group_width(1536), 256)
        self.assertEqual(scales.shape, (4, 6))
        products = integer_products(aq, weights)
        self.assertEqual(products.shape, (2, 6, 2, 4, 16))
        bias = rng.standard_normal(64).astype(np.float32)
        ws = np.full((64, 1), 0.125, np.float32)
        got = dequantize(products, scales, ws, bias)
        xq = (aq.reshape(4, 6, 256) * scales[..., None]).reshape(4, 1536)
        ref = (xq.astype(np.float64) @ weights.astype(np.float64).T) * ws.T + bias
        np.testing.assert_allclose(got, ref, atol=0.01, rtol=1e-6)
        wrong = (aq.astype(np.float64) @ weights.astype(np.float64).T) * scales[:, :1] * ws.T + bias
        self.assertGreater(float(np.abs(wrong - ref).max()), 100)
        packed_a = pack_activations(aq)
        np.testing.assert_array_equal(packed_a.transpose(1, 0, 2).reshape(4, 1536), aq)


if __name__ == "__main__":
    unittest.main()
