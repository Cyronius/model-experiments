"""Training augmentation must preserve geometry, palette structure and labels."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'sprite-train'))
from augment import palette_swap


class PaletteAugmentationTests(unittest.TestCase):
    def fixture(self):
        palette = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 255], [24, 24, 24, 255],
                                [120, 120, 120, 255], [255, 255, 255, 255],
                                [64, 25, 12, 255], [135, 65, 25, 255],
                                [210, 145, 55, 255], [50, 150, 60, 128]], dtype=torch.uint8)
        rgba = palette[torch.arange(32 * 32).reshape(32, 32) % len(palette)]
        x = rgba.permute(2, 0, 1).float() / 127.5 - 1
        return x.unsqueeze(0).repeat(9, 1, 1, 1), torch.arange(9)

    def test_preserves_alpha_background_neutrals_and_other_categories(self):
        x, categories = self.fixture()
        original = x.clone()
        y = palette_swap(x, categories, probability=1, generator=torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(x, original))
        self.assertTrue(torch.equal(x[:, 3], y[:, 3]))
        self.assertTrue(torch.equal(x[[0, 1, 4, 5, 8]], y[[0, 1, 4, 5, 8]]))
        protected = ((x[:, 0] == x[:, 1]) & (x[:, 1] == x[:, 2])) | (x[:, 3] == -1)
        self.assertTrue(torch.equal(x[:, :3].permute(0, 2, 3, 1)[protected],
                                    y[:, :3].permute(0, 2, 3, 1)[protected]))
        for i in [2, 3, 6, 7]:
            self.assertFalse(torch.equal(x[i], y[i]))
        self.assertTrue(torch.equal(palette_swap(x, categories, probability=0), x))

    def test_preserves_luminance_and_does_not_add_colours(self):
        x, categories = self.fixture()
        y = palette_swap(x, categories, probability=1, generator=torch.Generator().manual_seed(2))
        weights = torch.tensor([0.2126, 0.7152, 0.0722]).reshape(1, 3, 1, 1)
        difference = (((y[:, :3] - x[:, :3]) * 0.5) * weights).sum(1).abs()
        self.assertLessEqual(float(difference.max()), 0.501 / 255)
        self.assertGreaterEqual(float(y.min()), -1)
        self.assertLessEqual(float(y.max()), 1)
        for before, after in zip(x, y):
            opaque = before[3].reshape(-1) > -1
            a = before[:3].reshape(3, -1)[:, opaque].T
            b = after[:3].reshape(3, -1)[:, opaque].T
            self.assertLessEqual(len(torch.unique(b, dim=0)), len(torch.unique(a, dim=0)))

    def test_repeatable_given_generator(self):
        x, categories = self.fixture()
        a = palette_swap(x, categories, generator=torch.Generator().manual_seed(17))
        b = palette_swap(x, categories, generator=torch.Generator().manual_seed(17))
        self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
