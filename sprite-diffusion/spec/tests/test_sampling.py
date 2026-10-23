"""Check statistical exposure, unknown subtype handling and checkpoint replay."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'sprite-train'))
from sampling import SpriteSampler


class SamplingTests(unittest.TestCase):
    def fixture(self):
        categories = np.array([0] * 1000 + [1] * 100 + [2] * 10 + [-1] * 2000)
        subtypes = np.array(['human'] * 10 + ['monster'] * 20 + ['unknown'] * 970 + [''] * 2110)
        return categories, subtypes

    def test_uniform_and_category_exposure(self):
        cats, _ = self.fixture()
        for mode, expected in [('uniform', [1000/3110, 100/3110, 10/3110, 2000/3110]),
                               ('category', [.3, .3, .3, .1])]:
            sampler = SpriteSampler(cats, mode=mode)
            selected = cats[sampler.draw(100_000).numpy()]
            observed = [(selected == c).mean() for c in [0, 1, 2, -1]]
            np.testing.assert_allclose(observed, expected, atol=.006, rtol=0)

    def test_subtypes_balanced_without_relabelling_unknowns(self):
        cats, subtypes = self.fixture()
        original = subtypes.copy()
        sampler = SpriteSampler(cats, mode='category-subtype', subtypes=subtypes)
        ix = sampler.draw(100_000).numpy()
        np.testing.assert_array_equal(subtypes, original)
        for name in ['human', 'monster', 'unknown']:
            self.assertAlmostEqual(float(((cats[ix] == 0) & (subtypes[ix] == name)).mean()), .1, delta=.006)
        self.assertEqual([r['images'] for r in sampler.report[:3]], [10, 20, 970])

    def test_missing_annotations_and_categories(self):
        sampler = SpriteSampler([0, 0, 7], mode='category-subtype')
        self.assertEqual([r['probability'] for r in sampler.report], [.5, .5])
        self.assertEqual(sampler.report[0]['subtype'], 'unknown')
        self.assertTrue(set(sampler.draw(100).tolist()) <= {0, 1, 2})
        only_null = SpriteSampler([-1, 8], mode='category', unlabelled_prob=1)
        self.assertEqual(only_null.report[0]['probability'], 1)

    def test_replay_from_checkpoint_state(self):
        cats, _ = self.fixture()
        sampler = SpriteSampler(cats, mode='category', seed=3)
        sampler.draw(23)
        state = sampler.get_state()
        expected = sampler.draw(100)
        resumed = SpriteSampler(cats, mode='category', seed=99)
        resumed.set_state(state)
        self.assertTrue(torch.equal(resumed.draw(100), expected))

    def test_bad_inputs_and_zero_mass(self):
        for categories, options in [([], {}), ([9], {}), ([.5], {}),
                                    ([0], {'unlabelled_prob': 2}),
                                    ([0], {'mode': 'invalid'}),
                                    ([-1], {'mode': 'category'}),
                                    ([0], {'subtypes': ['elf']}),
                                    ([0], {'subtypes': []}),
                                    ([0, -1], {'mode': 'category', 'unlabelled_prob': 0})]:
            if categories == [0, -1]:
                sampler = SpriteSampler(categories, **options)
                self.assertEqual(sampler.draw(20).tolist(), [0] * 20)
            else:
                with self.assertRaises(ValueError):
                    SpriteSampler(categories, **options)


if __name__ == '__main__':
    unittest.main()
