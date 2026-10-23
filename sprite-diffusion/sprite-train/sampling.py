"""Explicit training distributions; subtype annotations never change output labels."""
import numpy as np
import torch

CATS = ('creature', 'item', 'weapon', 'food', 'effect', 'terrain', 'structure', 'nature')


class SpriteSampler:
    def __init__(self, categories, *, mode='uniform', subtypes=None,
                 unlabelled_prob=0.1, device='cpu', seed=0):
        categories = np.asarray(categories)
        if categories.ndim != 1 or not len(categories):
            raise ValueError('categories must be a nonempty vector')
        if not np.isin(categories, [-1, *range(9)]).all():
            raise ValueError('category IDs must be -1 or 0..8 (8 is unlabelled)')
        if mode not in {'uniform', 'category', 'category-subtype'}:
            raise ValueError('unknown sampling mode')
        if not 0 <= unlabelled_prob <= 1:
            raise ValueError('unlabelled probability must be in [0, 1]')
        categories = np.where(categories < 0, 8, categories)
        if subtypes is None:
            subtypes = np.full(len(categories), 'unknown')
        subtypes = np.asarray(subtypes)
        if subtypes.shape != categories.shape:
            raise ValueError('subtypes must align with categories')
        subtypes = np.where(subtypes == '', 'unknown', subtypes)
        if not np.isin(subtypes[categories == 0], ['human', 'monster', 'unknown']).all():
            raise ValueError('creature subtypes must be human, monster or unknown')

        present = [c for c in range(8) if (categories == c).any()]
        has_unlabelled = bool((categories == 8).any())
        if mode != 'uniform' and not present and unlabelled_prob != 1:
            raise ValueError('balanced sampling requires labelled sprites or unlabelled probability 1')
        null_mass = unlabelled_prob if has_unlabelled else 0.0
        buckets, probabilities, self.report = [], [], []
        for c in range(9):
            members = np.flatnonzero(categories == c)
            if not len(members):
                continue
            mass = (len(members) / len(categories) if mode == 'uniform' else
                    null_mass if c == 8 else (1 - null_mass) / len(present))
            parts = ([('human', members[subtypes[members] == 'human']),
                      ('monster', members[subtypes[members] == 'monster']),
                      ('unknown', members[subtypes[members] == 'unknown'])]
                     if mode == 'category-subtype' and c == 0 else [('', members)])
            parts = [(name, indices) for name, indices in parts if len(indices)]
            for subtype, indices in parts:
                probability = mass / len(parts)
                if probability == 0:
                    continue
                buckets.append(indices)
                probabilities.append(probability)
                self.report.append(dict(category=CATS[c] if c < 8 else 'unlabelled',
                                        subtype=subtype or None, images=len(indices),
                                        probability=probability,
                                        probability_per_image=probability / len(indices)))
        if not buckets:
            raise ValueError('sampling configuration gives no images positive probability')
        self.device = device
        self.indices = torch.as_tensor(np.concatenate(buckets), dtype=torch.long, device=device)
        lengths = np.array([len(bucket) for bucket in buckets])
        self.lengths = torch.as_tensor(lengths, device=device)
        self.offsets = torch.as_tensor(np.r_[0, np.cumsum(lengths)[:-1]], device=device)
        self.probabilities = torch.tensor(probabilities, dtype=torch.float32, device=device)
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def draw(self, batch):
        bucket = torch.multinomial(self.probabilities, batch, replacement=True, generator=self.generator)
        position = (torch.rand(batch, device=self.device, generator=self.generator)
                    * self.lengths[bucket]).long()
        return self.indices[self.offsets[bucket] + position]

    def get_state(self):
        return self.generator.get_state()

    def set_state(self, state):
        self.generator.set_state(state.cpu())
