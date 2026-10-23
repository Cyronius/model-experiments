"""Measure sampling exposure from NPZ metadata without loading sprite pixels."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sampling import CATS, SpriteSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--draws', type=int, default=200_000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error('--draws must be positive')
    torch.set_num_threads(4)
    with np.load(args.data) as data:
        cats = data['cat']
        subtypes = data['creature_subtype'] if 'creature_subtype' in data else np.full(len(cats), 'unknown')
    cats = np.where(cats < 0, 8, cats)
    subtypes = np.where(subtypes == '', 'unknown', subtypes)
    report = dict(data=str(Path(args.data).resolve()), images=len(cats), draws=args.draws,
                  seed=args.seed, modes={}, notes=[
                      'Sampling exposure is not a distinct-design count or quality measurement.',
                      'Subtype mode gives equal creature mass to each populated human/monster/unknown pool.',
                      'Known subtype pools are small and may overfit; unknown images remain unknown.',
                      'Unlabelled share precedes independent 10% category-conditioning dropout.',
                      'No training was performed. This is not unseen validation.'])
    for mode in ['uniform', 'category', 'category-subtype']:
        sampler = SpriteSampler(cats, mode=mode, subtypes=subtypes, seed=args.seed)
        indices = sampler.draw(args.draws).numpy()
        rows = []
        for row in sampler.report:
            category = CATS.index(row['category']) if row['category'] != 'unlabelled' else 8
            selected = cats[indices] == category
            if row['subtype']:
                selected &= subtypes[indices] == row['subtype']
            rows.append(dict(row, observed_probability=float(selected.mean())))
        report['modes'][mode] = dict(pools=rows, unique_images_drawn=len(np.unique(indices)),
                                   category_probability={name: float((cats[indices] == c).mean())
                                                         for c, name in enumerate([*CATS, 'unlabelled'])},
                                   creature_subtype_probability={name: float(((cats[indices] == 0) & (subtypes[indices] == name)).mean())
                                                                 for name in ['human', 'monster', 'unknown']})
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
