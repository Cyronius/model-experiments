"""Show real training sprites beside reproducible palette variations."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from augment import palette_swap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='C:/dev/sprite-data/trainset.npz')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=23)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    categories = ['structure', 'nature', 'food', 'weapon']
    with np.load(args.data) as data:
        names = data['cats'].tolist()
        labels, groups, sprites = data['cat'], data['group'], data['sprites']
        chosen = []
        for category in categories:
            pool = rng.permutation(np.flatnonzero(labels == names.index(category)))
            seen_groups, selected = set(), []
            for index in pool:
                if int(groups[index]) in seen_groups or (sprites[index, :, :, 3] > 0).sum() < 60:
                    continue
                seen_groups.add(int(groups[index]))
                selected.append(int(index))
                if len(selected) == 8:
                    break
            if len(selected) < 8:
                raise ValueError(f'need eight source groups for {category}')
            chosen.extend(selected)
        original = sprites[chosen].copy()
        cat = labels[chosen].copy()
    del sprites
    x = torch.from_numpy(original).permute(0, 3, 1, 2).float() / 127.5 - 1
    generator = torch.Generator().manual_seed(args.seed)
    versions = [original]
    for _ in range(3):
        y = palette_swap(x, torch.from_numpy(cat), probability=1, generator=generator)
        versions.append(((y + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy())
    for version in versions[1:]:
        assert np.array_equal(version[..., 3], original[..., 3])
    # Two category columns. Each row has the original and three palette variants.
    canvas = Image.new('RGB', (896, 1800), '#20242a')
    draw = ImageDraw.Draw(canvas)
    try:
        title = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 24)
        small = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 17)
    except OSError:
        title, small = ImageFont.load_default(size=24), ImageFont.load_default(size=17)
    for block, category in enumerate(categories):
        left, top = (block % 2) * 448, (block // 2) * 900
        draw.text((left + 12, top + 6), category.capitalize(), font=title, fill='#f0f2f4')
        for col, label in enumerate(['Original', 'Palette 1', 'Palette 2', 'Palette 3']):
            draw.text((left + 12 + col * 108, top + 42), label, font=small, fill='#c0c6cf')
        for row in range(8):
            for col, version in enumerate(versions):
                tile = Image.new('RGBA', (96, 96), '#343941')
                tile.alpha_composite(Image.fromarray(version[block * 8 + row]).resize((96, 96), Image.Resampling.NEAREST))
                canvas.paste(tile.convert('RGB'), (left + 12 + col * 108, top + 72 + row * 102))
    canvas.save(out / 'palette-preview.png')
    np.savez_compressed(out / 'palette-preview.npz', original=original, variants=np.stack(versions[1:]),
                        source_index=np.array(chosen), category=cat)
    metadata = {'data': args.data, 'seed': args.seed, 'hue_degrees': 25,
                'categories': categories, 'examples_per_category': 8,
                'variants_per_example': 3, 'source_index': chosen,
                'note': 'Augmented copies are variations of the same designs, not new designs.'}
    (out / 'palette-preview.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(out / 'palette-preview.png')


if __name__ == '__main__':
    main()
