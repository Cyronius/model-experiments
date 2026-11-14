"""Freeze Q4 group32/group16 exports; only scale overhead changes, no labels."""
import datetime
import json
from pathlib import Path

import torch

from export_pair_precision import export
from pair_training_state import atomic_json, digest

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    frozen = ROOT / 'q4-groups/results/frozen.json'
    if frozen.exists():
        raise RuntimeError('Already frozen')
    source = json.loads((ROOT / 'float-blends/results/frozen.json').read_text())['models']['float-25']
    assert digest(Path(source['path'])) == source['sha256']
    models = {}
    for group in (32, 16):
        name = f'q4-g{group}'
        output = ROOT / 'work/q4-groups' / (name + '.pt')
        output.parent.mkdir(parents=True, exist_ok=True)
        report = export(Path(source['path']), output, 4, group_size=group)
        models[name] = dict(path=str(output.resolve()), sha256=report['sha256'], bytes=report['bytes'],
                            precision='q4', layers=3, eligible_for_promotion=False,
                            source_float_sha256=source['sha256'], bits=4, activation_bits=8, group_size=group,
                            note='Four-bit dense codes; more BF16 scales, with embeddings/output still float. Packed accuracy/reference artifact; native kernel speed unmeasured.')
    frozen.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(frozen, dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), local_only=True,
                             selected_before_external_scores=True, no_calibration_data_or_labels=True, models=models))
    print('Both Q4 group-size variants frozen before scoring.', flush=True)


if __name__ == '__main__':
    main()
