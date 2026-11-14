"""Freeze a better weight-MSE Q4 export; no data/labels or paid service used."""
import datetime
import json
from pathlib import Path

import torch

from export_pair_precision import export
from pair_training_state import atomic_json, digest

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    frozen = ROOT / 'q4-mse/results/frozen.json'
    if frozen.exists():
        raise RuntimeError('Already frozen')
    source = json.loads((ROOT / 'float-blends/results/frozen.json').read_text())['models']['float-25']
    assert digest(Path(source['path'])) == source['sha256']
    output = ROOT / 'work/q4-mse/model-q4-mse.pt'
    output.parent.mkdir(parents=True, exist_ok=True)
    report = export(Path(source['path']), output, 4, scale_method='mse-grid')
    selection = dict(path=str(output.resolve()), sha256=report['sha256'], precision='q4', layers=3,
                     eligible_for_promotion=False, source_float_sha256=source['sha256'],
                     weight_scale_method='Per-group minimum weight-MSE over21 BF16 scale candidates; includes original maxabs scale.',
                     bits=4, activation_bits=8, group_size=64,
                     note='Accuracy/reference packed artifact. No native packed-kernel speed claim; float parent is a development blend.')
    frozen.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(frozen, dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), local_only=True,
                             selected_before_external_scores=True, no_calibration_data_or_labels=True, models={'q4-mse': selection}))
    print('Frozen Q4 export with weight-MSE scale selection.', flush=True)


if __name__ == '__main__':
    main()
