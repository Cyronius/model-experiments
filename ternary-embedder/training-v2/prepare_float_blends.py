"""Freeze a small local float-weight interpolation grid before quality scoring.

Inspired by weight-space averaging, https://arxiv.org/abs/2203.05482 .
This is a development diagnostic, not a trained or independently confirmed model.
"""
import datetime
import json
from pathlib import Path

import torch

from pair_training_state import atomic_json, atomic_torch_save, digest

ROOT = Path(__file__).resolve().parent


def main():
    out = ROOT / 'float-blends/results'
    freeze = out / 'frozen.json'
    if freeze.exists():
        raise RuntimeError('The blend grid is already frozen')
    torch.set_num_threads(2)
    original = json.loads((ROOT / 'recovered-depth/results/frozen.json').read_text())['models']['float']
    trained = json.loads((ROOT / 'science-float/results/frozen.json').read_text())['models']['3-float-fresh']
    sources = [original, trained]
    for source in sources:
        assert digest(Path(source['path'])) == source['sha256']
    parent, child = [torch.load(v['path'], map_location='cpu', weights_only=False) for v in sources]
    assert not parent['quant'] and not child['quant']
    assert parent['dims'] == child['dims'] and parent['dims']['layers'] == 3
    assert parent['state'].keys() == child['state'].keys()
    out.mkdir(parents=True, exist_ok=True)
    result = dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  selected_before_external_scores=True, local_only=True,
                  scope='Three fixed float weight blends; observed development diagnostics. No new paid job.',
                  rule='Fractions fixed before any blend score; no within-grid checkpoint training or selection.', models={})
    for fraction in (.25, .5, .75):
        name = f'float-{int(fraction * 100)}'
        path = ROOT / 'work/float-blends' / (name + '.pt')
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {}
        for key, value in parent['state'].items():
            other = child['state'][key]
            assert value.shape == other.shape and value.dtype == other.dtype
            if value.is_floating_point():
                state[key] = value.lerp(other, fraction)
            else:
                assert torch.equal(value, other)
                state[key] = value.clone()
        provenance = dict(parent_sha256=original['sha256'], trained_sha256=trained['sha256'], trained_fraction=fraction)
        atomic_torch_save(path, dict(dims=parent['dims'], quant=False, state=state,
                                    meta=dict(local_weight_blend=provenance)))
        result['models'][name] = dict(path=str(path.resolve()), sha256=digest(path), precision='float', layers=3,
                                    eligible_for_promotion=False, blend=provenance,
                                    eligibility_note='Diagnostic blend; source/NLI training-selection guards and seed confirmation have not been established.')
    atomic_json(freeze, result)
    print('Frozen local float blends:', ', '.join(result['models']), flush=True)


if __name__ == '__main__':
    main()
