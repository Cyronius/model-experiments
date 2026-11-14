"""Freeze six local QAT-parameter blends; inference remains ternary/int8.

Average only same-lineage QAT master parameters. These are never float models.
The deployed ternary code/row-scale derivation remains unchanged.
"""
import datetime
import json
from pathlib import Path

import torch

from pair_training_state import atomic_json, atomic_torch_save, digest

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def main():
    torch.set_num_threads(2)
    output = ROOT / 'ternary-blends/results/frozen.json'
    if output.exists():
        raise RuntimeError('The grid is already frozen')
    source = json.loads((ROOT / 'science-ternary/results/frozen.json').read_text())['models']
    plan = json.loads((ROOT / 'science-ternary/runpod-job.json').read_text())
    specs = {v['id']: v for v in plan['training']['arms']}
    frozen = dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  selected_before_external_scores=True, local_only=True,
                  scope='Same-lineage QAT master-parameter interpolation; fully ternary inference. Observed development diagnostics, no paid job.',
                  rule='Both depths and fractions fixed before any blend score.', models={})
    for depth, arm in ((2, 'ternary-fresh'), (3, '3-ternary-fresh')):
        selected = source[arm]
        parent_path = PROJECT / specs[arm]['checkpoint']
        child_path = Path(selected['path'])
        assert digest(parent_path) == selected['initial_checkpoint_sha256']
        assert digest(child_path) == selected['sha256']
        parent, child = [torch.load(p, map_location='cpu', weights_only=False) for p in (parent_path, child_path)]
        assert parent['quant'] and child['quant']
        assert parent['dims'] == child['dims'] and parent['dims']['layers'] == depth
        assert parent['state'].keys() == child['state'].keys()
        for fraction in (.25, .5, .75):
            name = f'ternary-{depth}-{int(fraction * 100)}'
            path = ROOT / 'work/ternary-blends' / (name + '.pt')
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
            provenance = dict(parent_sha256=digest(parent_path), trained_sha256=selected['sha256'], trained_fraction=fraction)
            atomic_torch_save(path, dict(dims=parent['dims'], quant=True, state=state,
                                        meta=dict(local_qat_parameter_blend=provenance)))
            frozen['models'][name] = dict(path=str(path.resolve()), sha256=digest(path), precision='ternary', layers=depth,
                                          eligible_for_promotion=False, blend=provenance,
                                          eligibility_note='Development diagnostic; new held-source/NLI guards and independent confirmation not established.')
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, frozen)
    print('Frozen six ternary QAT-parameter blends.', flush=True)


if __name__ == '__main__':
    main()
