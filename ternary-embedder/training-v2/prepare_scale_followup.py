"""Free/local preparation of five fixed calibration/continuation candidates.

No allocation, budget reservation or approval is created. Both depths remain
ternary; learned scales are folded into the existing BF16 row-scale format.
"""
import copy
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if (ROOT / 'work/scales-run/pod.json').exists():
        raise RuntimeError('An allocated bounded job cannot be reconfigured')
    reports = {p: json.loads((ROOT / p / 'results/summary.json').read_text())
               for p in ('science-float', 'science-ternary')}
    assert sum(len(v['models']) for v in reports.values()) == 6
    plan = copy.deepcopy(json.loads((ROOT / 'science-ternary/runpod-job.json').read_text()))
    plan.update(profile='scales', status='Prepared only; reserve and record campaign-scoped approval before allocation',
                maximum_elapsed_hours=1.6, proposed_total_cap_usd=1.5,
                expected_elapsed_hours=[1.0, 1.35], expected_total_usd=[.90, 1.22],
                work_deadline_minutes=80, watchdog_delete_minutes=90)
    plan.pop('parallel_campaign', None)
    plan['data_inputs'] = {k: v for k, v in plan['data_inputs'].items()
                           if k in ('train', 'dev', 'aux_train', 'aux_dev')}
    t = plan['training']
    t.update(epochs_each=1, seed=43, eval_every=200, save_every=100,
             aux_lr_multiplier=100., aux_batch=192, aux_warmup_steps=100,
             comparison='T2 fixed versus learned row scales; T3 learned scales with all masters trainable versus frozen; F3 low-rate continuation. All five use the same fresh science mixture, one full epoch, and the same NLI data order.',
             selection='Highest mean(held-source macro nDCG, held-NLI classifier macro accuracy), every retrieval source within2 points and NLI accuracy within1 of post-head-warmup initialization. Null encoder control eligible. Freeze all five before external scoring.',
             parent_selection='Fixed eligible science-fresh checkpoints selected before this followup. Different historical pretraining prevents a pure causal depth comparison.',
             global_freeze='All five completed arms and deleted pod before any new external scoring.',
             aux_data_exposure='1004 updates x192 =192768 encoder-supervised pairs. Head warmup uses another19200; no claim of full300k encoder coverage in this stage.',
             template_note='Final fixed followup: encoderLR1e-5, scaleLR1e-3, headLR1e-3. New sparse boolean masks preserve the existing known-positive definition; CUDA preflight required.')
    parents = {
        '2-ternary': ('science-ternary', 'ternary-fresh'),
        '3-ternary': ('science-ternary', '3-ternary-fresh'),
        '3-float': ('science-float', '3-float-fresh'),
    }
    starts = ROOT / 'work/scales/starts'
    starts.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name, (profile, arm) in parents.items():
        selection = reports[profile]['models'][arm]['selection']
        assert selection['eligible_for_promotion']
        assert selection['precision'] == ('float' if name.endswith('float') else 'ternary')
        assert selection['layers'] == int(name[0])
        source = Path(selection['path'])
        assert digest(source) == selection['sha256']
        dest = starts / (name + '.pt')
        shutil.copy2(source, dest)
        assert digest(dest) == selection['sha256']
        provenance[name] = dict(selection=selection, source_summary=str((ROOT / profile / 'results/summary.json').resolve()),
                                source_arm=arm, source_summary_sha256=digest(ROOT / profile / 'results/summary.json'),
                                prepared_path=str(dest.relative_to(PROJECT)))
    specs = [('fixed', '2-ternary', False, False), ('learned', '2-ternary', True, False),
             ('3-learned', '3-ternary', True, False), ('3-scales-only', '3-ternary', True, True),
             ('float-low-lr', '3-float', False, False)]
    arms = []
    for name, parent, learned, only in specs:
        arm = dict(id=name, checkpoint=provenance[parent]['prepared_path'], layers=int(parent[0]),
                   batch=1024, micro_batch=128, lr=1e-5, use_negatives=False,
                   aux_encoder_weight=.1 if parent.endswith('float') else .25)
        if learned:
            arm.update(learn_row_scales=True, scale_lr_multiplier=100.)
        if only:
            arm['train_row_scales_only'] = True
        arms.append(arm)
    t['arms'] = arms
    plan['recovery']['remote_save_every_steps'] = 100
    assert (plan['maximum_gpu_rate_usd_per_hour'] + plan['disk_gb'] * .1 / 720) * plan['maximum_elapsed_hours'] < plan['proposed_total_cap_usd']
    (ROOT / 'scales/parent-selection.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (ROOT / 'scales/runpod-job.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(arms=[v['id'] for v in arms], cap_usd=1.5, max_hours=1.6,
                         train_rows=t['train_rows'], epochs=1, allocated=False), indent=2))


if __name__ == '__main__':
    main()
