"""Prepare a small, human-contradiction mix and bounded final-job proposal.

No allocation, approval, or budget reservation. Preserve the known-positive
closure of every selected query so newly trusted negatives cannot override it.
"""
import collections
import copy
import json
import shutil
from pathlib import Path

from pair_training_state import atomic_json, digest

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def main():
    profile = 'human-mix'
    output = ROOT / 'work' / profile
    if (output / 'audit.json').exists():
        raise RuntimeError('Already prepared; preserve this proposal')
    original = ROOT / 'work/projection'
    projection_plan = json.loads((original / 'plan.json').read_text())
    for name, checksum in projection_plan['prepared_sha256'].items():
        assert digest(original / name) == checksum
    rows = [json.loads(line) for line in (original / 'train.jsonl').open()]
    expected_known = {k: set(v) for k, v in json.loads((original / 'known.json').read_text()).items()}
    query_keys = {r['query_visible_key'] for r in rows}
    pairs = {(r['query_visible_key'], r['positive_visible_key']) for r in rows}
    base = ROOT / 'work/science-mix-clean/train.jsonl'
    assert digest(base) == projection_plan['input_sha256'][str(base.relative_to(PROJECT))]
    added = 0
    for line in base.open():
        row = json.loads(line)
        pair = row['query_visible_key'], row['positive_visible_key']
        if pair[0] in query_keys and pair not in pairs:
            rows.append({k: v for k, v in row.items() if not k.startswith('negative')})
            pairs.add(pair)
            added += 1
    known = collections.defaultdict(set)
    for row in rows:
        known[row['query_visible_key']].add(row['positive_visible_key'])
    assert dict(known) == expected_known
    rejected = 0
    for row in rows:
        if row.get('negative') and row['negative_visible_key'] in known[row['query_visible_key']]:
            for key in list(row):
                if key.startswith('negative'):
                    del row[key]
            rejected += 1
    dev = [json.loads(line) for line in (original / 'dev.jsonl').open()]
    def identities(data):
        return {r[k] for r in data for k in ('query_visible_key', 'positive_visible_key', 'negative_visible_key') if r.get(k)}
    assert not identities(rows) & identities(dev)
    assert all(not r.get('negative') or r['negative_source'] == 'human same-premise contradiction' for r in rows)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'train.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    shutil.copy2(original / 'dev.jsonl', output / 'dev.jsonl')
    audit = dict(prepared_only=True, paid_resource_allocated=False, teacher_targets=False,
                 source_projection_plan_sha256=digest(original / 'plan.json'),
                 selected_rows_before_positive_closure=projection_plan['train_rows'],
                 added_known_positive_pairs=added, ambiguous_negative_attachments_removed=rejected,
                 rows=dict(train=len(rows), dev=len(dev)), by_source=dict(collections.Counter(r['source'] for r in rows)),
                 human_negative_rows=sum(bool(r.get('negative')) for r in rows),
                 output_sha256={split: digest(output / (split + '.jsonl')) for split in ('train', 'dev')},
                 all_known_positives_of_selected_queries_preserved=True, train_dev_visible_overlap=0,
                 reserved_judgments_read=False, benchmark_exclusions='Inherited clean inputs, including observed-task text exclusions.')
    atomic_json(output / 'audit.json', audit)
    starts = output / 'starts'
    starts.mkdir(exist_ok=True)
    provenance = {}
    for precision, parent_profile, arm in (('float', 'float-blends', 'float-25'), ('ternary', 'ternary-blends', 'ternary-3-25')):
        summary = ROOT / parent_profile / 'results/summary.json'
        model = json.loads(summary.read_text())['models'][arm]
        selection = model['selection']
        assert selection['precision'] == precision and selection['layers'] == 3
        source = Path(selection['path'])
        assert digest(source) == selection['sha256']
        target = starts / (precision + '.pt')
        shutil.copy2(source, target)
        provenance[precision] = dict(selection=selection, source_summary=str(summary.resolve()), source_arm=arm,
                                     source_summary_sha256=digest(summary), prepared_path=str(target.relative_to(PROJECT)))
    plan = copy.deepcopy(json.loads((ROOT / 'scales/runpod-job.json').read_text()))
    plan.update(profile=profile, status='Prepared only; choose after existing results, then reserve within remaining approved$10 before allocation',
                maximum_elapsed_hours=.65, proposed_total_cap_usd=.60,
                expected_elapsed_hours=[.33, .5], expected_total_usd=[.30, .45],
                work_duration_minutes=32, work_deadline_minutes=35, watchdog_delete_minutes=38)
    plan['training'] = dict(
        arms=[dict(id=name, checkpoint=provenance[precision]['prepared_path'], layers=3, batch=512, micro_batch=128,
                   lr=lr, use_negatives=True) for name, precision, lr in
              [('ternary-low-lr', 'ternary', 3e-6), ('ternary', 'ternary', 1e-5), ('float', 'float', 1e-5)]],
        epochs_each=3, seed=57, eval_every=150, save_every=100, source_floor=2.,
        validation_negatives=True, teacher_targets=False, auxiliary_nli=False,
        train_rows=len(rows), dev_rows=len(dev), data_sha256=audit['output_sha256']['train'],
        selection='Highest held-source macro nDCG across all9 sources; every source within2 points of initialization. Human contradictions included in held NLI ranking. Freeze all3 before external scoring.',
        comparison='Whole-model continuation on clean relevance plus human entailment/contradiction. Two ternary rates and one independent float branch; source proportions differ from the large science mixture. Frozen-output-projection experiments are context, not a clean same-learning-rate ablation.',
        direct_nli='Human entailment positives and same-premise human contradictions in contrastive ranking; no disposable classifier or teacher.')
    plan['data_inputs'] = {split: dict(path=str((output / (split + '.jsonl')).relative_to(PROJECT)), sha256=audit['output_sha256'][split]) for split in ('train', 'dev')}
    plan['data_audit_files'] = [str((output / 'audit.json').relative_to(PROJECT)), 'training-v2/work/projection/plan.json', 'training-v2/work/human-triplets/audit.json']
    directory = ROOT / profile
    directory.mkdir(exist_ok=True)
    atomic_json(directory / 'parent-selection.json', provenance)
    atomic_json(directory / 'runpod-job.json', plan)
    assert (plan['maximum_gpu_rate_usd_per_hour'] + plan['disk_gb'] * .1 / 720) * plan['maximum_elapsed_hours'] < plan['proposed_total_cap_usd']
    print(json.dumps(dict(audit=audit, cap_usd=.6, maximum_hours=.65, allocated=False), indent=2), flush=True)


if __name__ == '__main__':
    main()
