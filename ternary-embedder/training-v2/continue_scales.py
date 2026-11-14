"""Prepare or consolidate an exactly resumed, separately bounded scale job.

No resource allocation or reservation occurs here. The original five-arm plan,
training settings, seeds, and global freeze remain unchanged.
"""
import argparse
import copy
import json
import shutil
from pathlib import Path

from pair_training_state import atomic_json, digest

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ORIGINAL = ROOT / 'work/scales-run'
REST = ROOT / 'work/scales-rest-run'


def closed(directory):
    path = directory / 'pod.json'
    state = json.loads(path.read_text())
    assert state['deleted'] and state['deletion_verified_http_status'] == 404
    return state


def prepare():
    closed(ORIGINAL)
    if (REST / 'pod.json').exists() or (ROOT / 'scales-rest/runpod-job.json').exists():
        raise RuntimeError('Do not reuse or replace a prepared continuation')
    if (ROOT / 'scales/results/frozen.json').exists():
        raise RuntimeError('No continuation after external candidate freezing')
    plan = json.loads((ROOT / 'scales/runpod-job.json').read_text())
    remaining = [v for v in plan['training']['arms']
                 if not (ORIGINAL / 'backups' / ('scales-' + v['id']) / 'completed.json').exists()]
    if not remaining:
        print('All original arms completed; no continuation needed.', flush=True)
        return
    original_manifest = json.loads((ORIGINAL / 'package-manifest.json').read_text())
    for name in ('pair_train.py', 'pair_validation.py', 'pair_training_state.py',
                 'contrastive_cache.py', 'learned_row_scales.py', 'nli_auxiliary.py'):
        assert digest(ROOT / name) == original_manifest['training-v2/' + name]
    assert digest(PROJECT / 'tern-embed/model.py') == original_manifest['tern-embed/model.py']
    receipt = json.loads((ORIGINAL / 'backups/verified.json').read_text())
    plan = copy.deepcopy(plan)
    plan.update(profile='scales-rest', run_namespace='scales',
                status='Prepared only; separately reserve within the approved cumulative $10 campaign before allocation',
                proposed_total_cap_usd=.85, maximum_elapsed_hours=.9,
                expected_elapsed_hours=[.4, .8], expected_total_usd=[.36, .73],
                work_duration_minutes=46, work_deadline_minutes=50, watchdog_delete_minutes=53,
                continuation_of='scales', original_plan_sha256=digest(ROOT / 'scales/runpod-job.json'),
                resume_inputs=[])
    plan['training']['arms'] = remaining
    starts = ROOT / 'work/scales-rest/starts'
    starts.mkdir(parents=True, exist_ok=True)
    for arm in remaining:
        old_start = PROJECT / arm['checkpoint']
        new_start = starts / old_start.name
        shutil.copy2(old_start, new_start)
        assert digest(new_start) == digest(old_start)
        arm['checkpoint'] = str(new_start.relative_to(PROJECT))
        run = 'scales-' + arm['id']
        previous = ORIGINAL / 'backups' / run
        if not (previous / 'latest.json').exists():
            continue
        latest = json.loads((previous / 'latest.json').read_text())
        required = {latest['file'], 'latest.json', 'config.json'}
        if not required.issubset({p.name for p in previous.iterdir()}):
            raise RuntimeError('Incomplete resume inputs')
        local_run = ROOT / 'runs' / run
        local_run.mkdir(parents=True, exist_ok=True)
        files = required | {name for name in ('best.pt', 'log.jsonl', 'validation.jsonl') if (previous / name).exists()}
        for name in sorted(files):
            source = previous / name
            checksum = digest(source)
            assert receipt[run + '/' + name]['sha256'] == checksum
            target = local_run / name
            if target.exists() and digest(target) != checksum:
                raise RuntimeError('Conflicting local training output; refuse overwrite')
            shutil.copy2(source, target)
            assert digest(target) == checksum
            plan['resume_inputs'].append(dict(path=str(target.relative_to(PROJECT)), sha256=checksum))
        assert digest(local_run / latest['file']) == latest['sha256']
    assert (plan['maximum_gpu_rate_usd_per_hour'] + plan['disk_gb'] * .1 / 720) * plan['maximum_elapsed_hours'] < plan['proposed_total_cap_usd']
    directory = ROOT / 'scales-rest'
    directory.mkdir(exist_ok=True)
    atomic_json(directory / 'runpod-job.json', plan)
    print(json.dumps(dict(remaining_arms=[v['id'] for v in remaining], cap_usd=.85,
                          maximum_hours=.9, paid_resource_allocated=False), indent=2), flush=True)


def consolidate():
    closed(ORIGINAL)
    closed(REST)
    plan = json.loads((ROOT / 'scales-rest/runpod-job.json').read_text())
    assert plan['original_plan_sha256'] == digest(ROOT / 'scales/runpod-job.json')
    source_receipt = json.loads((REST / 'backups/verified.json').read_text())
    target_receipt = json.loads((ORIGINAL / 'backups/verified.json').read_text())
    # Verify all completion markers before publishing any of them to the original
    # cohort; the scoring waiter cannot freeze a partially consolidated cohort.
    for arm in plan['training']['arms']:
        run = REST / 'backups' / ('scales-' + arm['id'])
        config = json.loads((run / 'config.json').read_text())
        completion = json.loads((run / 'completed.json').read_text())
        assert completion['step'] == config['total_steps'] == 1004
    pending = []
    for arm in plan['training']['arms']:
        name = 'scales-' + arm['id']
        source = REST / 'backups' / name
        target = ORIGINAL / 'backups' / name
        target.mkdir(parents=True, exist_ok=True)
        for file in source.iterdir():
            if not file.is_file() or file.suffix not in ('.json', '.jsonl', '.pt'):
                continue
            relative = name + '/' + file.name
            checksum = digest(file)
            assert source_receipt[relative]['sha256'] == checksum
            temp = target / (file.name + '.consolidating')
            shutil.copy2(file, temp)
            assert digest(temp) == checksum
            if file.name == 'completed.json':
                pending.append((temp, target / file.name))
            else:
                temp.replace(target / file.name)
            target_receipt[relative] = dict(source_receipt[relative], consolidated_from='scales-rest')
    atomic_json(ORIGINAL / 'backups/verified.json', target_receipt)
    atomic_json(ORIGINAL / 'continuations.json', [dict(state_path=str((REST / 'pod.json').resolve()),
                                                     state_sha256=digest(REST / 'pod.json'),
                                                     original_plan_sha256=plan['original_plan_sha256'])])
    for source, target in pending:
        source.replace(target)
    print('All five original arms complete; both pods closed; continuation provenance published.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'consolidate'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else consolidate()
