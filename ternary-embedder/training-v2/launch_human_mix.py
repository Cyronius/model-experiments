"""Run the prepared focused-data job after the calibration continuation closes.

Uses the explicit cumulative $10 multi-experiment authorization; refuses to
allocate if the new bounded reservation does not fit the remaining campaign.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from campaign_budget import atomic_json, reserve, sync
from pair_training_state import digest

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def main():
    previous = ROOT / 'work/scales-rest-run'
    while True:
        path = previous / 'pod.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        if state.get('deleted') and state.get('deletion_verified_http_status') == 404 and 'estimated_total_usd' in state:
            break
        time.sleep(20)
    continuation = json.loads((ROOT / 'scales-rest/runpod-job.json').read_text())
    for arm in continuation['training']['arms']:
        run = previous / 'backups' / ('scales-' + arm['id'])
        completed = json.loads((run / 'completed.json').read_text())
        config = json.loads((run / 'config.json').read_text())
        assert completed['step'] == config['total_steps'] == 1004
    profile = 'human-mix'
    directory = ROOT / 'work/human-mix-run'
    plan_path = ROOT / profile / 'runpod-job.json'
    plan = json.loads(plan_path.read_text())
    package = json.loads((directory / 'package.json').read_text())
    assert package['profile'] == profile and package['plan_sha256'] == digest(plan_path)
    assert plan['proposed_total_cap_usd'] == .6 and plan['maximum_elapsed_hours'] == .65
    assert plan['training']['teacher_targets'] is False and len(plan['training']['arms']) == 3
    assert not (directory / 'pod.json').exists()
    ledger = sync()
    reserve(dict(name=profile, reserved_cap_usd=.6, max_hours=.65, status='reserved', gpu='RTX4090',
                 state_path=str(directory / 'pod.json')))
    atomic_json(directory / 'approval.json', dict(
        authorized=True, cap_usd=.6, max_hours=.65, campaign_cap_usd=10., authorization=ledger['authorization'],
        scope='Three fixed full-model continuation arms from independent float and ternary blends, on audited clean relevance plus human same-premise contradictions. One Secure RTX4090 expected$0.89/hour, max$0.90/hour; expected20–30min/$0.30–$0.45, hardcap$0.60 including temporarydisk; maximum39min, independent38min watchdog. Within explicit cumulative$10 experiment authorization.'))
    atomic_json(ROOT / profile / 'launch-decision.json', dict(
        reason='Frozen output-projection training did not improve the stronger ternary blend; test trusted human contradictions with whole-model updates while preserving known positives and nine held-source floors.',
        parents='Fixed float25% and ternary3layer25% blends; no calibration-cohort external result selects these starts.',
        campaign_commitment_upper_usd=sync()['committed_upper_bound_usd'],
        plan_sha256=digest(plan_path), package_sha256=package['sha256']))
    print('Human-mix reservation; full campaign commitment:', sync()['committed_upper_bound_usd'], flush=True)
    env = dict(os.environ, DIBS_AGENT='runpod_human_mix')
    subprocess.run(['dibs', 'run', 'runpod-ternary', 'network', '--label', 'Focused human-label full-model experiment under$0.60 cap',
                    '--eta', '40m', '--', sys.executable, str(ROOT / 'pair_runpod.py'), 'launch', '--profile', profile],
                   check=True, cwd=PROJECT, env=env)
    for arm in plan['training']['arms']:
        run = directory / 'backups' / (profile + '-' + arm['id'])
        completion = json.loads((run / 'completed.json').read_text())
        config = json.loads((run / 'config.json').read_text())
        assert completion['step'] == config['total_steps']
    print('All three focused-data arms completed; paid pod deleted.', flush=True)


if __name__ == '__main__':
    main()
