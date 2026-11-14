"""Finish the fixed scale cohort if its first bounded allocation times out.

Runs within the user's explicit cumulative $10 multi-experiment authorization.
A new $0.85 reservation is made only after the first pod is deleted and reconciled.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from campaign_budget import atomic_json, reserve, sync

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def run(args, **kwargs):
    subprocess.run(args, check=True, cwd=PROJECT, **kwargs)


def main():
    state_path = ROOT / 'work/scales-run/pod.json'
    while True:
        state = json.loads(state_path.read_text())
        if state.get('deleted') and state.get('deletion_verified_http_status') == 404 and 'estimated_total_usd' in state:
            break
        time.sleep(20)
    original = json.loads((ROOT / 'scales/runpod-job.json').read_text())
    missing = [arm for arm in original['training']['arms'] if not
               (ROOT / 'work/scales-run/backups' / ('scales-' + arm['id']) / 'completed.json').exists()]
    if not missing:
        print('All five arms finished on the first bounded pod; no extra allocation.', flush=True)
        return
    env = dict(os.environ, DIBS_AGENT='scales_finish_prepare')
    run(['dibs', 'run', 'cpu', 'build', '--label', 'Prepare verified complete-state scale continuation', '--eta', '5m', '--',
         sys.executable, str(ROOT / 'continue_scales.py'), 'prepare'], env=env)
    run(['dibs', 'run', 'cpu', 'build', '--label', 'Package unfinished scale arms and resume state', '--eta', '5m', '--',
         sys.executable, str(ROOT / 'pair_runpod.py'), 'prepare', '--profile', 'scales-rest'], env=env)
    ledger = sync()
    profile = 'scales-rest'
    directory = ROOT / 'work/scales-rest-run'
    reserve(dict(name=profile, reserved_cap_usd=.85, max_hours=.9, status='reserved', gpu='RTX4090',
                 state_path=str(directory / 'pod.json')))
    atomic_json(directory / 'approval.json', dict(
        authorized=True, cap_usd=.85, max_hours=.9, campaign_cap_usd=10., authorization=ledger['authorization'],
        scope='Finish only unfinished members of the original five-arm cohort using verified optimizer/RNG/sampler checkpoints. One Secure RTX4090 expected$0.89/hour, max$0.90/hour; expected24–48min/$0.36–$0.73, hardcap$0.85 including temporarydisk; max54min, independent53min deletion watchdog. Within explicit cumulative$10 multi-experiment authorization.'))
    print('Continuation reservation; campaign commitment:', sync()['committed_upper_bound_usd'], flush=True)
    env = dict(os.environ, DIBS_AGENT='runpod_scales_rest')
    run(['dibs', 'run', 'runpod-ternary', 'network', '--label', 'Finish original scale cohort under separate$0.85 cap', '--eta', '55m', '--',
         sys.executable, str(ROOT / 'pair_runpod.py'), 'launch', '--profile', profile], env=env)
    env = dict(os.environ, DIBS_AGENT='scales_consolidate')
    run(['dibs', 'run', 'cpu', '--label', 'Consolidate verified scale cohort after both pod deletions', '--eta', '3m', '--',
         sys.executable, str(ROOT / 'continue_scales.py'), 'consolidate'], env=env)


if __name__ == '__main__':
    main()
