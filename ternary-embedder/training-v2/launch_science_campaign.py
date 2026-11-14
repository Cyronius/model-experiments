"""Launch the two separately bounded, approved science jobs concurrently.

Run under dibs run runpod-ternary network. Each child owns its watchdog and
deletes only its own pod. This wrapper never kills a billing controller.
"""
import contextlib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from campaign_budget import guard

ROOT = Path(__file__).resolve().parent
PROFILES = ('science-float', 'science-ternary')


def main():
    # Validate the entire allocation before starting either paid action.
    combined_cap = 0.
    for profile in PROFILES:
        work = ROOT / 'work' / (profile + '-run')
        if (work / 'pod.json').exists():
            raise RuntimeError('A bounded job cannot be reused: ' + profile)
        plan = json.loads((ROOT / profile / 'runpod-job.json').read_text())
        approval = json.loads((work / 'approval.json').read_text())
        if not approval.get('authorized') or approval.get('consumed'):
            raise RuntimeError('Missing or consumed approval: ' + profile)
        if approval['cap_usd'] != plan['proposed_total_cap_usd'] or approval['max_hours'] != plan['maximum_elapsed_hours']:
            raise RuntimeError('Approval bounds differ: ' + profile)
        guard(profile, plan, approval)
        if not (work / 'package.json').exists():
            raise RuntimeError('Missing prepared package: ' + profile)
        package = json.loads((work / 'package.json').read_text())
        plan_hash = hashlib.sha256((ROOT / profile / 'runpod-job.json').read_bytes()).hexdigest()
        if package.get('profile') != profile or package.get('plan_sha256') != plan_hash:
            raise RuntimeError('Prepared package plan is stale: ' + profile)
        combined_cap += plan['proposed_total_cap_usd']
    if combined_cap > 3.25 + 1e-9:
        raise RuntimeError('The bounded science comparison exceeds its combined$3.25 cap')
    children = []
    with contextlib.ExitStack() as stack:
        try:
            for profile in PROFILES:
                log = stack.enter_context((ROOT / 'work' / (profile + '-run') / 'controller.log').open('a'))
                child = subprocess.Popen([sys.executable, str(ROOT / 'pair_runpod.py'), 'launch', '--profile', profile],
                                         stdout=log, stderr=subprocess.STDOUT)
                children.append((profile, child))
                print('Started controller', profile, 'PID', child.pid, flush=True)
        finally:
            codes = {profile: child.wait() for profile, child in children}
            print('Controllers complete', codes, flush=True)
    if any(codes.values()):
        raise RuntimeError('One or more science controllers failed; inspect owned states and recovery logs')


if __name__ == '__main__':
    main()
