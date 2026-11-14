"""Prepare, launch, sync and terminate ONE pod that runs section 17's checks again with a 4,096-token
cap (slim_opd.py check with opd_cap_rules.json), with run_opd_pod's settings and run_slim_pod's launcher.

Default only packages the bundle. --execute rents the pod: on 2026-11-01, after section 17, I offered
to check whether the distilled student's sampled answers that ran to the 2,048-token cap are long or
go round in circles, and the user said "check with longer limit" (runpod_budget_opd_cap_20261101.json,
$3 cap). One 48 GB GPU at most 0.60 USD/h, deleted on completion, failure or the 4-hour deadline.
"""
import argparse

import run_opd_pod
from run_opd_pod import pod

pod.JOBS = pod.REPO / 'target/manifold-opd-cap'
pod.OUTPUTS = pod.ROOT / 'results/opd-cap-pod-20261101'
pod.LEDGER = pod.ROOT / 'runpod_budget_opd_cap_20261101.json'
pod.CAP = 3.0
# Setup, the reproduction check and six arms of 96 answers at up to 4,096 tokens (3.4 h at most).
pod.HOURS = 4
pod.JOB_CAP = (pod.RATE + pod.DISK_RATE) * pod.HOURS
pod.PREFIX, pod.WORKER, pod.ARCHIVE = 'mc-opdc-', 'opd_cap_worker.py', 'opd-cap-bundle.tar'
pod.CODE = run_opd_pod.pod.CODE + ['opd_cap_rules.json', 'opd_cap_worker.py', 'probe_prompts_96.json']
# Section 17's distilled student.
pod.INPUTS = run_opd_pod.pod.INPUTS + [f'{run_opd_pod.RUN}/{name}' for name in ('opd_long/adapter.pt', 'opd_long/train.json')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--quote', action='store_true')
    args = parser.parse_args()
    if pod.JOB_CAP > pod.CAP:
        raise ValueError('One job could exceed the cap')
    if args.quote:
        print(pod.json.dumps(pod.quote()), flush=True)
    else:
        archive = pod.bundle()
        pod.execute(archive) if args.execute else print(str(archive))
