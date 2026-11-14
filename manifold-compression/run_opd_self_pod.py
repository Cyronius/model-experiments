"""Prepare, launch, sync and terminate ONE pod for the on-policy distillation from the every-neuron
teacher (slim_opd.py with opd_self_rules.json), with run_opd_pod's settings and run_slim_pod's
launcher.

Default only packages the bundle. --execute rents the pod: on 2026-10-31, while the dense-teacher
run was still training, the user approved this run with the same $10 cap ("Actually start it
now"; runpod_budget_opd_self_20261031.json). One 48 GB GPU at most 0.60 USD/h, deleted on
completion, failure or the 10-hour deadline; tend_pod.py takes over if the launcher stops.
"""
import argparse

import run_opd_pod
from run_opd_pod import pod

pod.JOBS = pod.REPO / 'target/manifold-opd-self'
pod.OUTPUTS = pod.ROOT / 'results/opd-self-pod-20261031'
pod.LEDGER = pod.ROOT / 'runpod_budget_opd_self_20261031.json'
pod.PREFIX, pod.WORKER, pod.ARCHIVE = 'mc-opds-', 'opd_self_worker.py', 'opd-self-bundle.tar'
pod.CODE = run_opd_pod.pod.CODE + ['opd_self_rules.json', 'opd_self_worker.py']


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
