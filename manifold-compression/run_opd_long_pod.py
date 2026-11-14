"""Prepare, launch, sync and terminate ONE pod for the every-neuron-teacher distillation carried on
from section 16's adapter, with healing text (slim_opd.py with opd_long_rules.json), with
run_opd_pod's settings and run_slim_pod's launcher.

Default only packages the bundle. --execute rents the pod: on 2026-10-31, after section 16, the user
approved carrying that run on for 12 more hours with healing text, checked on 96 prompts, about $8
with a $10 cap and tend_pod.py's timer as a backup ("yes to all of it";
runpod_budget_opd_long_20261031.json). One 48 GB GPU at most 0.60 USD/h, deleted on completion,
failure or the 15.5-hour deadline.
"""
import argparse

import run_opd_pod
from run_opd_pod import pod

pod.JOBS = pod.REPO / 'target/manifold-opd-long'
pod.OUTPUTS = pod.ROOT / 'results/opd-long-pod-20261031'
pod.LEDGER = pod.ROOT / 'runpod_budget_opd_long_20261031.json'
# Setup, the checks before (1.5 h at most), training (12 h at most, ending 75 minutes before the
# worker's deadline) and the checks after (1 h at most).
pod.HOURS = 15.5
pod.JOB_CAP = (pod.RATE + pod.DISK_RATE) * pod.HOURS
pod.PREFIX, pod.WORKER, pod.ARCHIVE = 'mc-opdl-', 'opd_long_worker.py', 'opd-long-bundle.tar'
pod.CODE = run_opd_pod.pod.CODE + ['opd_long_rules.json', 'opd_long_worker.py', 'probe_prompts_96.json']
# The healing text, and the adapter this run starts from.
pod.INPUTS = run_opd_pod.pod.INPUTS + [f'{run_opd_pod.RUN}/{name}' for name in (
    'heal/data.json', 'heal/data.npz', 'opd_self/adapter.pt', 'opd_self/train.json')]


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
