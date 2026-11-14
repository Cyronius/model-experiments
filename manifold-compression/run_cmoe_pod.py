"""Prepare, launch, sync and terminate ONE pod for the CMoE router comparison (cmoe_router.py),
with run_slim_pod's launcher and its independent watchdog.

Default only packages the bundle. --execute rents the pod: on 2026-10-29 the user asked to "try
cmoe's router as you suggested for comparison"; the job is capped at $1.50
(runpod_budget_cmoe_20261029.json). One 48 GB GPU at most 1.09 USD/h, cheapest first, deleted on
completion, failure or the 1-hour deadline.
"""
import argparse

import run_slim_pod as pod

RUN = 'results/moe-carve-v2-20261029'
pod.JOBS = pod.REPO / 'target/manifold-cmoe'
pod.OUTPUTS = pod.ROOT / 'results/cmoe-pod-20261029'
pod.LEDGER = pod.ROOT / 'runpod_budget_cmoe_20261029.json'
pod.CAP = 1.5
pod.GPUS = ('NVIDIA A40', 'NVIDIA RTX A6000', 'NVIDIA L40', 'NVIDIA RTX 6000 Ada Generation', 'NVIDIA L40S')
pod.RATE = 1.09
pod.HOURS = 1.0
pod.JOB_CAP = (pod.RATE + pod.DISK_RATE) * pod.HOURS
pod.PREFIX, pod.WORKER, pod.REQUIREMENTS, pod.ARCHIVE = 'mc-cmoe-', 'cmoe_worker.py', 'slim_requirements.txt', 'cmoe-bundle.tar'
pod.CODE = ['moe_carve.py', 'cmoe_router.py', 'h0_mlp_probe.py', 'slim_worker.py', 'heal_worker.py', 'cmoe_worker.py',
            'slim_requirements.txt', 'moe_carve_rules_v2.json', 'test_cmoe_router.py']
pod.INPUTS = [f'{RUN}/{name}' for name in ('corpus.json', 'sets.pt', 'routers.pt', 'cmoe/representatives.pt',
                                           'cmoe/pick.json')]


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
