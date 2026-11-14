"""Prepare, launch, sync and terminate ONE pod for the on-policy distillation pilot (slim_opd.py),
with run_slim_pod's launcher and its independent watchdog.

Default only packages the bundle. --execute rents the pod: on 2026-10-31, after section 14 of the
results showed no fold stops the loops, the user approved on-policy distillation at roughly $5-10
("ok do the $5-10 run"), capped at $10 (runpod_budget_opd_20261031.json). One 48 GB GPU at most
0.60 USD/h, deleted on completion, failure or the 10-hour deadline.
"""
import argparse

import run_slim_pod as pod

RUN = 'results/moe-carve-v2-20261029'
pod.JOBS = pod.REPO / 'target/manifold-opd'
pod.OUTPUTS = pod.ROOT / 'results/opd-pod-20261031'
pod.LEDGER = pod.ROOT / 'runpod_budget_opd_20261031.json'
pod.CAP = 10.0
# Most of the time goes to generating, which per-token overhead bounds more than compute, so the
# cheapest cards win.
pod.GPUS = ('NVIDIA A40', 'NVIDIA RTX A6000')
pod.RATE = 0.60
# Setup, the checks before (1.2 h at most), training (7.5 h at most, ending 50 minutes before the
# worker's deadline) and the checks after (0.7 h at most).
pod.HOURS = 10
pod.JOB_CAP = (pod.RATE + pod.DISK_RATE) * pod.HOURS
pod.PREFIX, pod.WORKER, pod.REQUIREMENTS, pod.ARCHIVE = 'mc-opd-', 'opd_worker.py', 'slim_requirements.txt', 'opd-bundle.tar'
pod.CODE = ['moe_carve.py', 'slim_batch.py', 'slim_heal.py', 'heal_probe.py', 'slim_opd.py', 'h0_mlp_probe.py',
            'slim_worker.py', 'heal_worker.py', 'opd_worker.py', 'slim_requirements.txt', 'moe_carve_rules_v2.json',
            'slim_rules.json', 'heal_rules.json', 'opd_rules.json', 'opd_prompts.json', 'probe_prompts_48.json',
            'slim_prompts.json', 'test_slim_heal.py', 'test_heal_probe.py', 'test_slim_opd.py']
pod.INPUTS = [f'{RUN}/{name}' for name in ('corpus.json', 'sets.pt', 'routers.pt', 'slim/vocab64k.json',
                                           'slim/deltanet.json', 'heal/heal.json', 'heal/adapter.pt')]


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
