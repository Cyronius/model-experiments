"""Prepare, launch, sync and terminate ONE pod for the healing pilot (slim_heal.py), with
run_slim_pod's launcher and its independent watchdog.

Default only packages the bundle. --execute rents the pod: the user approved a pilot of about 10M
tokens for roughly $3-5 on 2026-10-29 ('do it!'), capped at $5 (runpod_budget_heal_20261029.json).
One 48 GB GPU at most 1.09 USD/h, most compute per dollar first, deleted on completion, failure or
the 4.5-hour deadline.
"""
import argparse

import run_slim_pod as pod

RUN = 'results/moe-carve-v2-20261029'
pod.JOBS = pod.REPO / 'target/manifold-heal'
pod.OUTPUTS = pod.ROOT / 'results/heal-pod-20261029'
pod.LEDGER = pod.ROOT / 'runpod_budget_heal_20261029.json'
pod.CAP = 5.0
# The pilot is compute-bound: the Ada cards do about 2.4x an A40's bf16 work for under 2x the price.
pod.GPUS = ('NVIDIA RTX 6000 Ada Generation', 'NVIDIA L40S', 'NVIDIA A40', 'NVIDIA RTX A6000', 'NVIDIA L40')
pod.RATE = 1.09
pod.HOURS = 4.5
pod.JOB_CAP = (pod.RATE + pod.DISK_RATE) * pod.HOURS
pod.PREFIX, pod.WORKER, pod.REQUIREMENTS, pod.ARCHIVE = 'mc-heal-', 'heal_worker.py', 'slim_requirements.txt', 'heal-bundle.tar'
pod.CODE = ['moe_carve.py', 'slim_batch.py', 'slim_heal.py', 'h0_mlp_probe.py', 'slim_worker.py', 'heal_worker.py',
            'slim_requirements.txt', 'moe_carve_rules_v2.json', 'slim_rules.json', 'heal_rules.json', 'slim_prompts.json',
            'test_slim_heal.py']
pod.INPUTS = [f'{RUN}/{name}' for name in ('corpus.json', 'sets.pt', 'routers.pt', 'slim/vocab64k.json',
                                           'slim/deltanet.json', 'slim/combined.json', 'heal/data.npz', 'heal/data.json')]


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
