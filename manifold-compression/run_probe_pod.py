"""Prepare, launch, sync and terminate ONE pod for the checks on the healed student (heal_probe.py),
with run_slim_pod's launcher and its independent watchdog.

Default only packages the bundle. --execute rents the pod: on 2026-10-30 the user approved ("do it")
about $1-1.50 for an fp32-state rerun of the GSM8K generations and a perfect-router evaluation of
the healed student; the job is capped at $1.50 (runpod_budget_probe_20261030.json). One 48 GB GPU
at most 0.60 USD/h, deleted on completion, failure or the 2.4-hour deadline.
"""
import argparse

import run_slim_pod as pod

RUN = 'results/moe-carve-v2-20261029'
pod.JOBS = pod.REPO / 'target/manifold-probe'
pod.OUTPUTS = pod.ROOT / 'results/probe-pod-20261030'
pod.LEDGER = pod.ROOT / 'runpod_budget_probe_20261030.json'
pod.CAP = 1.5
# Generation here is bound by per-token overhead more than by compute, so the cheapest cards win.
pod.GPUS = ('NVIDIA A40', 'NVIDIA RTX A6000')
pod.RATE = 0.60
pod.HOURS = 2.4
pod.JOB_CAP = (pod.RATE + pod.DISK_RATE) * pod.HOURS
pod.PREFIX, pod.WORKER, pod.REQUIREMENTS, pod.ARCHIVE = 'mc-probe-', 'probe_worker.py', 'slim_requirements.txt', 'probe-bundle.tar'
pod.CODE = ['moe_carve.py', 'slim_batch.py', 'slim_heal.py', 'heal_probe.py', 'h0_mlp_probe.py', 'slim_worker.py',
            'heal_worker.py', 'probe_worker.py', 'slim_requirements.txt', 'moe_carve_rules_v2.json', 'slim_rules.json',
            'heal_rules.json', 'heal_probe_rules.json', 'slim_prompts.json', 'test_slim_heal.py', 'test_heal_probe.py']
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
