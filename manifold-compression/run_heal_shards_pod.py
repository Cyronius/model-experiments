"""Prepare, launch, sync and terminate ONE pod that generates one healing-data shard
(heal_shards.py, heal_shards_rules.json), with run_slim_pod's launcher and its independent watchdog.

Default only packages the bundle. --execute rents the pod. On 2026-11-04 the user approved
generating healing data with the dense 9B in vLLM in time-boxed sessions; vLLM passed its proof
(vllm_proof_rules.json); on 2026-11-05 the user said 'do it' to launching shard 0 with a $5 cap
(runpod_budget_heal_shard0_20261105.json). One 48 GB GPU at most 0.60 USD/h, deleted on
completion, failure or the 3.5-hour deadline. Each later shard needs its own ledger.
"""
import argparse

import run_slim_pod as pod

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--quote', action='store_true')
    args = parser.parse_args()
    pod.JOBS = pod.REPO / 'target/manifold-heal-shards'
    pod.OUTPUTS = pod.ROOT / 'results/heal-shards-pod-20261105'
    pod.LEDGER = pod.ROOT / args.ledger
    pod.CAP = 5.0
    pod.GPUS = ('NVIDIA A40', 'NVIDIA RTX A6000')
    pod.RATE = 0.60
    # Setup and installs (about 25 min), about 2 h of generation, margin; the worker stops starting
    # chunks 20 minutes before the deadline (10 for the launcher, 10 its own).
    pod.HOURS = 3.5
    pod.JOB_CAP = (pod.RATE + pod.DISK_RATE) * pod.HOURS
    pod.PREFIX, pod.WORKER, pod.REQUIREMENTS, pod.ARCHIVE = 'mc-shard-', 'heal_shards_worker.py', 'slim_requirements.txt', 'heal-shards-bundle.tar'
    pod.CODE = ['heal_shards.py', 'heal_shards_worker.py', 'heal_shards_rules.json', 'slim_worker.py', 'slim_requirements.txt',
                f'heal_shards/problems_{args.shard:03d}.jsonl']
    pod.INPUTS = []
    if pod.JOB_CAP > pod.CAP:
        raise ValueError('One job could exceed the cap')
    if args.quote:
        print(pod.json.dumps(pod.quote()), flush=True)
    else:
        archive = pod.bundle()
        pod.execute(archive) if args.execute else print(str(archive))
