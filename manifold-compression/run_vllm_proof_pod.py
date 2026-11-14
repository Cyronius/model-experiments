"""Prepare, launch, sync and terminate ONE pod for the vLLM proof (vllm_proof.py, vllm_proof_rules.json),
with run_slim_pod's launcher and its independent watchdog.

Default only packages the bundle. --execute rents the pod: on 2026-11-04 the user approved the plan
to generate healing data with the dense 9B in vLLM in time-boxed sessions (first pod job about $5)
and added "we have to prove the vllm as well" (runpod_budget_vllm_proof_20261104.json, $5 cap). One
48 GB GPU at most 0.60 USD/h, deleted on completion, failure or the 3-hour deadline.
"""
import argparse

import run_slim_pod as pod

RUN = 'results/moe-carve-v2-20261029'
pod.JOBS = pod.REPO / 'target/manifold-vllm-proof'
pod.OUTPUTS = pod.ROOT / 'results/vllm-proof-pod-20261104'
pod.LEDGER = pod.ROOT / 'runpod_budget_vllm_proof_20261104.json'
pod.CAP = 5.0
pod.GPUS = ('NVIDIA A40', 'NVIDIA RTX A6000')
pod.RATE = 0.60
# Setup and vLLM install (about 20 min), the two check parts (about 15 min), 40 minutes of
# throughput, the HF comparison (about 20 min), with margin.
pod.HOURS = 3
pod.JOB_CAP = (pod.RATE + pod.DISK_RATE) * pod.HOURS
pod.PREFIX, pod.WORKER, pod.REQUIREMENTS, pod.ARCHIVE = 'mc-vllm-', 'vllm_proof_worker.py', 'slim_requirements.txt', 'vllm-proof-bundle.tar'
pod.CODE = ['vllm_proof.py', 'vllm_proof_worker.py', 'vllm_proof_rules.json', 'vllm_proof_problems.jsonl',
            'probe_prompts_96.json', 'geometry_check_rules.json', 'slim_worker.py', 'slim_requirements.txt',
            'moe_carve.py', 'slim_batch.py', 'h0_mlp_probe.py', 'heal_probe.py', 'slim_heal.py', 'slim_opd.py',
            'heal_geometry.py', 'moe_carve_rules_v2.json', 'slim_rules.json', 'heal_rules.json']
pod.INPUTS = [f'{RUN}/slim/vocab64k.json']


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
