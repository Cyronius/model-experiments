"""Prepare, launch, sync and terminate ONE pod that carries run_opd_long_pod.py's job on from its last
checkpoint (opd_long_resume_worker.py: slim_opd.py train --resume, then after, with opd_long_rules.json),
with run_opd_long_pod's settings and run_slim_pod's launcher.

The first pod (mc-opdl-687d86dea437, $4.53) trained 22 rounds, then ran out of memory in the halfway
check: it generated all 96 check prompts at once beside the optimizer, and retried smaller batches
inside the except clause, whose traceback still held the failed attempt's cache, until one prompt
failed too. slim_opd.py now generates that check a round's worth of prompts at a time, splits a
batch only once the failed attempt is freed, and uses expandable allocator segments. This pod starts
from the checkpoint after 21 rounds (results/moe-carve-v2-20261029/opd_long/, the first pod's
outputs) with the rest of the 12-hour budget, about 6.2 hours, then runs the checks after; the
checks before are the first pod's. runpod_budget_opd_long_20261031.json allows a retry while both
pods together stay within its $10 cap. Default only packages the bundle; --execute rents the pod.
"""
import argparse

import run_opd_long_pod
from run_opd_long_pod import pod

# Setup, the halfway check and the rest of training (6.2 hours), the checks after (1 hour at most) and
# margin. With the first pod's $4.53, and its disk, this keeps both under the $10 cap.
pod.HOURS = 8.6
pod.JOB_CAP = (pod.RATE + pod.DISK_RATE) * pod.HOURS
pod.WORKER, pod.ARCHIVE = 'opd_long_resume_worker.py', 'opd-long-resume-bundle.tar'
pod.CODE = pod.CODE + ['opd_long_resume_worker.py']
pod.INPUTS = pod.INPUTS + [f'{run_opd_long_pod.run_opd_pod.RUN}/opd_long/{name}' for name in (
    'before.json', 'eval_healed.jsonl', 'eval_start.jsonl', 'gen_healed_greedy.jsonl', 'gen_healed_sampled.jsonl',
    'gen_start_greedy.jsonl', 'gen_start_sampled.jsonl', 'progress.json', 'rollouts.jsonl', 'latest.pt')]


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
