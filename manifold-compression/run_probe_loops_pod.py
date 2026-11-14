"""Prepare, launch, sync and terminate ONE pod of the loop checks on the healed student
(heal_probe.py with heal_probe_loops_<experiment>_rules.json), with run_slim_pod's launcher and its
independent watchdog. The six experiments run as six pods in parallel, one launcher each.

Default only packages the bundle. --execute rents the pod: on 2026-10-30, after a quarter of the
fold lowered KL and looped in 4 of 16, the user said "well keep pursuing this line and see if we can
eliminate the looping. run several experiments in parallel". Each pod has its own ledger
(runpod_budget_probe_loops_<experiment>_20261030.json) and cap; together about $6 expected and at
most $10.30. One 48 GB GPU at most 0.60 USD/h, deleted on completion, failure or the deadline.
"""
import argparse

import run_slim_pod as pod

RUN = 'results/moe-carve-v2-20261029'
# Hours: setup, the fold's fit (if any), the evaluate arms and up to 2.1 hours per 48 generations.
EXPERIMENTS = {'control': (2.4, 1.5), 'fold': (2.8, 1.75), 'order': (3.0, 1.85), 'track': (3.4, 2.1),
               'sample': (2.4, 1.5), 'sample_fold': (2.6, 1.6)}


def configure(experiment):
    pod.HOURS, pod.CAP = EXPERIMENTS[experiment]
    pod.JOBS = pod.REPO / f'target/manifold-probe-loops-{experiment}'
    pod.OUTPUTS = pod.ROOT / 'results/probe-loops-pod-20261030'
    pod.LEDGER = pod.ROOT / f'runpod_budget_probe_loops_{experiment}_20261030.json'
    # Generation here is bound by per-token overhead more than by compute, so the cheapest cards win.
    pod.GPUS = ('NVIDIA A40', 'NVIDIA RTX A6000')
    pod.RATE = 0.60
    pod.JOB_CAP = (pod.RATE + pod.DISK_RATE) * pod.HOURS
    pod.PREFIX = f'mc-loops-{experiment.replace("_", "-")}-'
    pod.WORKER, pod.REQUIREMENTS = f'probe_loops_worker.py {experiment}', 'slim_requirements.txt'
    pod.ARCHIVE = f'probe-loops-{experiment}-bundle.tar'
    pod.CODE = ['moe_carve.py', 'slim_batch.py', 'slim_heal.py', 'heal_probe.py', 'h0_mlp_probe.py', 'slim_worker.py',
                'heal_worker.py', 'probe_loops_worker.py', 'slim_requirements.txt', 'moe_carve_rules_v2.json',
                'slim_rules.json', 'heal_rules.json', f'heal_probe_loops_{experiment}_rules.json',
                'heal_probe_loops_rules.json', 'slim_prompts.json', 'probe_prompts_48.json', 'test_slim_heal.py',
                'test_heal_probe.py']
    pod.INPUTS = [f'{RUN}/{name}' for name in ('corpus.json', 'sets.pt', 'routers.pt', 'slim/vocab64k.json',
                                               'slim/deltanet.json', 'heal/heal.json', 'heal/adapter.pt')]
    if pod.JOB_CAP > pod.CAP:
        raise ValueError('One job could exceed the cap')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=EXPERIMENTS)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--quote', action='store_true')
    args = parser.parse_args()
    configure(args.experiment)
    if args.quote:
        print(pod.json.dumps(pod.quote()), flush=True)
    else:
        archive = pod.bundle()
        pod.execute(archive) if args.execute else print(str(archive))
