"""Disposable CUDA worker for the fold check on the healed student (heal_probe.py with
heal_probe_fold_rules.json): heal_worker's bundle and model checks, with the probe_fold/ folder in
outputs/. No RunPod credential.

The evaluate stage checks the student and fits the folds, which stay on the pod for the generations."""
import heal_worker

PROBE = ['heal_probe.py']
RULES = ['--rules', 'heal_probe_fold_rules.json']
STAGES = [
    ('evaluate', PROBE + ['evaluate'] + RULES, []),
    ('generate_fold', PROBE + ['generate', '--arm', 'fold'] + RULES, ['evaluate']),
]

if __name__ == '__main__':
    heal_worker.run(STAGES, 'probe_fold')
