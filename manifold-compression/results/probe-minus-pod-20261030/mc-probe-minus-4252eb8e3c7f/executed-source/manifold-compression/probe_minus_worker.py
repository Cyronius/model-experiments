"""Disposable CUDA worker for the subtracted-fold check on the healed student (heal_probe.py with
heal_probe_minus_rules.json): heal_worker's bundle and model checks, with the probe_minus/ folder in
outputs/. No RunPod credential.

The evaluate stage checks the student and refits the fold, which stays on the pod for the
generations; the 'best' generations run only if a scale earned them (heal_probe.best_scale)."""
import heal_worker

PROBE = ['heal_probe.py']
RULES = ['--rules', 'heal_probe_minus_rules.json']
STAGES = [
    ('evaluate', PROBE + ['evaluate'] + RULES, []),
    ('generate_subtract', PROBE + ['generate', '--arm', 'subtract'] + RULES, ['evaluate']),
    ('generate_best', PROBE + ['generate', '--arm', 'best'] + RULES, ['evaluate']),
]

if __name__ == '__main__':
    heal_worker.run(STAGES, 'probe_minus')
