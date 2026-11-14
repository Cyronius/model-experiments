"""Disposable CUDA worker for the soft-edge check on the healed student (heal_probe.py with
heal_probe_soft_rules.json): heal_worker's bundle and model checks, with the probe_soft/ folder in
outputs/. No RunPod credential.

The evaluate stage checks the student and works out the soft-edge widths, so the others run after it;
the generations, the main question, run before the state check."""
import heal_worker

PROBE = ['heal_probe.py']
RULES = ['--rules', 'heal_probe_soft_rules.json']
STAGES = [
    ('evaluate', PROBE + ['evaluate'] + RULES, []),
    ('generate_soft', PROBE + ['generate', '--arm', 'soft'] + RULES, ['evaluate']),
    ('state', PROBE + ['state'] + RULES, ['evaluate']),
]

if __name__ == '__main__':
    heal_worker.run(STAGES, 'probe_soft')
