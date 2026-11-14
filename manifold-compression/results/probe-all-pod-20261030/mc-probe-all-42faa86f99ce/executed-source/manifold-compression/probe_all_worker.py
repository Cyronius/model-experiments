"""Disposable CUDA worker for the all-neuron check on the healed student (heal_probe.py with
heal_probe_all_rules.json): heal_worker's bundle and model checks, with the probe_all/ folder in
outputs/. No RunPod credential.

The generations use the student the evaluate stage checked, so they run only after it."""
import heal_worker

PROBE = ['heal_probe.py']
RULES = ['--rules', 'heal_probe_all_rules.json']
STAGES = [
    ('evaluate', PROBE + ['evaluate'] + RULES, []),
    ('generate_all', PROBE + ['generate', '--arm', 'all'] + RULES, ['evaluate']),
    ('generate_all_unhealed', PROBE + ['generate', '--arm', 'all_unhealed'] + RULES, ['evaluate']),
]

if __name__ == '__main__':
    heal_worker.run(STAGES, 'probe_all')
