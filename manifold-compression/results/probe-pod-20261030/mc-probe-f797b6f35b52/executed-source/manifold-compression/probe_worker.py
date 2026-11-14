"""Disposable CUDA worker for the checks on the healed student (heal_probe.py): heal_worker's bundle
and model checks, with the probe/ folder in outputs/. No RunPod credential.

The other stages use the student the evaluate stage checked, so they run only after it."""
import heal_worker

PROBE = ['heal_probe.py']
STAGES = [
    ('evaluate', PROBE + ['evaluate'], []),
    ('state', PROBE + ['state'], ['evaluate']),
    ('generate_fp32', PROBE + ['generate', '--arm', 'fp32'], ['evaluate']),
    ('generate_perfect', PROBE + ['generate', '--arm', 'perfect'], ['evaluate']),
]

if __name__ == '__main__':
    heal_worker.run(STAGES, 'probe')
