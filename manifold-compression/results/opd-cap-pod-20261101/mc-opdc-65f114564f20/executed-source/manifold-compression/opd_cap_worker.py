"""Disposable CUDA worker for section 17's checks again with a 4,096-token cap (slim_opd.py check with
opd_cap_rules.json), with the opd_cap/ folder in outputs/. No training. No RunPod credential."""
import heal_worker

STAGES = [
    ('check', ['slim_opd.py', 'check', '--rules', 'opd_cap_rules.json'], []),
]


if __name__ == '__main__':
    heal_worker.run(STAGES, 'opd_cap')
