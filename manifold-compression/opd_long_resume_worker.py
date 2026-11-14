"""Disposable CUDA worker that carries the opd_long distillation on from its last checkpoint
(slim_opd.py train --resume, then after, with opd_long_rules.json). The first pod's checks before,
progress, rollouts and checkpoint come in the bundle's opd_long/ folder, which moves to outputs/ as
opd_long_worker's does. No RunPod credential."""
import heal_worker

OPD = ['slim_opd.py']
RULES = ['--rules', 'opd_long_rules.json']
STAGES = [
    ('train', OPD + ['train', '--resume'] + RULES, []),
    ('after', OPD + ['after'] + RULES, ['train']),
]


if __name__ == '__main__':
    heal_worker.run(STAGES, 'opd_long')
