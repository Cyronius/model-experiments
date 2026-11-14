"""Disposable CUDA worker for the on-policy distillation pilot (slim_opd.py): heal_worker's bundle
and model checks, with the opd/ folder in outputs/. No RunPod credential.

before (the healed student's checks and the references) and train do not depend on each other;
after needs only the distilled adapter."""
import heal_worker

OPD = ['slim_opd.py']
STAGES = [
    ('before', OPD + ['before'], []),
    ('train', OPD + ['train'], []),
    ('after', OPD + ['after'], ['train']),
]


if __name__ == '__main__':
    heal_worker.run(STAGES, 'opd')
