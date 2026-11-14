"""Disposable CUDA worker for the on-policy distillation from the every-neuron teacher (slim_opd.py
with opd_self_rules.json): opd_worker's stages, with the opd_self/ folder in outputs/. No RunPod
credential."""
import heal_worker

OPD = ['slim_opd.py']
RULES = ['--rules', 'opd_self_rules.json']
STAGES = [
    ('before', OPD + ['before'] + RULES, []),
    ('train', OPD + ['train'] + RULES, []),
    ('after', OPD + ['after'] + RULES, ['train']),
]


if __name__ == '__main__':
    heal_worker.run(STAGES, 'opd_self')
