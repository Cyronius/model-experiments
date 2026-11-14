"""Disposable CUDA worker for the every-neuron-teacher distillation carried on from section 16's
adapter, with healing text (slim_opd.py with opd_long_rules.json): opd_worker's stages, with the
opd_long/ folder in outputs/. No RunPod credential."""
import heal_worker

OPD = ['slim_opd.py']
RULES = ['--rules', 'opd_long_rules.json']
STAGES = [
    ('before', OPD + ['before'] + RULES, []),
    ('train', OPD + ['train'] + RULES, []),
    ('after', OPD + ['after'] + RULES, ['train']),
]


if __name__ == '__main__':
    heal_worker.run(STAGES, 'opd_long')
