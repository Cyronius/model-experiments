"""Load a saved head (npz written by stage2/run.py::save_head) as an object with .raw(features)."""
import json
import numpy as np


class NpzHead:
    def __init__(self, z):
        self.z = {k: z[k] for k in z.files}
        self.kind = str(self.z['kind'])
        self.meta = json.loads(str(self.z['meta'])) if 'meta' in self.z else {}

    def raw(self, X):
        z = self.z
        Zs = (X - z['mu']) / z['sd']
        if self.kind == 'ridge':
            return Zs @ z['W'] + z['b']
        h = Zs @ z['fc1.weight'].T + z['fc1.bias']
        h = 0.5 * h * (1 + np.tanh(0.7978845608028654 * (h + 0.044715 * h ** 3)))
        return h @ z['fc2.weight'].T + z['fc2.bias'] + Zs @ z['skip.weight'].T + z['skip.bias']


def head_from_npz(path):
    return NpzHead(np.load(path))
