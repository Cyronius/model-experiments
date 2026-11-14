"""Stage 2 arm 5, extension to the full training pool (the curve was still rising at 300,000).
Refits the selected arm (H2, native reading) on all eligible non-validation Jaswanth texts with the
same hyperparameters. If its validation loss is lower than the 300k fit, it becomes the selected head
and the final evaluation, calibration, mixed-index test and gates are recomputed with it
(the 300k results stay in result-300k.json)."""
import json, sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R
from heads import train_mlp


def main():
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
    res = R.res
    arm, mode = res['selected_arm']['arm'], res['selected_arm']['mode']
    assert arm == 'H2'
    W2 = R.W2
    X = np.concatenate([R.load_split('q8', mode, 'train'), np.load(W2 / f'F_q8_{mode}_rest.npy')])
    Y = np.concatenate([np.asarray(R.Ytr, np.float16), np.load(W2 / 'targets_rest.npy')])
    Xv = R.load_split('q8', mode, 'val')
    n = len(X)
    R.log('full set', n)
    head, hist = train_mlp(X, Y, Xv, R.Yv, epochs=6, log=lambda h: R.log('full', h))
    R.save_head(head, W2 / f'head_H2_q8_{mode}_full.npz', {'arm': arm, 'mode': mode, 'precision': 'q8', 'n': n})
    ev = R.eval_head(head, 'q8', mode)
    res['data_curve'][str(n)] = {'val_loss': hist[-1]['val_loss'], 'core_TS_504_1024': ev['core|TS_504_1024'],
                                 'history': hist}
    res['arms'][f'H2_q8_{mode}_full'] = {'fit': {'val_loss': hist[-1]['val_loss'], 'history': hist, 'n': n}, 'eval': ev}
    R.save()
    if hist[-1]['val_loss'] < res['arms'][f'H2_q8_{mode}']['fit']['val_loss']:
        res['selected_arm']['n_train'] = n
        res['selected_arm']['full_set_rule'] = 'full-set refit replaces the 300k fit because its validation loss is lower'
        R.finalize(head, arm, mode, n)
    else:
        res['selected_arm']['n_train'] = 300000
        res['selected_arm']['full_set_rule'] = 'full-set refit did not lower validation loss; 300k fit kept'
        R.save()
    R.log(json.dumps(res['gates'], indent=1))


if __name__ == '__main__':
    main()
