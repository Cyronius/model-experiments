"""Compare a single ranking and healing stage to the saved iterative trajectory."""
import json
import numpy as np
import torch
from transformers import AutoTokenizer

import adaptive_mlp_pilot as pilot
from iterate_mlp import OUT, capture_inputs, resize


def run_arm(kept, teacher_x, model, guard, test):
    corrections = {}
    for i, indices in kept.items():
        mlp = model.model.layers[i].mlp
        x = teacher_x[i]
        mask = torch.ones(x.shape[1], device=x.device, dtype=torch.bool)
        mask[indices] = False
        target = x[:, mask] @ mlp.down_proj.weight[:, mask].T
        features = x[:, indices]
        gram = features @ features.T
        gram.diagonal().add_(0.1 * gram.diag().mean())
        corrections[i] = (features.T @ torch.linalg.solve(gram, target)).T
        resize(mlp, indices)
    raw = pilot.evaluate(model, test)
    for i, delta in corrections.items():
        model.model.layers[i].mlp.down_proj.weight.add_(delta)
    return dict(test_raw=raw, test=pilot.evaluate(model, test), guard=pilot.evaluate(model, guard),
                count=sum(p.numel() for p in model.parameters()), fits=len(kept))


def main():
    ref = json.loads((OUT / 'results.json').read_text())
    tok = AutoTokenizer.from_pretrained(pilot.CHECKPOINT, local_files_only=True)
    train = pilot.samples(tok, 'train', 40, 716)
    val = pilot.samples(tok, 'valid', 80, 1717)
    test = pilot.samples(tok, 'test', 128, 2718)
    model = pilot.load_model()
    indices = range(len(model.model.layers))
    energy_inputs = capture_inputs(model, train[:8], indices)
    ordered, scores = {}, {}
    for i in indices:
        w = model.model.layers[i].mlp.down_proj.weight
        score = energy_inputs.pop(i).square().mean(0) * w.square().sum(0)
        ordered[i] = score.argsort(stable=True)
        scores[i] = np.mean(pilot.masked(model, i, ordered[i][:1536], val[:16])).item()
    remaining, rem = divmod(ref['parameters'] - ref['final_parameters'], 3 * model.config.hidden_size)
    assert rem == 0
    independent = {}
    for i in sorted(scores, key=scores.get):
        cut = min(remaining, 1536)
        independent[i] = ordered[i][cut:].sort().values
        remaining -= cut
        if not remaining:
            break
    assert remaining == 0
    matched = {}
    files = [pilot.OUT / 'surgery.pt'] + [OUT / r['file'] for r in ref['rounds'] if r['accepted']]
    for path in files:
        edit = torch.load(path, map_location='cpu')
        i = edit['first_layer']
        previous = matched.get(i, torch.arange(6144))
        matched[i] = previous[edit['kept_indices']]
    matched = {i: v.to(pilot.DEVICE) for i, v in matched.items()}
    inputs = capture_inputs(model, train[8:], sorted(set(independent) | set(matched)))
    result = {'rank_scores': scores, 'checkpoint': ref['checkpoint'],
              'healing_tokens_per_fit': len(train[8:]) * pilot.LENGTH}
    iterative = np.asarray(ref['final_test'][ref['milestones']['5']])
    for name, kept in [('independent', independent), ('matched', matched)]:
        if name == 'matched':
            model = pilot.load_model()
        arm = run_arm(kept, inputs, model, val[16:], test)
        assert arm['count'] == ref['final_parameters']
        arm['widths'] = {i: len(v) for i, v in kept.items()}
        gaps = np.asarray(arm['test']) - iterative
        rng = np.random.default_rng(517)
        bootstrap = rng.choice(gaps, (20000, len(gaps)), replace=True).mean(1)
        arm['relative_ppl_vs_iterative'] = np.expm1(gaps.mean()).item()
        arm['paired_95_interval'] = np.expm1(np.quantile(bootstrap, [0.025, 0.975])).tolist()
        result[name] = arm
        print(name, np.mean(arm['test']), flush=True)
        del model
    (OUT / 'oneshot.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
