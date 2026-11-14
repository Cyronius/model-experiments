"""Matched symmetric/query-only projection tests on clean relevance pairs.

Uses existing native feature caches, source-training labels only, and no paid
service. The query-only variant leaves document and generic text vectors intact.
"""
import copy
import datetime
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pair_training_state import SourceSampler, atomic_json, atomic_torch_save, digest
from projection_experiment import read, validate
from pair_train import batch_documents, batch_positive_mask, positive_pair_loss

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'work/role-projection'


def main():
    torch.set_num_threads(2)
    WORK.mkdir(parents=True, exist_ok=True)
    frozen_path = ROOT / 'role-projection/results/frozen.json'
    if frozen_path.exists():
        raise RuntimeError('Role projection cohort already frozen')
    cache = ROOT / 'work/projection-blend'
    previous = json.loads((cache / 'plan.json').read_text())
    for name, checksum in previous['prepared_sha256'].items():
        assert digest(cache / name) == checksum
    assert digest(cache / 'features.npy') == json.loads((cache / 'features.json').read_text())['sha256']
    parent = previous['parent']
    assert digest(Path(parent['path'])) == parent['sha256']
    source_summary = ROOT / 'ternary-blends/results/summary.json'
    assert json.loads(source_summary.read_text())['models']['ternary-3-25']['selection']['sha256'] == parent['sha256']
    train = [r for r in read(cache / 'train.jsonl') if r['source'] != 'human-nli-triplets']
    dev = [r for r in read(cache / 'dev.jsonl') if r['source'] != 'human-nli-triplets']
    assert len(train) == 80000 and len(dev) == 21011
    assert all(not r.get('negative') for r in train + dev)
    plan = dict(fixed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                parent=parent, source_cache_plan_sha256=digest(cache / 'plan.json'),
                teacher_targets=False, paid_resource_allocated=False,
                train_rows=80000, dev_rows=21011, source_count=8,
                epochs=3, batch=512, lr=.0003, seed=83, regularizers=[1., 10.],
                modes=['symmetric', 'query-only'],
                selection='Highest full held-source macro nDCG across8 relevance sources, every source within2 points of identity; identity eligible. All four frozen before external scoring.',
                scope='Same data, seed, rates and regularizers isolate query-only versus shared projection. No human contradictions in this test. Query head folds into one role-selected output linear operation.')
    atomic_json(WORK / 'plan.json', plan)
    vectors = torch.from_numpy(np.load(cache / 'features.npy').copy())
    lookup = {k: i for i, (k, _) in enumerate(json.loads((cache / 'texts.json').read_text()))}
    known = {k: set(v) for k, v in json.loads((cache / 'known.json').read_text()).items()}
    by_source = {}
    for i, row in enumerate(train):
        by_source.setdefault(row['source'], []).append(i)
    identity = torch.eye(384)
    frozen = dict(local_only=True, selected_before_external_scores=True, frozen_utc=None,
                  source_plan_sha256=digest(WORK / 'plan.json'), models={})
    for asymmetric in (False, True):
        baseline = validate(identity, vectors, dev, lookup, asymmetric=asymmetric)
        mode = 'query' if asymmetric else 'shared'
        for regularizer in plan['regularizers']:
            torch.manual_seed(plan['seed'])
            matrix = torch.nn.Parameter(identity.clone())
            optimizer = torch.optim.AdamW([matrix], lr=plan['lr'], weight_decay=0.)
            sampler = SourceSampler(by_source, plan['batch'], plan['seed'])
            best, best_held, best_step = identity.clone(), baseline, 0
            history = []
            for step in range(1, plan['epochs'] * sampler.batches_per_epoch + 1):
                _, indexes = sampler.next()
                batch = [train[i] for i in indexes]
                documents = batch_documents(batch)
                q = F.normalize(vectors[[lookup[r['query_visible_key']] for r in batch]] @ matrix.T, dim=-1)
                d = vectors[[lookup[k] for k in documents]]
                if not asymmetric:
                    d = F.normalize(d @ matrix.T, dim=-1)
                mask = batch_positive_mask(batch, list(documents), known, 'cpu')
                loss = positive_pair_loss(q, d, mask) + regularizer * (matrix - identity).square().sum() / 384
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_([matrix], 1.))
                optimizer.step()
                if step % sampler.batches_per_epoch == 0:
                    held = validate(matrix, vectors, dev, lookup, asymmetric=asymmetric)
                    eligible = all(held['by_source'][k] >= value - 2 for k, value in baseline['by_source'].items())
                    history.append(dict(step=step, held=held, source_floor_passed=eligible))
                    if eligible and held['macro_ndcg10'] > best_held['macro_ndcg10']:
                        best, best_held, best_step = matrix.detach().clone(), held, step
            name = f'{mode}-{regularizer:g}'
            atomic_json(WORK / (name + '-history.json'), history)
            if best_step == 0:
                # Same physical checkpoint: safe to reuse its already scored
                # quality rather than spend minutes re-encoding an identity.
                frozen['models'][name] = dict(parent, step=0, selected_identity=True,
                                              source_summary=str(source_summary.resolve()), source_arm='ternary-3-25',
                                              source_summary_sha256=digest(source_summary), experiment_mode=mode)
            else:
                checkpoint = torch.load(parent['path'], map_location='cpu', weights_only=False)
                if asymmetric:
                    # State is byte-for-byte the document encoder's state.
                    checkpoint['meta'] = dict(query_projection=best, query_head_format='optional-query-output-v1',
                                              parent_sha256=parent['sha256'], validation=best_held, step=best_step,
                                              additional_query_parameters=best.numel())
                else:
                    checkpoint['state']['out.weight'] = best @ checkpoint['state']['out.weight']
                    checkpoint['meta'] = dict(shared_output_projection=True, parent_sha256=parent['sha256'],
                                              validation=best_held, step=best_step)
                path = WORK / (name + '.pt')
                atomic_torch_save(path, checkpoint)
                frozen['models'][name] = dict(path=str(path.resolve()), sha256=digest(path), precision='ternary', layers=3,
                                              step=best_step, eligible_for_promotion=True,
                                              initial_checkpoint_sha256=parent['sha256'], validation=best_held,
                                              experiment_mode=mode, query_head=asymmetric,
                                              additional_query_parameters=147456 if asymmetric else 0,
                                              matrix_condition_number=float(torch.linalg.cond(best)))
            print(name, 'selected step', best_step, 'held', round(best_held['macro_ndcg10'], 4), flush=True)
    frozen['frozen_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(frozen_path, frozen)
    print('All four role-projection candidates frozen before external scores.', flush=True)


if __name__ == '__main__':
    main()
