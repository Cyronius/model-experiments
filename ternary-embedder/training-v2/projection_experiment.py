"""Learn the existing output projection from cached, frozen ternary features.

No teacher targets and no new inference operation: normalize(A normalize(W h))
equals normalize((A W) h). All Transformer weights remain ternary/int8.
"""
import argparse
import collections
import datetime
import json
import random
import copy
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pair_training_state import SourceSampler, atomic_json, atomic_torch_save, digest

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
WORK = ROOT / 'work/projection'
PROFILE = 'projection'


def read(path):
    return [json.loads(line) for line in path.open()]


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    plan_path = WORK / 'plan.json'
    if plan_path.exists():
        return json.loads(plan_path.read_text())
    if PROFILE == 'projection-blend':
        previous = ROOT / 'work/projection'
        plan = copy.deepcopy(json.loads((previous / 'plan.json').read_text()))
        for name, checksum in plan['prepared_sha256'].items():
            assert digest(previous / name) == checksum
            shutil.copy2(previous / name, WORK / name)
        selection = json.loads((ROOT / 'ternary-blends/results/frozen.json').read_text())['models']['ternary-3-25']
        assert digest(Path(selection['path'])) == selection['sha256']
        plan.update(profile=PROFILE, parent=selection, fixed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    comparison='Identical cached training/held rows, seed and projection hyperparameters as projection; backbone is the fixed25% QAT blend instead of science-fresh.')
        atomic_json(plan_path, plan)
        print('Prepared matched projection test on the three-layer ternary blend.', flush=True)
        return plan
    source = ROOT / 'work/science-mix-clean'
    audit = json.loads((source / 'audit.json').read_text())
    for split in ('train', 'dev'):
        assert digest(source / (split + '.jsonl')) == audit['output_sha256'][split]
    human = ROOT / 'work/human-triplets'
    human_audit = json.loads((human / 'audit.json').read_text())
    assert digest(human / 'train.jsonl') == human_audit['output_sha256']['train']
    selected = set(random.Random(71).sample(range(audit['rows']['train']), 80000))
    train = []
    for index, line in enumerate((source / 'train.jsonl').open()):
        if index in selected:
            row = json.loads(line)
            train.append({k: v for k, v in row.items() if not k.startswith('negative')})
    assert len(train) == 80000
    train += read(human / 'train.jsonl')
    dev = [{k: v for k, v in row.items() if not k.startswith('negative')}
           for row in read(source / 'dev.jsonl')]
    from prepare_human_triplets import triplets
    aux = ROOT / 'work/nli-aux-global'
    aux_audit = json.loads((aux / 'audit.json').read_text())
    assert digest(aux / 'dev.jsonl') == aux_audit['output_sha256']['dev']
    nli_dev, _ = triplets(read(aux / 'dev.jsonl'))
    dev += nli_dev
    def identities(rows):
        return {r[k] for r in rows for k in ('query_visible_key', 'positive_visible_key', 'negative_visible_key') if r.get(k)}
    assert not identities(train) & identities(dev)
    queries = {r['query_visible_key'] for r in train}
    known = collections.defaultdict(set)
    for line in (source / 'train.jsonl').open():
        row = json.loads(line)
        if row['query_visible_key'] in queries:
            known[row['query_visible_key']].add(row['positive_visible_key'])
    for row in train:
        known[row['query_visible_key']].add(row['positive_visible_key'])
    texts = {}
    for row in train + dev:
        for field in ('query', 'positive', 'negative'):
            if row.get(field):
                texts.setdefault(row[field + '_visible_key'], row[field])
    selection = json.loads((ROOT / 'science-ternary/results/frozen.json').read_text())['models']['3-ternary-fresh']
    assert digest(Path(selection['path'])) == selection['sha256']
    for name, rows in (('train', train), ('dev', dev)):
        with (WORK / (name + '.jsonl')).open('w') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    atomic_json(WORK / 'known.json', {k: sorted(v) for k, v in known.items()})
    atomic_json(WORK / 'texts.json', list(texts.items()))
    plan = dict(fixed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), parent=selection,
                paid_resource_allocated=False, teacher_targets=False,
                backbone_frozen=True, existing_float_output_projection_only=True,
                seed=71, sample_relevance_rows=80000, human_triplet_rows=human_audit['rows'],
                train_rows=len(train), dev_rows=len(dev), unique_texts=len(texts),
                epochs=3, batch=512, lr=.0003, regularizers=[.1, 1., 10.],
                regularizer_formula='lambda * sum((A-I)^2) /384',
                selection='Highest full held-source macro nDCG; all nine held sources within2 points of identity. Includes official held human-NLI triplets. Null identity eligible. Freeze all three before external scoring.',
                train_dev_visible_overlap=0, reserved_judgments_read=False,
                input_sha256={str(p.relative_to(PROJECT)): digest(p) for p in
                              (source / 'train.jsonl', source / 'dev.jsonl', human / 'train.jsonl', aux / 'dev.jsonl')},
                prepared_sha256={name: digest(WORK / name) for name in ('train.jsonl', 'dev.jsonl', 'known.json', 'texts.json')})
    atomic_json(plan_path, plan)
    print(json.dumps({k: plan[k] for k in ('train_rows', 'dev_rows', 'unique_texts', 'regularizers')}, indent=2), flush=True)
    return plan


@torch.inference_mode()
def cache():
    torch.set_num_threads(2)
    plan = prepare()
    for name, checksum in plan['prepared_sha256'].items():
        assert digest(WORK / name) == checksum
    final = WORK / 'features.npy'
    if final.exists():
        assert digest(final) == json.loads((WORK / 'features.json').read_text())['sha256']
        return
    sys.path.insert(0, str(PROJECT / 'native-stage'))
    from fused_encoder import FusedEncoder
    from transformers import AutoTokenizer
    texts = json.loads((WORK / 'texts.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(PROJECT / 'tokenizer', local_files_only=True)
    partial = WORK / 'features.partial.npy'
    progress_path = WORK / 'cache-progress.json'
    progress = json.loads(progress_path.read_text()) if partial.exists() and progress_path.exists() else None
    if progress:
        assert progress['parent_sha256'] == plan['parent']['sha256']
        assert progress['texts_sha256'] == plan['prepared_sha256']['texts.json']
    done = progress['done'] if progress else 0
    features = np.lib.format.open_memmap(partial, mode='r+' if done else 'w+', dtype=np.float32, shape=(len(texts), 384))
    encoder = FusedEncoder(128, checkpoint=Path(plan['parent']['path']))
    try:
        for start in range(done, len(texts), 128):
            batch = [v[1] for v in texts[start:start + 128]]
            tokens = tokenizer(batch, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
            for index in range(len(batch)):
                features[start + index] = encoder(tokens['input_ids'][index:index + 1],
                                                  tokens['attention_mask'][index:index + 1].bool())[0].numpy()
            done = start + len(batch)
            if done % 8192 == 0 or done == len(texts):
                features.flush()
                atomic_json(progress_path, dict(done=done, parent_sha256=plan['parent']['sha256'], texts_sha256=plan['prepared_sha256']['texts.json']))
                print('Cached native training features', done, '/', len(texts), flush=True)
                encoder.profile(reset=True)
        assert np.isfinite(features).all()
        assert np.max(np.abs(np.linalg.norm(features, axis=1) - 1)) < .0001
    finally:
        encoder.close()
    features.flush()
    del features
    partial.replace(final)
    atomic_json(WORK / 'features.json', dict(sha256=digest(final), parent_sha256=plan['parent']['sha256'],
                                            backend='Actual native128 ternary encoder; cached features, no teacher. Not a timing measurement.'))


def validate(matrix, vectors, rows, lookup, asymmetric=False):
    result = {}
    by_source = collections.defaultdict(list)
    for row in rows:
        by_source[row['source']].append(row)
    with torch.no_grad():
        for source, group in by_source.items():
            queries = {r['query_visible_key']: None for r in group}
            docs = {r['positive_visible_key']: None for r in group}
            for row in group:
                if row.get('negative'):
                    docs.setdefault(row['negative_visible_key'], None)
            qi, di = {k: i for i, k in enumerate(queries)}, {k: i for i, k in enumerate(docs)}
            q = F.normalize(vectors[[lookup[k] for k in queries]] @ matrix.T, dim=-1)
            documents = vectors[[lookup[k] for k in docs]]
            d = documents if asymmetric else F.normalize(documents @ matrix.T, dim=-1)
            scores = q @ d.T
            positives = torch.zeros_like(scores, dtype=torch.bool)
            positives[[qi[r['query_visible_key']] for r in group], [di[r['positive_visible_key']] for r in group]] = True
            count = positives.sum(-1)
            k = min(10, len(docs))
            discounts = 1 / torch.log2(torch.arange(2, k + 2, dtype=torch.float32))
            actual = positives.gather(1, scores.topk(k, dim=-1).indices).float() @ discounts
            ideal = (torch.arange(k)[None, :] < count[:, None]).float() @ discounts
            result[source] = float((actual / ideal).mean()) * 100
    return dict(by_source=result, macro_ndcg10=sum(result.values()) / len(result))


def train():
    torch.set_num_threads(2)
    plan = json.loads((WORK / 'plan.json').read_text())
    assert digest(Path(plan['parent']['path'])) == plan['parent']['sha256']
    assert digest(WORK / 'features.npy') == json.loads((WORK / 'features.json').read_text())['sha256']
    frozen_path = ROOT / PROFILE / 'results/frozen.json'
    if frozen_path.exists():
        raise RuntimeError('Projection candidates already frozen')
    for name, checksum in plan['prepared_sha256'].items():
        assert digest(WORK / name) == checksum
    vectors = torch.from_numpy(np.load(WORK / 'features.npy').copy())
    lookup = {key: i for i, (key, _) in enumerate(json.loads((WORK / 'texts.json').read_text()))}
    train_rows, dev_rows = read(WORK / 'train.jsonl'), read(WORK / 'dev.jsonl')
    known = {k: set(v) for k, v in json.loads((WORK / 'known.json').read_text()).items()}
    by_source = collections.defaultdict(list)
    for i, row in enumerate(train_rows):
        by_source[row['source']].append(i)
    identity = torch.eye(384)
    baseline = validate(identity, vectors, dev_rows, lookup)
    from pair_train import batch_documents, batch_positive_mask, positive_pair_loss
    frozen = dict(frozen_utc=None, selected_before_external_scores=True, local_only=True,
                  rule=plan['selection'], models={}, source_plan_sha256=digest(WORK / 'plan.json'))
    for regularizer in plan['regularizers']:
        torch.manual_seed(plan['seed'])
        matrix = torch.nn.Parameter(identity.clone())
        optimizer = torch.optim.AdamW([matrix], lr=plan['lr'], weight_decay=0.)
        sampler = SourceSampler(dict(by_source), plan['batch'], plan['seed'])
        best, best_validation, best_step = identity.clone(), baseline, 0
        steps = plan['epochs'] * sampler.batches_per_epoch
        history = []
        for step in range(1, steps + 1):
            _, indexes = sampler.next()
            batch = [train_rows[i] for i in indexes]
            docs = batch_documents(batch, True)
            q = F.normalize(vectors[[lookup[r['query_visible_key']] for r in batch]] @ matrix.T, dim=-1)
            d = F.normalize(vectors[[lookup[k] for k in docs]] @ matrix.T, dim=-1)
            mask = batch_positive_mask(batch, list(docs), known, 'cpu')
            loss = positive_pair_loss(q, d, mask) + regularizer * (matrix - identity).square().sum() / 384
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_([matrix], 1.))
            optimizer.step()
            if step % sampler.batches_per_epoch == 0:
                held = validate(matrix, vectors, dev_rows, lookup)
                eligible = all(held['by_source'][k] >= v - 2 for k, v in baseline['by_source'].items())
                history.append(dict(step=step, loss=float(loss.detach()), validation=held, source_floor_passed=eligible))
                if eligible and held['macro_ndcg10'] > best_validation['macro_ndcg10']:
                    best, best_validation, best_step = matrix.detach().clone(), held, step
                atomic_torch_save(WORK / f'state-{regularizer:g}.pt', dict(matrix=matrix.detach(), optimizer=optimizer.state_dict(), sampler=sampler.state_dict(), step=step, best=best, best_step=best_step))
                print('Projection', regularizer, 'epoch', step // sampler.batches_per_epoch,
                      'held macro', round(held['macro_ndcg10'], 4), 'eligible', eligible, flush=True)
        parent = torch.load(plan['parent']['path'], map_location='cpu', weights_only=False)
        assert parent['quant']
        original = parent['state']['out.weight'].clone()
        parent['state']['out.weight'] = best @ original
        test = torch.randn(17, 384)
        direct = F.normalize(F.normalize(test @ original.T, dim=-1) @ best.T, dim=-1)
        folded = F.normalize(test @ parent['state']['out.weight'].T, dim=-1)
        torch.testing.assert_close(direct, folded, atol=2e-6, rtol=2e-6)
        name = f'projection-{regularizer:g}'
        path = WORK / (name + '.pt')
        parent['meta'] = dict(output_projection_training=True, parent_sha256=plan['parent']['sha256'],
                              validation=best_validation, regularizer=regularizer, step=best_step)
        atomic_torch_save(path, parent)
        atomic_json(WORK / (name + '-history.json'), history)
        frozen['models'][name] = dict(path=str(path.resolve()), sha256=digest(path), precision='ternary', layers=3,
                                     eligible_for_promotion=True, validation=best_validation, step=best_step,
                                     initial_checkpoint_sha256=plan['parent']['sha256'], regularizer=regularizer,
                                     operator_change='Existing float output matrix composed in place; ternary backbone unchanged.',
                                     validation_guard='Every held relevance/human-NLI retrieval source within2 nDCG points of identity; classifier guard is not this experiment objective.',
                                     matrix_condition_number=float(torch.linalg.cond(best)))
    frozen['frozen_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(frozen_path, frozen)
    print('All three projection candidates frozen before external scoring.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['cache', 'train'])
    parser.add_argument('--profile', choices=['projection', 'projection-blend'], default='projection')
    args = parser.parse_args()
    PROFILE = args.profile
    WORK = ROOT / 'work' / PROFILE
    cache() if args.action == 'cache' else train()
