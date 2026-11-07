"""Teacher-forced edit categories on one kind of a packed rollout split: what the model emits where an edit is due.

For the edit states (terminal states left out) of --kind in --split, it counts the gold
category of each position (KEEP, DELETE, REPLACE, APPEND, TRANSFORM) against the
predicted one, the exact-tag rate by gold category, whether the gold tag is in the top
five, the mean probability mass the model gives each category, and the exact rate of a
factored decision (the category with the most summed mass, then the best tag in it).
Diagnostic only: no policy, guard or checker is involved.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tagger import Config, Tagger, Tags, SteTokenizer
from packed import PackedDataset, Collator, KINDS

NAMES = ('KEEP', 'DELETE', 'REPLACE', 'APPEND', 'TRANSFORM')


def confusion(model, tok, data, split, kind, limit, batch_size=16):
    tags = Tags(tok)
    groups = [(0, 1), (1, 2), (2, 2 + tags.vocab), (2 + tags.vocab, tags.transform_offset), (tags.transform_offset, tags.size)]
    def category(tag):
        return next(n for n, (a, b) in zip(NAMES, groups) if a <= tag < b)
    index = np.load(data / f'{split}.index.npy', mmap_mode='r')
    rows = np.flatnonzero((index[:, 5] == KINDS.index(kind)) & (index[:, 4] == 0))[:limit]
    dataset, collate = PackedDataset(data, split), Collator(tok)
    pairs, exact, factored, top5, mass = Counter(), Counter(), Counter(), Counter(), Counter()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = collate([dataset[i] for i in rows[start:start + batch_size]])
            logits, _ = model(batch)
            selected = batch['actions'] & batch['labels'].ne(-100)
            probs = logits[selected].float().softmax(-1)
            gold = batch['labels'][selected]
            sums = torch.stack([probs[:, a:b].sum(-1) for a, b in groups], -1)
            best = probs.argmax(-1)
            ranks = (probs > probs.gather(1, gold[:, None])).sum(-1)
            for k, (g, p, c) in enumerate(zip(gold.tolist(), best.tolist(), sums.argmax(-1).tolist())):
                a, b = groups[c]
                g_name = category(g)
                pairs[(g_name, category(p))] += 1
                factored[g_name] += int(int(probs[k, a:b].argmax()) + a == g)
                if g_name != 'KEEP':
                    exact[g_name] += int(p == g)
                    top5[g_name] += int(ranks[k] < 5)
                    for name, value in zip(NAMES, sums[k].tolist()):
                        mass[(g_name, name)] += value
    gold = Counter()
    for (g, _), n in pairs.items():
        gold[g] += n
    return {'kind': kind, 'split': split, 'edit_states': len(rows), 'gold_positions': dict(gold),
            'exact_by_gold': {g: round(exact[g] / gold[g], 4) for g in gold if g != 'KEEP'},
            'top5_by_gold': {g: round(top5[g] / gold[g], 4) for g in gold if g != 'KEEP'},
            'factored_exact_by_gold': {g: round(factored[g] / gold[g], 4) for g in gold},
            'mean_category_mass_given_gold': {g: {n: round(mass[(g, n)] / gold[g], 4) for n in NAMES} for g in gold if g != 'KEEP'},
            'predicted_given_gold': {g: {p: n for (h, p), n in sorted(pairs.items()) if h == g} for g in gold}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vocabulary', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kind', default='doc_teacher', choices=KINDS)
    p.add_argument('--splits', default='dev,train')
    p.add_argument('--limit', type=int, default=2000, help='edit states per split')
    args = p.parse_args()
    torch.set_num_threads(4)
    tok = SteTokenizer(args.vocabulary)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['vocab_sha256'] != hashlib.sha256((args.vocabulary / 'vocab.json').read_bytes()).hexdigest():
        raise ValueError('checkpoint/vocabulary fingerprint mismatch')
    model = Tagger(tok, Config(**checkpoint['model_config']))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    result = {'checkpoint': str(args.checkpoint), 'data': str(args.data),
              'splits': {s: confusion(model, tok, args.data, s, args.kind, args.limit) for s in args.splits.split(',')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
