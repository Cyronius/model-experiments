"""Random-weight CPU forward/loss/backward plus exact supervised tag decode on real IDs."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import torch
from tagger import (Config, Tags, Tagger, SteTokenizer, Lexicon, EncodingError,
                    collate, source_units, target_units, supervision, loss, rewrite, check, errors)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tokenizer'))
from prepare import input_paths, records
from ste_tokenizer import sha256


def smoke(vocab, pairs, output, batch_size=4, threads=4, scan_limit=5000):
    torch.manual_seed(0)
    torch.set_num_threads(threads)
    tok, lex = SteTokenizer(vocab), Lexicon()
    tags = Tags(tok)
    examples, labels, cases, scanned, skipped = [], [], [], 0, 0
    candidates = {}
    for path, _ in input_paths(pairs):
        if len(candidates) >= min(4, batch_size) or scanned >= scan_limit:
            break
        for row in records(path):
            if row['split'] != 'train':
                continue
            scanned += 1
            units, targets = source_units(row), target_units(row, tok)
            actions, casing = supervision(units, targets, tags)
            try:
                decoded = tags.apply(units, actions, casing, row['slots'])
            except EncodingError:
                decoded = None
            if decoded != targets:
                skipped += 1
            else:
                candidates.setdefault(row['kind'], [])
                if len(candidates[row['kind']]) < batch_size:
                    candidates[row['kind']].append((row, actions, casing))
            if len(candidates) >= min(4, batch_size) or scanned >= scan_limit:
                break
    chosen = [items[0] for _, items in sorted(candidates.items())]
    chosen += [item for _, items in sorted(candidates.items()) for item in items[1:]]
    for row, actions, casing in chosen[:batch_size]:
        examples.append(row)
        labels.append(actions)
        cases.append(casing)
    if len(examples) != batch_size:
        raise RuntimeError('not enough exactly representable one-pass training examples')
    batch = collate(examples, tok, lex)
    shape = batch['actions'].shape
    target_tags = torch.full(shape, -100, dtype=torch.long)
    target_cases = torch.zeros(shape, dtype=torch.long)
    for i, (actions, casing) in enumerate(zip(labels, cases)):
        target_tags[i, :len(actions)] = torch.tensor(actions)
        target_cases[i, :len(casing)] = torch.tensor(casing)
    model = Tagger(tok)
    start = time.monotonic()
    logits, case_logits = model(batch)
    objective = loss(logits, case_logits, target_tags, target_cases, batch['actions'], tags)
    if not torch.isfinite(objective):
        raise RuntimeError('nonfinite loss')
    objective.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError('nonfinite gradient')
    elapsed = time.monotonic() - start
    probe = next((r for r in examples if errors(check(r['src'], lex, r['mode']))), examples[0])
    result = {'scope': 'random weights; no training or quality/latency claim',
              'torch_version': torch.__version__, 'threads': threads, 'config': asdict(model.config),
              'parameters': sum(p.numel() for p in model.parameters()),
              'input_size': tok.meta['input_size'], 'output_size': tok.meta['output_size'], 'tags': tags.size,
              'batch_size': batch_size, 'training_records_scanned': scanned,
              'one_pass_skipped_in_prefix': skipped, 'teacher_forced_decodes_exact': batch_size,
              'loss': objective.item(), 'finite_gradients': True, 'forward_backward_seconds': elapsed,
              'slot_reference_pieces': batch['refs'].gt(0).sum().item(),
              'checker_feature_values': batch['features'].gt(0).sum().item(),
              'kinds': [r['kind'] for r in examples],
              'review_labels': [r['meaning_review'] for r in examples],
              'random_decode_probe_kind': probe['kind'],
              'random_decode': rewrite(model, tok, lex, probe['src'], probe['mode']),
              'vocab_sha256': sha256(vocab / 'vocab.json'),
              'pairs_summary_sha256': sha256(pairs / 'summary.json'),
              'code_sha256': {n: sha256(Path(__file__).parent / n) for n in ('tagger.py', 'smoke.py')}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vocab', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('batch size and threads must be positive')
    print(json.dumps(smoke(args.vocab, args.pairs, args.out, args.batch_size, args.threads), indent=2))
