"""Train the canvas editor on documentation pairs (smoke test): train split of one kind of an ID export.

Rows are re-encoded from their text with every detected span as a slot (as at
inference). Each epoch uses every pair plus identity pairs (the reference rewritten
to itself, so clean input is left alone) making up --identity-share of the rows; a
pair gets a fresh corruption every time it is drawn. Dev loss uses a fixed
corruption seed; a quick decode of the first --quick-decode dev pairs runs at each
evaluation. A --holdout-share of the train groups is held out; the epoch with the
lowest held-out loss is saved as best.pt (the decoded checkpoint), or with
--select decode the evaluation with the most exact rewrites of the first
--select-rows held-out pairs (ties: fewer clean references changed).

Pretraining: --kind takes a comma list (local_rule,rule,...), --cap KIND:NOISE=N
samples at most N rows of one kind and noise, and local-rule drop_article pairs that
delete an article before a backtick literal are skipped (they teach adding "the"
before code). --init-from starts from a saved checkpoint and its model config.
No test-set use.
"""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time
import torch
from canvas import CanvasEditor, Config, SteTokenizer, Lexicon, EncodingError, encode_example, training_batch, loss
from decode import dev_rows, decode_rows


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


ARTICLE_LITERAL = re.compile(r'\b(?:the|a|an)\s+`', re.I)


def article_before_literal(row):
    """A drop_article pair whose deleted article stood before a backtick literal."""
    return (row['kind'] == 'local_rule' and row.get('noise') == 'drop_article'
            and len(ARTICLE_LITERAL.findall(row['ste'])) > len(ARTICLE_LITERAL.findall(row['src'])))


def train_rows(pairs, kinds, caps, seed, skipped):
    """Train rows of the given kinds; caps {(kind, noise): n} keep a seeded sample of n."""
    kept = {}
    for path in sorted((pairs / 'shards').glob('train-*.jsonl.gz')):
        with gzip.open(path, 'rt') as f:
            for line in f:
                row = json.loads(line)
                if row['kind'] not in kinds:
                    continue
                if article_before_literal(row):
                    skipped['article_before_literal'] = skipped.get('article_before_literal', 0) + 1
                    continue
                kept.setdefault((row['kind'], row.get('noise')), []).append(row)
    rows = []
    for key, group in sorted(kept.items(), key=lambda item: str(item[0])):
        cap = caps.get(key)
        if cap is not None and len(group) > cap:
            skipped[f'cap {key[0]}:{key[1]}'] = len(group) - cap
            group = random.Random(f'{seed}/{key}').sample(group, cap)
        rows.extend(group)
    return rows


def parse_cap(text):
    key, value = text.split('=')
    kind, noise = key.split(':')
    return (kind, noise), int(value)


def encode_all(rows, tok, lex, identity=False):
    examples, rejected = [], {}
    for row in rows:
        try:
            src = row['ste'] if identity else row['src']
            examples.append(encode_example(tok, lex, src, row['ste'], row['mode']))
        except EncodingError as error:
            rejected[str(error)] = rejected.get(str(error), 0) + 1
    return examples, rejected


def batches(examples, size, rng):
    """Shuffled, then sorted by length inside chunks of 50 batches to limit padding."""
    order = examples[:]
    rng.shuffle(order)
    result = []
    for start in range(0, len(order), size * 50):
        chunk = sorted(order[start:start + size * 50], key=lambda r: len(r['source_ids']))
        result.extend(chunk[i:i + size] for i in range(0, len(chunk), size))
    rng.shuffle(result)
    return result


def evaluate(model, dev, tok, args):
    model.eval()
    rng, total, count = random.Random(0), 0., 0
    with torch.no_grad():
        for start in range(0, len(dev), args.batch_size):
            batch = training_batch(dev[start:start + args.batch_size], tok, model.mask_id, rng, args.start_share)
            logits, cases = model(batch)
            total += loss(logits, cases, batch, args.copy_weight).item()
            count += 1
    return total / max(count, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vocabulary', type=Path, required=True)
    p.add_argument('--train-pairs', type=Path, required=True, help='ID export with train shards')
    p.add_argument('--dev-pairs', type=Path, required=True, help='ID export with dev shards')
    p.add_argument('--kind', default='doc_teacher', help='comma list of row kinds')
    p.add_argument('--cap', type=parse_cap, action='append', default=[], help='KIND:NOISE=N')
    p.add_argument('--init-from', type=Path, help='checkpoint to start from (its model config is used)')
    p.add_argument('--select', choices=('loss', 'decode'), default='loss', help='what best.pt maximizes')
    p.add_argument('--select-rows', type=int, default=256, help='held-out pairs decoded for --select decode')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--warmup-steps', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--copy-weight', type=float, default=.4)
    p.add_argument('--start-share', type=float, default=.2)
    p.add_argument('--identity-share', type=float, default=.3)
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--layers', type=int, default=6)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--ffn', type=int, default=1024)
    p.add_argument('--eval-every', type=int, default=5, help='epochs')
    p.add_argument('--quick-decode', type=int, default=128)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-hours', type=float, default=2)
    p.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16')
    p.add_argument('--holdout-share', type=float, default=.05, help='train groups held out for selection')
    p.add_argument('--input-dropout', type=float, default=.1)
    p.add_argument('--weight-decay', type=float, default=.05)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng, start = random.Random(args.seed), time.time()
    tok, lex = SteTokenizer(args.vocabulary), Lexicon()
    vocab_sha = hashlib.sha256((args.vocabulary / 'vocab.json').read_bytes()).hexdigest()
    status = lambda **extra: atomic_json(args.out / 'status.json', dict(updated=time.time(), **extra))
    status(state='encoding')
    source, holdout_source, skipped = [], [], {}
    dev_source = dev_rows(args.dev_pairs)
    dev_origins = {r['origin_id'] for r in dev_source}
    dev_text = {r['src'] for r in dev_source} | {r['ste'] for r in dev_source}
    for row in train_rows(args.train_pairs, set(args.kind.split(',')), dict(args.cap), args.seed, skipped):
        if row['origin_id'] in dev_origins or row['src'] in dev_text or row['ste'] in dev_text:
            skipped['dev_overlap'] = skipped.get('dev_overlap', 0) + 1  # a shared sentence, not only a shared id
            continue
        share = int.from_bytes(hashlib.sha256(('canvas-holdout/' + str(row['group'])).encode()).digest()[:8], 'big') / 2**64
        (holdout_source if share < args.holdout_share else source).append(row)
    holdout, rejected_holdout = encode_all(holdout_source, tok, lex)
    pairs, rejected = encode_all(source, tok, lex)
    identities, rejected_identity = encode_all(source, tok, lex, identity=True)
    dev, rejected_dev = encode_all(dev_source, tok, lex)
    if {r['id'] for r in source} & {r['id'] for r in dev_source}:
        raise ValueError('train/dev overlap')
    if {r['group'] for r in source} & {r['group'] for r in holdout_source}:
        raise ValueError('train/holdout group overlap')

    per_epoch = round(len(pairs) * args.identity_share / (1 - args.identity_share))
    steps_per_epoch = math.ceil((len(pairs) + per_epoch) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    initial = None
    if args.init_from:
        initial = torch.load(args.init_from, map_location='cpu', weights_only=False)
        if initial['vocab_sha256'] != vocab_sha:
            raise ValueError('checkpoint/vocabulary fingerprint mismatch')
        model = CanvasEditor(tok, Config(**dict(initial['model_config'], input_dropout=args.input_dropout)))
        model.load_state_dict(initial['model'])
    else:
        model = CanvasEditor(tok, Config(args.width, args.layers, args.heads, args.ffn, input_dropout=args.input_dropout))
    parameters = sum(p.numel() for p in model.parameters())
    config = dict(vars(args), vocabulary=str(args.vocabulary), train_pairs=str(args.train_pairs),
                  dev_pairs=str(args.dev_pairs), out=str(args.out), vocab_sha256=vocab_sha, parameters=parameters,
                  init_from=str(args.init_from) if args.init_from else None,
                  init_sha256=hashlib.sha256(args.init_from.read_bytes()).hexdigest() if args.init_from else None,
                  cap={f'{k}:{n}': v for (k, n), v in args.cap}, skipped=skipped, train_rows=len(source),
                  train_pairs_encoded=len(pairs), train_rejected=rejected, identities_encoded=len(identities),
                  identity_rejected=rejected_identity, identities_per_epoch=per_epoch, holdout_encoded=len(holdout),
                  holdout_rejected=rejected_holdout, dev_encoded=len(dev),
                  dev_rejected=rejected_dev, total_steps=total_steps, model_config=asdict(model.config))
    atomic_json(args.out / 'config.json', config)
    print(json.dumps({k: config[k] for k in ('parameters', 'train_pairs_encoded', 'identities_per_epoch', 'holdout_encoded', 'dev_encoded', 'total_steps')}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step, deadline, best = 0, start + args.max_hours * 3600, math.inf
    selection = holdout_source[:args.select_rows]
    def save(name='last.pt', **extra):
        checkpoint(args.out / name, dict(model=model.state_dict(), model_config=asdict(model.config),
                                         step=step, vocab_sha256=vocab_sha, **extra))
    with (args.out / 'metrics.jsonl').open('a', buffering=1) as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            rows = pairs + rng.sample(identities, min(per_epoch, len(identities)))
            running, n = 0., 0
            for batch_rows in batches(rows, args.batch_size, rng):
                lr = args.lr * min(1., (step + 1) / args.warmup_steps) * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(1., step / total_steps))))
                for group in optimizer.param_groups:
                    group['lr'] = lr
                batch = training_batch(batch_rows, tok, model.mask_id, rng, args.start_share)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
                    logits, cases = model(batch)
                value = loss(logits, cases, batch, args.copy_weight)
                if not torch.isfinite(value):
                    raise RuntimeError('nonfinite training loss')
                value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                step += 1
                running, n = running + value.item(), n + 1
            entry = dict(epoch=epoch, step=step, train_loss=round(running / n, 4), seconds=round(time.time() - start))
            entry['holdout_loss'] = round(evaluate(model, holdout, tok, args), 4)
            evaluating = epoch % args.eval_every == 0 or epoch == args.epochs or time.time() > deadline
            if args.select == 'loss' and entry['holdout_loss'] < best:
                best = entry['holdout_loss']
                save('best.pt', epoch=epoch, holdout_loss=best)
                entry['best'] = True
            if args.select == 'decode' and evaluating:
                counts, _ = decode_rows(model, tok, lex, selection)
                entry['holdout_decode'] = counts
                score = -(counts.get('exact', 0) - counts.get('clean_changed', 0) / 1000)
                if score < best:
                    best = score
                    save('best.pt', epoch=epoch, holdout_decode=counts)
                    entry['best'] = True
            if evaluating:
                entry['dev_loss'] = round(evaluate(model, dev, tok, args), 4)
                counts, _ = decode_rows(model, tok, lex, dev_source[:args.quick_decode])
                entry['quick_decode'] = counts
                save()
            log.write(json.dumps(entry) + '\n')
            print(json.dumps(entry), flush=True)
            status(state='training', **entry)
            if time.time() > deadline:
                break
    save()
    status(state='complete' if epoch == args.epochs else 'time_cap', epoch=epoch, step=step,
           best=best, seconds=round(time.time() - start))


if __name__ == '__main__':
    main()
