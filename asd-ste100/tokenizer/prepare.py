"""Fit the tokenizer, export ID pairs, or independently audit an export locally."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import io
import json
from pathlib import Path

from ste_tokenizer import ROOT, SteTokenizer, EncodingError, build_vocabulary, sha256, model_tokens


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def input_paths(root):
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    if 'shards' in summary:
        paths = [(root / item['path'].replace('\\', '/'), item['sha256']) for item in summary['shards']]
    else:
        paths = [(root / f'{split}.jsonl', None) for split in ('train', 'dev', 'test')]
    for path, expected in paths:
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('input manifest path escapes corpus directory')
        actual = sha256(path)
        if expected and actual != expected:
            raise ValueError(f'input hash mismatch: {path}')
        yield path, actual


def records(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def fit(pairs, output, bpe_size):
    paths = list(input_paths(pairs))
    def texts():
        for path, _ in paths:
            for row in records(path):
                if row['split'] == 'train':
                    yield row['src']
    meta = build_vocabulary(ROOT / 'lexicon', texts(), output, bpe_size)
    dump(output / 'fit.json', {'inputs': {str(p): h for p, h in paths},
                              'output_size': meta['output_size'], 'input_size': meta['input_size'],
                              'bpe_size': meta['input_size'] - meta['bpe_offset'],
                              'fit_scope': meta['fit_scope']})
    return json.loads((output / 'fit.json').read_text(encoding='utf-8'))


WORKER = None


def initialize(vocab):
    global WORKER
    WORKER = SteTokenizer(Path(vocab))


def encode_batch(args):
    batch, probability, seed = args
    result = []
    for row in batch:
        try:
            encoded = WORKER.encode_pair(row, probability, seed)
            # Keep review labels and full attribution, replace surface alignment fields.
            skip = {'schema', 'src_tokens', 'tgt_tokens', 'src_offsets', 'tgt_offsets',
                    'protected_src', 'protected_tgt', 'edits', 'target_positions',
                    'bucket', 'source_canvas', 'target_canvas'}
            item = {key: value for key, value in row.items() if key not in skip}
            item.update(encoded, schema='ste-id-pairs-v1', surface_bucket=row['bucket'])
            result.append((item, None))
        except EncodingError as error:
            result.append(({'id': row['id'], 'split': row['split'], 'kind': row['kind']}, str(error)))
    return result


def export(pairs, vocab, output, probability=.7, seed=0, workers=1, limit=None, shard_size=25000):
    if not 0 <= probability <= 1 or workers < 1 or shard_size < 1 or (limit is not None and limit < 1):
        raise ValueError('invalid export parameters')
    inputs = list(input_paths(pairs))
    fingerprints = {name: sha256(Path(__file__).parent / name) for name in ('prepare.py', 'ste_tokenizer.py')}
    if output.resolve() in (pairs.resolve(), vocab.resolve()):
        raise ValueError('output must differ from input/vocabulary directories')
    if output.exists() and any(output.iterdir()):
        raise ValueError('use a new or empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'shards').mkdir(exist_ok=True)
    counts, rejected, buckets, promotion = Counter(), Counter(), Counter(), Counter()
    streams, paths = {}, []
    scanned = 0

    def close(split):
        text, raw = streams.pop(split)
        text.close()
        raw.close()

    def batches():
        batch = []
        count = 0
        for path, _ in inputs:
            for row in records(path):
                batch.append(row)
                count += 1
                if len(batch) == 128:
                    yield batch, probability, seed
                    batch = []
                if limit is not None and count >= limit:
                    if batch:
                        yield batch, probability, seed
                    return
        if batch:
            yield batch, probability, seed

    with (output / 'rejected.jsonl').open('w', encoding='utf-8', newline='\n') as failures:
        def write(results):
            nonlocal scanned
            for row, reason in results:
                scanned += 1
                if reason:
                    rejected[reason] += 1
                    row['reason'] = reason
                    failures.write(json.dumps(row, ensure_ascii=False) + '\n')
                    continue
                split = row['split']
                if split not in ('train', 'dev', 'test'):
                    raise ValueError('invalid split')
                if counts[split] % shard_size == 0:
                    if split in streams:
                        close(split)
                    path = output / 'shards' / f'{split}-{counts[split] // shard_size:05d}.jsonl.gz'
                    paths.append(path)
                    raw = path.open('wb')
                    compressed = gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=1)
                    streams[split] = io.TextIOWrapper(compressed, encoding='utf-8', newline='\n'), raw
                streams[split][0].write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                counts[split] += 1
                buckets[str(row['bucket'])] += 1
                promotion[f'{row["surface_bucket"]}->{row["bucket"]}'] += 1
            if scanned % 25600 == 0:
                print(f'encoded {scanned:,}; retained {sum(counts.values()):,}', flush=True)
        try:
            if workers == 1:
                initialize(vocab)
                for batch in batches():
                    write(encode_batch(batch))
            else:
                iterator = iter(batches())
                with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(vocab,)) as pool:
                    while True:
                        wave = []
                        for _ in range(workers * 2):
                            batch = next(iterator, None)
                            if batch is None:
                                break
                            wave.append(pool.submit(encode_batch, batch))
                        if not wave:
                            break
                        for future in wave:
                            write(future.result())
        finally:
            for split in list(streams):
                close(split)
    summary = {'schema': 'ste-id-pairs-v1', 'scanned': scanned, 'pairs': sum(counts.values()),
               'splits': dict(counts), 'rejected': dict(rejected), 'buckets': dict(buckets),
               'bucket_transitions': dict(promotion), 'slot_probability': probability,
               'seed': seed, 'limit': limit, 'workers': workers,
               'inputs': {str(p): h for p, h in inputs},
               'vocab_sha256': sha256(vocab / 'vocab.json'),
               'code_sha256': fingerprints,
               'shards': [{'path': p.relative_to(output).as_posix(), 'sha256': sha256(p),
                           'bytes': p.stat().st_size} for p in paths]}
    dump(output / 'summary.json', summary)
    return summary


def validate(output, vocab):
    if not __debug__:
        raise RuntimeError('run the audit without Python -O; assertions are required')
    tokenizer = SteTokenizer(vocab)
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    if sha256(vocab / 'vocab.json') != summary['vocab_sha256']:
        raise ValueError('vocabulary fingerprint mismatch')
    counts, seen, groups = Counter(), set(), {}
    for path, _ in input_paths(output):
        for row in records(path):
            key, split = row['id'], row['split']
            assert key not in seen, ('duplicate', key)
            seen.add(key)
            assert groups.setdefault(row['group'], split) == split, ('group leak', key)
            source, target = row['source_ids'], row['target_ids']
            cases, refs = row['source_case'], row['source_slot_refs']
            assert len(source) == len(cases) == len(refs)
            assert len(target) == len(row['target_case']) == len(row['target_positions'])
            assert all(0 <= value < tokenizer.meta['input_size'] for value in source)
            assert all(0 <= value < 4 for value in cases)
            active = {tokenizer.output_ids[slot['token']] for slot in row['slots']}
            assert all(0 <= value < len(tokenizer.output) and value not in tokenizer.control_ids
                       and (value not in tokenizer.slot_ids or value in active) for value in target)
            restored = tokenizer.restore_tokens(target, row['slots'], row['target_case'])
            assert restored == [t.text for t in model_tokens(row['ste'])], ('target roundtrip', key)
            table = {tokenizer.output_ids[slot['token']]: slot for slot in row['slots']}
            assert len(table) == len(row['slots'])
            observed_slots = Counter()
            tokens = model_tokens(row['src'])
            assert len(row['source_spans']) == len(tokens)
            previous = 0
            for (start, end, left, right), token in zip(row['source_spans'], tokens):
                assert start == previous and start < end <= len(source)
                assert (left, right) == (token.start, token.end)
                pieces = source[start:end]
                if pieces[0] >= tokenizer.bpe_offset:
                    assert all(p >= tokenizer.bpe_offset for p in pieces)
                    text = tokenizer.bpe.decode([p - tokenizer.bpe_offset for p in pieces])
                elif pieces[0] in table:
                    assert len(pieces) == 1
                    text = table[pieces[0]]['surface']
                else:
                    assert len(pieces) == 1
                    text = tokenizer.meta['input_word_tokens'][pieces[0]]
                    case = cases[start]
                    text = text[:1].upper() + text[1:] if case == 1 else text.upper() if case == 2 else text
                assert text == token.text, ('source roundtrip', key, token.text)
                expected_ref = next((sid for sid, slot in table.items() if slot['surface'] == token.text), -1)
                assert refs[start:end] == [expected_ref] * (end - start)
                if expected_ref >= 0:
                    observed_slots[expected_ref] += 1
                previous = end
            assert previous == len(source)
            assert observed_slots == Counter({sid: slot['count'] for sid, slot in table.items()})
            positions = row['target_positions']
            assert positions == sorted(set(positions))
            required = max(2 * len(source), max(positions, default=-1) + 1)
            assert row['bucket'] == next(size for size in (64, 128, 256) if size >= required)
            # Independently execute the stored edit script to cover every input and target ID.
            i = j = 0
            for op, si, tj in row['edits']:
                assert (si, tj) == (i, j)
                if op == 'keep':
                    assert source[i] == target[j]
                    i += 1; j += 1
                elif op == 'replace':
                    i += 1; j += 1
                elif op == 'delete':
                    i += 1
                elif op == 'insert':
                    j += 1
                else:
                    raise ValueError('invalid edit')
            assert (i, j) == (len(source), len(target))
            counts[split] += 1
    assert dict(counts) == summary['splits']
    rejection_counts = Counter(row['reason'] for row in records(output / 'rejected.jsonl'))
    assert dict(rejection_counts) == summary['rejected']
    assert sum(counts.values()) + sum(rejection_counts.values()) == summary['scanned']
    result = {'passed': True, 'pairs': sum(counts.values()), 'splits': dict(counts),
              'vocab_sha256': summary['vocab_sha256'], 'summary_sha256': sha256(output / 'summary.json'),
              'validation_code_sha256': sha256(Path(__file__))}
    dump(output / 'validation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['fit', 'export', 'validate'])
    parser.add_argument('--pairs', type=Path)
    parser.add_argument('--vocab', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--bpe-size', type=int, default=4096)
    parser.add_argument('--slot-probability', type=float, default=.7)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.command == 'fit':
        result = fit(args.pairs, args.vocab, args.bpe_size)
    elif args.command == 'export':
        result = export(args.pairs, args.vocab, args.out, args.slot_probability, args.seed, args.workers, args.limit)
    else:
        result = validate(args.out, args.vocab)
    print(json.dumps(result, ensure_ascii=False, indent=2))
