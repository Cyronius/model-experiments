"""Synthetic joint-split fixtures: no dictionary, downloads or paid services."""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_splits import build, validate, records, manifest, sha256, dump


def row(identity, src, ste, origin=None, docs=(), kind='original'):
    return {'id': identity, 'origin_id': origin or identity, 'src': src, 'ste': ste,
            'mode': 'PROC', 'kind': kind, 'noise': None, 'split': 'train', 'group': identity,
            'meaning_review': 'teacher_target_score5' if kind == 'original' else 'unjudged_rule',
            'target_review': 'checker_only' if kind == 'local_rule' else 'teacher_score5',
            'document_ids': list(docs), 'provenance': {'license': 'fixture'}, 'bucket': 64}


def corpus(path, items):
    path.mkdir()
    for split in ('train', 'dev', 'test'):
        (path / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in items if r['split'] == split))
    dump(path / 'summary.json', {'pairs': len(items)})
    return path


class JointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def inputs(self):
        teacher = [row('a', 'Unseal file.', 'Open the file.', origin='shared'),
                   row('b', 'Read file.', 'Read the file.', origin='shared'),
                   row('conflict', 'Clashing source.', 'First target.', docs=['bridge']),
                   row('bridge1', 'Side input.', 'Side target.', docs=['bridge'])]
        local = [row('dup', 'Unseal file.', 'Open the file.', docs=['local-doc'], kind='local_rule'),
                 row('c', 'Read more.', 'Read the file.', docs=['other-doc'], kind='local_rule'),
                 row('d', 'Independent source.', 'Independent target.', docs=['local-doc'], kind='local_rule'),
                 row('conflict2', 'Clashing source.', 'Second target.', docs=['far-doc'], kind='local_rule'),
                 row('bridge2', 'Far input.', 'Far target.', docs=['far-doc'], kind='local_rule')]
        return [('teacher', corpus(self.root / 'teacher', teacher)), ('local', corpus(self.root / 'local', local))]

    def test_joint_transitive_links_dedup_reviews_conflicts_and_reproducibility(self):
        inputs = self.inputs()
        a, b = self.root / 'a', self.root / 'b'
        summary = build(inputs, a, shard_size=2)
        self.assertTrue(validate(inputs, a)['passed'])
        self.assertEqual(summary['pairs'], 6)
        self.assertEqual(summary['duplicates_merged'], 1)
        self.assertEqual(summary['conflicting_records'], 2)
        output = [r for item in manifest(a)['shards'] for r in records(a / item['path'])]
        connected = [r for r in output if r['src'] not in ('Side input.', 'Far input.')]
        self.assertEqual(len({r['group'] for r in connected}), 1)
        bridge = [r for r in output if r['src'] in ('Side input.', 'Far input.')]
        self.assertEqual(len({r['group'] for r in bridge}), 1)
        duplicate = next(r for r in output if r['src'] == 'Unseal file.')
        self.assertEqual(duplicate['meaning_review'], 'teacher_target_score5')
        self.assertEqual({m['record']['meaning_review'] for m in duplicate['joint_members']}, {'teacher_target_score5', 'unjudged_rule'})
        repeat = build(list(reversed(inputs)), b, shard_size=2)
        self.assertEqual(summary, repeat)
        self.assertEqual((a / 'summary.json').read_bytes(), (b / 'summary.json').read_bytes())

    def mutate(self, output, callback):
        summary = json.loads((output / 'summary.json').read_text())
        for shard in summary['shards']:
            path = output / shard['path']
            items = list(records(path))
            for item in items:
                callback(item)
            with gzip.open(path, 'wt') as stream:
                for item in items:
                    stream.write(json.dumps(item) + '\n')
            shard['sha256'] = sha256(path)
        dump(output / 'summary.json', summary)

    def test_audit_catches_leakage_through_discarded_conflicts(self):
        inputs = self.inputs()
        output = self.root / 'out'
        build(inputs, output)
        self.mutate(output, lambda r: r.update(group='corrupt') if r['src'] == 'Far input.' else None)
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate(inputs, output)

    def test_audit_catches_review_and_member_tampering(self):
        inputs = self.inputs()
        output = self.root / 'out'
        build(inputs, output)
        self.mutate(output, lambda r: r['joint_members'][0]['record'].update(meaning_review='gold'))
        with self.assertRaisesRegex(ValueError, 'review'):
            validate(inputs, output)

    def test_audit_catches_changed_canonical_alignment(self):
        inputs = self.inputs()
        output = self.root / 'out'
        build(inputs, output)
        self.mutate(output, lambda r: r.update(source_canvas=['corrupt']))
        with self.assertRaisesRegex(ValueError, 'payload'):
            validate(inputs, output)

    def test_origin_ids_are_namespaced(self):
        inputs = [('teacher', corpus(self.root / 't', [row('a', 'One source.', 'One target.', origin='same')])),
                  ('local', corpus(self.root / 'l', [row('b', 'Two source.', 'Two target.', origin='same')]))]
        summary = build(inputs, self.root / 'out')
        self.assertEqual(summary['components'], 2)
        self.assertTrue(validate(inputs, self.root / 'out')['passed'])

    def test_holdouts_and_cross_mode_text_grouping(self):
        items = [row(str(i), f'Source {i}.', f'Target {i}.') for i in range(200)]
        extra = row('extra', '  SOURCE 0. ', 'Other target.')
        extra['mode'] = 'DESC'
        items.append(extra)
        inputs = [('teacher', corpus(self.root / 't', items))]
        summary = build(inputs, self.root / 'out', shard_size=19)
        self.assertTrue(all(summary['splits'][s] > 0 for s in ('train', 'dev', 'test')))
        self.assertTrue(validate(inputs, self.root / 'out')['passed'])
        emitted = [r for item in manifest(self.root / 'out')['shards'] for r in records(self.root / 'out' / item['path'])]
        connected = [r for r in emitted if r['src'].strip().casefold() == 'source 0.']
        self.assertEqual(len({r['group'] for r in connected}), 1)

    def test_manifest_hash_traversal_and_overwrite_fail_closed(self):
        inputs = self.inputs()
        output = self.root / 'out'
        build(inputs, output)
        with self.assertRaisesRegex(ValueError, 'empty'):
            build(inputs, output)
        shard = output / manifest(output)['shards'][0]['path']
        with shard.open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'hash'):
            validate(inputs, output)
        dump(self.root / 'local' / 'summary.json', {'shards': [{'path': '../teacher/train.jsonl'}]})
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            manifest(self.root / 'local')


if __name__ == '__main__':
    unittest.main()
