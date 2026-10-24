"""Synthetic fixtures; no dictionary-derived text is checked into these tests."""
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ste_tokenizer import SteTokenizer, EncodingError, build_vocabulary, model_tokens, align_ids, align
from prepare import export, validate


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        lex = cls.root / 'lex'
        lex.mkdir()
        rows = [{'lemma': w, 'forms': [w], 'status': 'approved', 'pos': 'v', 'alternatives': []}
                for w in ('use', 'open', 'the', 'file', 'and', 'set', 'to', 'a')]
        rows.append({'lemma': 'utilize', 'forms': ['utilize', 'utilized'], 'status': 'rejected',
                     'pos': 'v', 'alternatives': [{'lemma': 'use', 'pos': 'v'}]})
        (lex / 'lexicon.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        cls.vocab = cls.root / 'vocab'
        build_vocabulary(lex, ['Utilize the file. Use `x = 10`.'], cls.vocab, 300)
        cls.tok = SteTokenizer(cls.vocab)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def pair(self, src, ste, probability=1):
        return self.tok.encode_pair({'src': src, 'ste': ste, 'mode': 'PROC'}, probability)

    def test_closed_head_and_rejected_inflections_input_only(self):
        self.assertIn('utilized', self.tok.input_ids)
        self.assertNotIn('utilized', self.tok.output_ids)
        row = self.pair('Utilize the file.', 'Use the file.')
        self.assertEqual(self.tok.detokenize(row['target_ids'], row['slots']), 'Use the file.')
        with self.assertRaisesRegex(EncodingError, 'target_outside_vocabulary'):
            self.pair('Use the file.', 'Utilize the file.')

    def test_function_words_and_regular_rejected_forms(self):
        self.assertIn('he', self.tok.output_ids)
        self.assertIn('me', self.tok.output_ids)
        self.assertIn('utilizing', self.tok.input_ids)
        self.assertNotIn('utilizing', self.tok.output_ids)
        row = self.pair('He utilized the file.', 'He can use the file.')
        self.assertEqual(row['slots'], [])

    def test_first_capital_and_mixed_case(self):
        # Hyphenated sentence case is different from Python's str.title().
        from ste_tokenizer import case_id
        self.assertEqual(case_id('De-energize'), 1)
        row = self.pair('Use fILe.', 'Use fILe.')
        self.assertEqual(row['slots'][0]['surface'], 'fILe')

    def test_names_units_and_signed_numbers(self):
        src = 'Use United Filter Corporation at -10 °C and 12 in. Use 5 kg.'
        row = self.pair(src, src)
        self.assertIn({'token': 'TN_1', 'surface': 'United Filter Corporation', 'count': 1}, row['slots'])
        surfaces = [slot['surface'] for slot in row['slots']]
        for value in ('-10 °C', '12 in', '5 kg'):
            self.assertIn(value, surfaces)
        with self.assertRaises(EncodingError):
            self.pair('Set to 5 kg.', 'Set to 5 lb.')

    def test_trimmed_alignment_remains_optimal(self):
        import itertools
        sequences = [list(seq) for n in range(4) for seq in itertools.product(range(2), repeat=n)]
        for source in sequences:
            for target in sequences:
                actual, positions, required = align_ids(source, target)
                expected, _, _ = align(source, target)
                self.assertEqual(sum(op != 'keep' for op, _, _ in actual),
                                 sum(op != 'keep' for op, _, _ in expected))
                self.assertEqual(len(positions), len(target))
                self.assertGreaterEqual(required, 2 * len(source))

    def test_protected_roundtrip_and_repeated_slots(self):
        src = 'Use `x = 10`, https://example.test/a, fooBar, 3.25% and café. Use fooBar.'
        row = self.pair(src, src)
        self.assertEqual(len(row['slots']), 5)
        self.assertIn({'token': 'NUM_1', 'surface': '3.25%', 'count': 1}, row['slots'])
        self.assertEqual(self.tok.restore_tokens(row['target_ids'], row['slots'], row['target_case']),
                         [t.text for t in model_tokens(src)])
        self.assertEqual(Counter(s['surface'] for s in row['slots'])['fooBar'], 1)

    def test_drop_duplicate_and_invent_slot_fail(self):
        for src, tgt in [('Use fooBar.', 'Use the file.'), ('Use fooBar.', 'Use fooBar and fooBar.'),
                         ('Use the file.', 'Use fooBar.')]:
            with self.assertRaises(EncodingError):
                self.pair(src, tgt)
        row = self.pair('Use fooBar.', 'Use fooBar.')
        with self.assertRaisesRegex(EncodingError, 'inactive_slot'):
            self.tok.restore_tokens([self.tok.output_ids['PROTECT_2']], row['slots'])

    def test_input_byte_fallback_unicode_and_raw_slot_reference(self):
        for text in ('café', '变量', '🛠️', '```python\nfoo_bar = 3\n```'):
            ids = self.tok.fallback(text)
            self.assertEqual(self.tok.bpe.decode([i - self.tok.bpe_offset for i in ids]), text)
        row = self.pair('Use fooBar.', 'Use fooBar.', probability=0)
        self.assertTrue(any(i >= self.tok.bpe_offset for i in row['source_ids']))
        self.assertTrue(all(ref == self.tok.output_ids['PROTECT_1'] for i, ref in
                            zip(row['source_ids'], row['source_slot_refs']) if i >= self.tok.bpe_offset))
        self.assertEqual(row, self.pair('Use fooBar.', 'Use fooBar.', probability=0))

    def test_unknown_target_cannot_be_learned_as_new_word(self):
        with self.assertRaisesRegex(EncodingError, 'target_unseen_slot'):
            self.pair('Use the file.', 'Use café.')

    def test_slot_capacity_and_uppercase_variable(self):
        row = self.pair('Set A to 12.', 'Set A to 12.')
        self.assertEqual([s['surface'] for s in row['slots']], ['A', '12'])
        with self.assertRaisesRegex(EncodingError, 'slot_overflow_PROTECT'):
            self.tok.encode_source(' '.join(f'var_{i}' for i in range(17)))

    def test_bucket_rechecked_after_raw_fallback(self):
        src = ' '.join(['Use'] * 20 + ['`' + 'z' * 40 + '`'])
        row = self.pair(src, src, probability=0)
        self.assertGreater(row['bucket'], 64)
        with self.assertRaisesRegex(EncodingError, 'source_bucket_overflow'):
            self.tok.encode_source('`' + 'z' * 300 + '`', slot_probability=0)

    def test_mask_and_invalid_decode(self):
        row = self.pair('Use fooBar.', 'Use fooBar.')
        mask = self.tok.additive_mask(row['slots'])
        for name in ('PAD', 'SEP', 'PROC', 'DESC', 'PROTECT_2', 'NUM_1'):
            self.assertEqual(mask[self.tok.output_ids[name]], float('-inf'))
        for name in ('NULL', 'use', 'PROTECT_1'):
            self.assertEqual(mask[self.tok.output_ids[name]], 0)
        for value in (len(self.tok.output), -1, self.tok.output_ids['PAD']):
            with self.assertRaises(EncodingError):
                self.tok.restore_tokens([value], [])

    def test_export_audit_and_deterministic_shards(self):
        pairs = self.root / 'pairs'
        pairs.mkdir(exist_ok=True)
        (pairs / 'summary.json').write_text('{}', encoding='utf-8')
        for split in ('train', 'dev', 'test'):
            row = {'id': split, 'origin_id': split, 'group': split, 'kind': 'original',
                   'split': split, 'src': 'Utilize fooBar.', 'ste': 'Use fooBar.',
                   'mode': 'PROC', 'bucket': 64, 'meaning_review': 'teacher_target_score5'}
            (pairs / f'{split}.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
        outputs = [self.root / f'ids-{i}' for i in range(2)]
        results = [export(pairs, self.vocab, output) for output in outputs]
        self.assertTrue(validate(outputs[0], self.vocab)['passed'])
        self.assertEqual(results[0]['shards'], results[1]['shards'])
        # The audit catches corruption even with an updated shard checksum.
        import gzip
        first = outputs[0] / results[0]['shards'][0]['path']
        with gzip.open(first, 'rt', encoding='utf-8') as stream:
            row = json.loads(stream.readline())
        row['source_slot_refs'][0] = self.tok.output_ids['PROTECT_1']
        with gzip.open(first, 'wt', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        from ste_tokenizer import sha256
        results[0]['shards'][0]['sha256'] = sha256(first)
        (outputs[0] / 'summary.json').write_text(json.dumps(results[0]), encoding='utf-8')
        with self.assertRaises(AssertionError):
            validate(outputs[0], self.vocab)


if __name__ == '__main__':
    unittest.main()
