"""Offline correctness/safety tests; fixtures are hand-written, not dictionary extracts."""
import itertools
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_pairs import align, assign_splits, build, read_jsonl, write_jsonl
from noise_rules import Noiser
from text_utils import normalized, protected_values, tokenize


def lexicon(directory):
    directory.mkdir()
    rows = []
    for word, pos in [('use', 'v'), ('open', 'v'), ('close', 'v'), ('the', 'art'), ('a', 'art'),
                      ('file', 'n'), ('cause', 'n'), ('cause', 'v'), ('do', 'v'), ('you', 'pn')]:
        rows.append({'lemma': word, 'pos': pos, 'status': 'approved', 'forms': [word], 'alternatives': []})
    rows.append({'lemma': 'utilize', 'pos': 'v', 'status': 'rejected', 'forms': ['utilize'],
                 'alternatives': [{'lemma': 'use', 'pos': 'v'}]})
    for rejected, approved, pos in [('induce', 'cause', 'v'), ('conduct', 'do', 'v'), ('yourself', 'you', 'pn')]:
        rows.append({'lemma': rejected, 'pos': pos, 'status': 'rejected', 'forms': [rejected],
                     'alternatives': [{'lemma': approved, 'pos': pos}]})
    write_jsonl(directory / 'lexicon.jsonl', rows)
    write_jsonl(directory / 'pairs.jsonl', [{'ste': 'Open the gold file.', 'non_ste': 'Unseal the gold file.'}])


class TokenTests(unittest.TestCase):
    def test_lossless_offsets_and_protection(self):
        text = 'Use `a = 12` in src/main.py, C:\\work\\file.txt and https://host.test/x. Email a+b@host.test; set fooBar to 3.25%.'
        tokens = tokenize(text)
        self.assertTrue(all(text[t.start:t.end] == t.text for t in tokens))
        self.assertEqual(list(protected_values(text)), ['`a = 12`', 'src/main.py', 'C:\\work\\file.txt',
                                                       'https://host.test/x', 'a+b@host.test', 'fooBar', '3.25%'])
        self.assertEqual(tokens[-1].text, '.')

    def test_fenced_code_and_unicode(self):
        text = 'Read café.\n```python\nopen(the_file)\n```\nThen use αβ.'
        tokens = tokenize(text)
        self.assertIn('café', [t.text for t in tokens])
        self.assertEqual(sum(t.protected for t in tokens), 1)
        self.assertIn('αβ', [t.text for t in tokens])


class AlignmentTests(unittest.TestCase):
    def test_exhaustive_short_edits_reconstruct_both_sequences(self):
        sequences = [list(seq) for n in range(4) for seq in itertools.product('ab', repeat=n)]
        for source in sequences:
            for target in sequences:
                edits, positions, required = align(source, target)
                self.assertEqual(len(positions), len(target))
                self.assertEqual(positions, sorted(set(positions)))
                self.assertGreaterEqual(required, 2 * len(source))
                i = j = cost = 0
                rebuilt = []
                for operation, si, tj in edits:
                    self.assertEqual((si, tj), (i, j))
                    if operation in {'keep', 'replace'}:
                        self.assertEqual(operation == 'keep', source[i] == target[j])
                        rebuilt.append(target[j]); i += 1; j += 1
                    elif operation == 'delete':
                        i += 1
                    else:
                        rebuilt.append(target[j]); j += 1
                    cost += operation != 'keep'
                self.assertEqual((i, j), (len(source), len(target)))
                self.assertEqual(rebuilt, target)
                # Independent recursion for the optimum, on tiny sequences.
                def optimum(a, b):
                    if not a or not b:
                        return len(a) + len(b)
                    return min(optimum(a[1:], b[1:]) + (a[0] != b[0]),
                               optimum(a[1:], b) + 1, optimum(a, b[1:]) + 1)
                self.assertEqual(cost, optimum(source, target))
                canvas = [None] * required
                for position, value in zip(positions, target):
                    canvas[position] = value
                self.assertEqual([t for t in canvas if t is not None], target)

    def test_long_insertions_promote_bucket(self):
        source = ['a'] * 32
        target = ['a'] * 31 + ['b'] * 20 + ['a']
        _, positions, required = align(source, target)
        self.assertGreater(required, 64)
        self.assertLessEqual(required, 128)
        self.assertEqual(len(positions), len(target))


class NoiseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name) / 'lex'
        lexicon(self.directory)
        self.noiser = Noiser(self.directory)

    def tearDown(self):
        self.temp.cleanup()

    def test_coverage_and_protected_content(self):
        examples = [('Use the file.', 'PROC'), ('Open the file. Close the file.', 'PROC'),
                    ('They closed the file.', 'DESC'), ('Open `the_file = 12`.', 'PROC')]
        kinds = set()
        for text, mode in examples:
            variants = self.noiser.variants(text, mode, 100, 7)
            for row in variants:
                self.assertEqual(protected_values(text), protected_values(row['src']))
                self.assertNotEqual(normalized(text), normalized(row['src']))
                kinds.add(row['noise'])
            self.assertEqual(variants, self.noiser.variants(text, mode, 100, 7))
        self.assertTrue({'reverse_dictionary', 'drop_article', 'merge_sentences', 'nominalize',
                         'perfect_tense', 'passive', 'verbose'} <= kinds)

    def test_negative_conditional_and_code_are_not_voice_transformed(self):
        for text in ('Do not open the file.', 'Open the file if it is safe.',
                     'Open the file and close the door.', '`Open the file.`'):
            variants = self.noiser.variants(text, 'PROC', 100)
            self.assertFalse({'passive', 'nominalize'} & {row['noise'] for row in variants})
        self.assertEqual(self.noiser.variants('```\nOpen the file.\n```', 'PROC', 10), [])

    def test_auxiliary_pronoun_and_ambiguous_pos_are_not_substituted(self):
        for text in ('Do not open the file.', 'You use the file.', 'The cause is unknown.'):
            self.assertNotIn('reverse_dictionary', {r['noise'] for r in self.noiser.variants(text, 'PROC', 100)})

    def test_voice_changes_work_within_a_document(self):
        variants = self.noiser.variants('Read the file. Open the file.', 'PROC', 100)
        self.assertIn('Read the file. The file should be opened.', [r['src'] for r in variants])


class SplitTests(unittest.TestCase):
    def test_transitive_overlap_stays_in_one_split(self):
        rows = [{'origin_id': '1', 'src': 'a', 'ste': 'b'},
                {'origin_id': '2', 'src': 'b', 'ste': 'c'},
                {'origin_id': '3', 'src': 'd', 'ste': 'c'},
                {'origin_id': '3', 'src': 'e', 'ste': 'f'}]
        assign_splits(rows, 42, .2, .2)
        self.assertEqual(len({r['group'] for r in rows}), 1)
        expected = {r['src']: r['split'] for r in rows}
        random.Random(4).shuffle(rows)
        assign_splits(rows, 42, .2, .2)
        self.assertEqual(expected, {r['src']: r['split'] for r in rows})


class BuilderTests(unittest.TestCase):
    def test_filters_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            lexicon(base / 'lex')
            run = base / 'run'; run.mkdir()
            def row(key, src, ste, **extra):
                return {'id': key, 'src': src, 'ste': ste, 'mode': 'PROC', 'score': 5, 'changed': False, **extra}
            judged = [row('ok', 'Please use the file.', 'Use the file.'),
                      row('changed', 'Open the file.', 'Close the file.', changed=True),
                      row('low', 'Open the file.', 'Close the file.', score=4),
                      row('gold', 'Please open the gold file.', 'Open the gold file.'),
                      row('bad', 'Use the file.', 'Utilize the file.'),
                      row('number', 'Open file12.', 'Open file13.'),
                      row('same', 'Close the file.', 'Close the file.'),
                      row('conflict1', 'Do this.', 'Open a file.'),
                      row('conflict2', 'Do this.', 'Close a file.')]
            messy = [{'id': 'ok', 'ste': 'Use the file.', 'mode': 'PROC', 'messy': 'Use the file.'},
                     {'id': 'ok', 'ste': 'Use the file.', 'mode': 'PROC', 'messy': 'Style: Use the file.'},
                     {'id': 'ok', 'ste': 'Use the file.', 'mode': 'PROC', 'messy': 'Use `invented`.'},
                     {'id': 'ok', 'ste': 'Close the file.', 'mode': 'PROC', 'messy': 'Close it.'},
                     {'id': 'ok', 'ste': 'Use the file.', 'mode': 'PROC', 'messy': 'Please use the file.'}]
            write_jsonl(run / 'judged.jsonl', judged); write_jsonl(run / 'messy.jsonl', messy)
            summary = build(run, base / 'out', 0, lex_dir=base / 'lex')
            counts = summary['rejections']
            for key in ('target.judge_not_faithful', 'target.gold_overlap', 'target.checker',
                        'original.protected_mismatch', 'original.unchanged', 'original.conflicting_target',
                        'messy.unchanged', 'messy.preamble', 'messy.protected_mismatch',
                        'messy.target_mismatch', 'messy.duplicate'):
                self.assertGreater(counts['rejected.' + key], 0, key)
            rows = sum((read_jsonl(base / 'out' / f'{s}.jsonl') for s in ('train', 'dev', 'test')), [])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['kind'], 'original')
            for row in rows:
                self.assertEqual([t for t in row['source_canvas'] if t is not None], row['src_tokens'])
                self.assertEqual([t for t in row['target_canvas'] if t is not None], row['tgt_tokens'])
            # Input order is irrelevant, including the order of back-translation rows.
            write_jsonl(run / 'judged.jsonl', list(reversed(judged)))
            write_jsonl(run / 'messy.jsonl', list(reversed(messy)))
            build(run, base / 'again', 0, lex_dir=base / 'lex')
            for name in ('train.jsonl', 'dev.jsonl', 'test.jsonl', 'rejected.jsonl'):
                self.assertEqual((base / 'out' / name).read_bytes(), (base / 'again' / name).read_bytes())


if __name__ == '__main__':
    unittest.main()
