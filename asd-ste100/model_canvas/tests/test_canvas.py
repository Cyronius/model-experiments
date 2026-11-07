"""Canvas layout, corruption, masking and an end-to-end memorization check, with a synthetic lexicon."""
import gzip
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canvas import (CanvasEditor, Config, SteTokenizer, Lexicon, SLACK, MASK_CASE, source_canvas,
                    encode_example, encode_source, collate, corrupt, training_batch, loss, rewrite,
                    fit_slots, unfreeze_repeats)
from ste_tokenizer import build_vocabulary
from train import article_before_literal, train_rows


class CanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        lex = root / 'lex'
        lex.mkdir()
        rows = [{'lemma': w, 'forms': [w], 'status': 'approved', 'pos': 'n', 'alternatives': []}
                for w in ('use', 'open', 'file', 'the', 'and', 'set', 'to', 'it', 'is')]
        rows.append({'lemma': 'utilize', 'forms': ['utilize'], 'status': 'rejected',
                     'pos': 'v', 'alternatives': [{'lemma': 'use', 'pos': 'v'}]})
        (lex / 'lexicon.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        build_vocabulary(lex, ['Utilize file. Use fooBar and `x = 10`.'], root / 'vocab', 300)
        cls.tok, cls.lex = SteTokenizer(root / 'vocab'), Lexicon(lex)
        cls.row = encode_example(cls.tok, cls.lex, 'Utilize the file and open `x`.', 'Use the file. Open `x`.', 'PROC')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def collapse(self, ids, cases, row):
        null = self.tok.output_ids['NULL']
        values = [(v, c) for v, c in zip(ids, cases) if v != null]
        return self.tok.detokenize([v for v, _ in values], row['slots'], [c for _, c in values])

    def test_canvases_place_source_on_even_slots_and_collapse_to_the_reference(self):
        row, null = self.row, self.tok.output_ids['NULL']
        ids, _ = source_canvas(row, self.tok)
        self.assertEqual(len(ids), 2 * len(row['source_ids']) + SLACK)
        self.assertEqual(ids[0::2][:len(row['source_ids'])], row['source_ids'])
        self.assertTrue(all(v == null for v in ids[1::2]))
        self.assertEqual(len(row['target']), len(ids))
        self.assertEqual(self.collapse(row['target'], row['target_case'], row), 'Use the file. Open `x`.')

    def test_corruption_shows_target_where_not_corrupted_and_start_share_gives_the_source(self):
        mask = self.tok.meta['input_size']
        source, _ = source_canvas(self.row, self.tok)
        ids, _, corrupted = corrupt(self.row, self.tok, mask, random.Random(1), start_share=1.)
        self.assertEqual(ids, source)
        self.assertTrue(all(corrupted))
        for seed in range(20):
            ids, cases, corrupted = corrupt(self.row, self.tok, mask, random.Random(seed), start_share=0.)
            self.assertTrue(any(corrupted))
            for p, hit in enumerate(corrupted):
                if not hit:
                    self.assertEqual(ids[p], self.row['target'][p])
                else:
                    self.assertIn(ids[p], (mask, source[p]))
                    self.assertEqual(cases[p] == MASK_CASE, ids[p] == mask)

    def test_output_excludes_controls_and_inactive_slots_but_allows_null(self):
        model = CanvasEditor(self.tok, Config(32, 1, 2, 64))
        row = encode_source(self.tok, self.lex, 'Use the file.', 'PROC')
        logits, cases = model(collate([row], [source_canvas(row, self.tok)], self.tok, model.mask_id))
        self.assertEqual(logits.shape[:2], (1, 2 * len(row['source_ids']) + SLACK))
        self.assertEqual(cases.shape[-1], 3)
        out = self.tok.output_ids
        self.assertTrue(torch.isfinite(logits[0, 0, out['NULL']]))
        for name in ('PAD', 'SEP', 'PROC', 'DESC', 'PROTECT_1', 'TN_1'):
            self.assertTrue(torch.isinf(logits[0, 0, out[name]]))

    def test_loss_counts_only_corrupted_slots(self):
        model = CanvasEditor(self.tok, Config(32, 1, 2, 64))
        batch = training_batch([self.row], self.tok, model.mask_id, random.Random(3), start_share=0.)
        logits, cases = model(batch)
        value = loss(logits, cases, batch)
        logits.retain_grad()
        value.backward()
        untouched = ~batch['selected'][0]
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(logits.grad[0, untouched].abs().sum().eq(0))

    def test_fit_slots_gives_each_slot_its_source_count(self):
        out, row = self.tok.output_ids, self.row
        slot = out[row['slots'][0]['token']]
        null, word = out['NULL'], out['file']
        ids, cases = [slot, null, slot, null, word, null], [0] * 6
        probs = torch.zeros((6, self.tok.meta['output_size']))
        probs[0, slot], probs[0, word], probs[2, slot], probs[2, word] = .4, .3, .9, .1
        fit_slots(ids, cases, probs, row['slots'], self.tok)
        self.assertEqual(ids, [word, null, slot, null, word, null])
        ids = [null, word, null, word]
        probs = torch.zeros((4, self.tok.meta['output_size']))
        probs[3, slot] = .2
        fit_slots(ids, [0] * 4, probs, row['slots'], self.tok)
        self.assertEqual(ids, [null, word, null, slot])

    def test_adjacent_repeats_are_unfrozen_at_the_less_confident_copy(self):
        out, mask = self.tok.output_ids, self.tok.meta['input_size']
        null, word = out['NULL'], out['the']
        ids, frozen, cases = [word, null, word, null, out['file']], [True] * 5, [0] * 5
        unfreeze_repeats(ids, frozen, [.9, 1., .6, 1., 1.], null, self.tok, mask, cases)
        self.assertEqual(ids, [word, null, mask, null, out['file']])
        self.assertEqual(frozen, [True, True, False, True, True])

    def test_a_small_model_memorizes_one_pair_and_rewrite_returns_it(self):
        torch.manual_seed(0)
        model = CanvasEditor(self.tok, Config(64, 2, 2, 128, dropout=0.))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        rng = random.Random(0)
        identity = encode_example(self.tok, self.lex, 'Use the file. Open `x`.', 'Use the file. Open `x`.', 'PROC')
        for _ in range(300):
            model.train()
            batch = training_batch([self.row, identity], self.tok, model.mask_id, rng, start_share=.5)
            logits, cases = model(batch)
            optimizer.zero_grad()
            loss(logits, cases, batch, copy_weight=1.).backward()
            optimizer.step()
        result = rewrite(model, self.tok, self.lex, 'Utilize the file and open `x`.', 'PROC')
        self.assertEqual(result['text'], 'Use the file. Open `x`.')
        self.assertEqual(rewrite(model, self.tok, self.lex, 'Use the file. Open `x`.', 'PROC')['text'],
                         'Use the file. Open `x`.')

    def test_pretraining_rows_skip_articles_before_literals_and_apply_caps(self):
        rule = lambda src, ste, noise='drop_article': dict(kind='local_rule', noise=noise, src=src, ste=ste)
        self.assertTrue(article_before_literal(rule('Set `x` to 1.', 'Set the `x` to 1.')))
        self.assertFalse(article_before_literal(rule('Open file `x`.', 'Open the file `x`.')))
        self.assertFalse(article_before_literal(rule('Set `x`.', 'Set the `x`.', 'verbose')))
        rows = [rule(f'Open file {i}.', f'Open the file {i}.') for i in range(10)]
        rows += [rule('Set `x` to 1.', 'Set the `x` to 1.'), rule('A. B.', 'A and B.', 'merge_sentences'),
                 dict(kind='doc_teacher', src='a', ste='b')]
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'shards').mkdir()
            with gzip.open(Path(folder) / 'shards/train-00000.jsonl.gz', 'wt') as f:
                f.write(''.join(json.dumps(r) + '\n' for r in rows))
            skipped = {}
            kept = train_rows(Path(folder), {'local_rule'}, {('local_rule', 'drop_article'): 4}, 0, skipped)
        self.assertEqual(sorted(r['noise'] for r in kept), ['drop_article'] * 4 + ['merge_sentences'])
        self.assertEqual(skipped, {'article_before_literal': 1, 'cap local_rule:drop_article': 6})


if __name__ == '__main__':
    unittest.main()
