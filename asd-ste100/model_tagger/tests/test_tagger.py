"""Offline tag/action/gradient tests with a synthetic lexicon."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tagger import (Config, Tagger, Tags, SteTokenizer, EncodingError, Lexicon, collate,
                    source_units, target_units, supervision, loss, rewrite)
from ste_tokenizer import build_vocabulary


class TaggerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        lex = root / 'lex'
        lex.mkdir()
        rows = [{'lemma': w, 'forms': [w], 'status': 'approved', 'pos': 'n', 'alternatives': []}
                for w in ('use', 'open', 'file', 'the', 'and', 'set', 'to')]
        rows.append({'lemma': 'utilize', 'forms': ['utilize'], 'status': 'rejected',
                     'pos': 'v', 'alternatives': [{'lemma': 'use', 'pos': 'v'}]})
        (lex / 'lexicon.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        build_vocabulary(lex, ['Utilize file. Use fooBar and `x = 10`.'], root / 'vocab', 300)
        cls.tok, cls.lex = SteTokenizer(root / 'vocab'), Lexicon(lex)
        cls.tags = Tags(cls.tok)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def pair(self, src, target, probability=1):
        return dict(self.tok.encode_pair({'src': src, 'ste': target, 'mode': 'PROC'}, probability), src=src, mode='PROC')

    def test_supervised_edit_passes_reconstruct_targets(self):
        for src, target in [('Utilize the file.', 'Use the file.'), ('Open file.', 'Open the file.'),
                            ('File.', 'Use the file.'), ('Use the file.', 'Use file.'),
                            ('Use the file.', 'USE the file.')]:
            row = self.pair(src, target)
            units = source_units(row)
            targets = target_units(row, self.tok)
            for _ in range(5):
                labels, cases = supervision(units, targets, self.tags)
                units = self.tags.apply(units, labels, cases, row['slots'])
                if units == targets:
                    break
            self.assertEqual(units, targets, (src, target))

    def test_raw_bpe_copy_spans_are_atomic_and_slot_counts_are_hard_gated(self):
        row = self.pair('Utilize fooBar and fooBar.', 'Use fooBar and fooBar.', 0)
        self.assertGreater(len(row['source_ids']), len(row['source_spans']))
        units = source_units(row)
        labels, cases = supervision(units, target_units(row, self.tok), self.tags)
        output = self.tags.apply(units, labels, cases, row['slots'])
        self.assertEqual(self.tok.detokenize([v for v,c in output], row['slots'], [c for v,c in output]), 'Use fooBar and fooBar.')
        index = next(i for i, (v, c) in enumerate(units) if v in self.tok.slot_ids)
        labels[index + 1] = self.tags.DELETE
        with self.assertRaisesRegex(EncodingError, 'slot_fidelity'):
            self.tags.apply(units, labels, cases, row['slots'])

    def test_masks_controls_inactive_slots_and_encoder_only_keep(self):
        row = self.pair('Utilize file.', 'Use file.')
        mask = self.tags.mask(source_units(row), row['slots'])
        self.assertFalse(mask[1, self.tags.KEEP])
        for value in self.tok.control_ids | self.tok.slot_ids:
            self.assertFalse(mask[:, self.tags.replace(value)].any())
            self.assertFalse(mask[:, self.tags.append(value)].any())
        self.assertFalse(mask[0, self.tags.DELETE])

    def test_transform_outputs_are_closed_and_merge_consumes_two_units(self):
        units = [(self.tok.output_ids['file'], 0)]
        self.assertEqual(self.tags.transformed(units, 0, 'ADD_S'), (self.tok.output_ids['files'], 0))
        self.assertIsNone(self.tags.transformed(units, 0, 'ADD_ING'))
        out = self.tags.apply(units, [0, self.tags.transform('TITLE')], [0,0], [])
        self.assertEqual(out, [(self.tok.output_ids['file'], 1)])
        units = [(self.tok.output_ids[w], 0) for w in ('in', 'to')]
        out = self.tags.apply(units, [0, self.tags.transform('MERGE_NEXT'), self.tags.KEEP], [0,0,0], [])
        self.assertEqual(out, [(self.tok.output_ids['into'], 0)])
        labels, cases = supervision(units, out, self.tags)
        self.assertEqual(labels[1], self.tags.transform('MERGE_NEXT'))
        self.assertEqual(self.tags.apply(units, labels, cases, []), out)

    def test_forward_loss_backward_consumes_case_slot_and_checker_features(self):
        row = self.pair('Utilize fooBar.', 'Use fooBar.', 0)
        batch = collate([row], self.tok, self.lex)
        self.assertTrue(batch['refs'].gt(0).any())
        self.assertTrue(batch['features'].gt(0).any())
        labels, cases = supervision(source_units(row), target_units(row, self.tok), self.tags)
        model = Tagger(self.tok, Config(width=32, layers=1, heads=4, ffn=64, dropout=0))
        logits, case_logits = model(batch)
        objective = loss(logits, case_logits, torch.tensor([labels]), torch.tensor([cases]), batch['actions'], self.tags)
        self.assertTrue(torch.isfinite(objective))
        objective.backward()
        for parameter in (model.token.weight, model.case.weight, model.slot_ref.weight, model.violation.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_decode_rechecks_slots_and_reports_unresolved(self):
        tok, tags = self.tok, self.tags
        class Fixed(nn.Module):
            def __init__(self, drop):
                super().__init__()
                self.placeholder = nn.Parameter(torch.zeros(1))
                self.drop = drop
            def forward(self, batch):
                logits = torch.full(batch['legal'].shape, -100.)
                logits[:,:,tags.KEEP] = 0
                logits[:,1,tags.replace(tok.output_ids['use'])] = 100
                if self.drop:
                    logits[:,2,tags.DELETE] = 100
                cases = torch.zeros((*batch['legal'].shape[:2], 3))
                cases[:,1,1] = 10
                return logits.masked_fill(~batch['legal'], float('-inf')), cases
        result = rewrite(Fixed(False), tok, self.lex, 'Utilize fooBar.')
        self.assertTrue(result['resolved'])
        self.assertEqual(result['text'], 'Use fooBar.')
        result = rewrite(Fixed(True), tok, self.lex, 'Utilize fooBar.')
        self.assertFalse(result['resolved'])
        self.assertIsNone(result['text'])
        self.assertEqual(result['reason'], 'slot_fidelity')


if __name__ == '__main__':
    unittest.main()
