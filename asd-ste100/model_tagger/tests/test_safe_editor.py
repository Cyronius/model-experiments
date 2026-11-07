import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safe_editor import Proposal, run_loop, validate
from ste_tokenizer import model_tokens


class SafeEditorTests(unittest.TestCase):
    def test_rejected_edit_is_not_next_input_and_feedback_changes_proposal(self):
        calls = []
        def propose(text, banned):
            calls.append((text, banned))
            return (Proposal('Open.', ((1, 1),)) if not banned else
                    Proposal('Open the file.', ((1, 2),)))
        result = run_loop('Open file.', propose, lambda _: True)
        self.assertEqual(result['text'], 'Open the file.')
        self.assertTrue(result['resolved'])
        self.assertEqual(calls[1][0], 'Open file.')
        self.assertIn((1, 1), calls[1][1])

    def test_slot_failure_can_retry(self):
        def propose(text, banned):
            return Proposal(None, ((1, 3),), 'slot_fidelity') if not banned else Proposal(text)
        result = run_loop('Use `x`.', propose, lambda _: True)
        self.assertTrue(result['resolved'])
        self.assertEqual(result['passes'], 2)

    def test_last_valid_state_survives_cap(self):
        def propose(text, banned):
            return Proposal('Open.', ((1, len(banned) + 1),))
        result = run_loop('Open file.', propose, lambda _: False)
        self.assertEqual(result['text'], 'Open file.')
        self.assertEqual(result['reason'], 'pass_cap')
        self.assertEqual(result['passes'], 5)
        self.assertFalse(result['resolved'])

    def test_exhausted_and_stalled(self):
        result = run_loop('Open file.', lambda *_: Proposal(None, reason='unsupported'), lambda _: True)
        self.assertFalse(result['resolved'])
        self.assertEqual(result['reason'], 'exhausted')
        result = run_loop('Open file.', lambda text, _: Proposal(text), lambda _: False)
        self.assertEqual(result['reason'], 'stalled')

    def test_semantic_operators_and_content_cannot_disappear(self):
        for source, target in [('Do not delete the file.', 'Delete the file.'),
                               ('Do not open file.', 'Do open not file.'),
                               ('If the test fails, stop.', 'The test fails, stop.'),
                               ('Open the file.', 'Open.'),
                               ('Use the cache.', 'Give the cache.')]:
            self.assertIn('content_changed', validate(source, target))
        self.assertEqual(validate('Open file.', 'Open the file.'), [])

    def test_literal_detection_and_guards(self):
        literals = ['nfs://host/path', 'foo(x)', '``a ` b``', '```unclosed\nx=y', '"ready now"',
                    'C++', 'C#', '.NET', '<a href="x">', r'\(x=y\)', '**important**']
        for literal in literals:
            source = 'Use ' + literal + '.'
            self.assertTrue(any(t.protected and literal in t.text for t in model_tokens(source)), literal)
            self.assertTrue(validate(source, source.replace(literal, 'other')), literal)
        self.assertTrue(validate('Use `x`.', 'Use `x` `x`.'))
        self.assertFalse(any(t.protected and 'not' in t.text for t in model_tokens("Do not open it.")))

    def test_cap_validation(self):
        for cap in (0, 6):
            with self.assertRaises(ValueError):
                run_loop('x', None, None, cap)


if __name__ == '__main__':
    unittest.main()
