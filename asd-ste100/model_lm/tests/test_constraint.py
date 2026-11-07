import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ste_constraint import Allowed, ends_phrase, extends_ok, tail_word  # noqa: E402

STE = {'the', 'must', 'send', 'data', 'if', 'it', 'is', 'not', 'a', 'message', 'change'}
SRC = 'The frontend must now send copy-in data (if not prepared, send a CopyFail message to modify it).'


def allowed(copy_ok=None):
    return Allowed(sorted(STE), STE, SRC, copy_ok)


class ConstraintTests(unittest.TestCase):
    def test_words_split_across_tokens(self):
        a = allowed()
        self.assertTrue(extends_ok(a, '', ' front'))
        self.assertTrue(extends_ok(a, 'front', 'end.'))
        self.assertTrue(extends_ok(a, 'Copy', 'Fail'))
        self.assertFalse(extends_ok(a, 'Copy', 'fail'))
        self.assertTrue(extends_ok(a, '', ' The'))
        self.assertTrue(extends_ok(a, '', ' THE'))  # STE words are case-free

    def test_unknown_words_and_unfinished_words(self):
        a = allowed()
        self.assertFalse(extends_ok(a, '', ' utilize'))
        self.assertFalse(extends_ok(a, 'chan', ' it'))  # 'chan' is not a finished word
        self.assertTrue(extends_ok(a, 'chan', 'ge'))
        self.assertTrue(extends_ok(a, '', ' copy-in'))

    def test_copy_filter(self):
        a = allowed(copy_ok=lambda w: w not in {'modify', 'now'})
        self.assertFalse(extends_ok(a, '', ' modify'))
        self.assertFalse(extends_ok(a, '', ' now'))
        self.assertTrue(extends_ok(a, '', ' CopyFail'))

    def test_protected_span_words_are_copyable(self):
        a = Allowed(sorted(STE), STE, 'Run `BEGIN` first.', copy_ok=lambda w: w != 'BEGIN',
                    protected=['`BEGIN`'])
        self.assertTrue(extends_ok(a, '', ' `BEGIN`'))
        b = Allowed(sorted(STE), STE, 'Run `BEGIN` first.', copy_ok=lambda w: w != 'BEGIN')
        self.assertFalse(extends_ok(b, '', ' `BEGIN`'))

    def test_rejected_phrases(self):
        phrases = {('such', 'as'), ('according', 'to')}
        self.assertTrue(ends_phrase(['use', 'such'], ['as'], phrases, 2))
        self.assertTrue(ends_phrase(['use'], ['such', 'as'], phrases, 2))
        self.assertFalse(ends_phrase(['such', 'a'], ['file'], phrases, 2))
        self.assertFalse(ends_phrase([], ['as'], phrases, 2))

    def test_tail_word(self):
        self.assertEqual(tail_word('send copy'), 'copy')
        self.assertEqual(tail_word('send copy.'), '')


if __name__ == '__main__':
    unittest.main()
