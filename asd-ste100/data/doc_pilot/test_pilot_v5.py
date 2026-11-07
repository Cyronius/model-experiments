"""Checks for the v5 prompt variants, the template prefix cut and the parallel runner."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from checker import Lexicon
from ste_context_v4 import build_system_prompt_v4
from ste_context_v5 import VARIANTS, judge_prompt, rewrite_prompt
import synthesize_v5
from merge_v5 import literal_moved


class Prompts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lex = Lexicon()

    def test_full_is_the_v4_prompt(self):
        v4 = build_system_prompt_v4(self.lex)
        self.assertEqual(rewrite_prompt(self.lex, "full"), v4)
        self.assertEqual(judge_prompt(self.lex, "full"), v4)

    def test_short_variants_drop_the_lists_they_name(self):
        sizes = {v: len(rewrite_prompt(self.lex, v)) for v in VARIANTS}
        self.assertGreater(sizes["full"], sizes["core"])
        self.assertGreater(sizes["core"], sizes["lean"])
        self.assertGreater(sizes["lean"], sizes["tiny"])
        for variant in ("core", "lean", "tiny"):
            prompt = rewrite_prompt(self.lex, variant)
            self.assertNotIn("UNAPPROVED WORDS", prompt)
            self.assertNotIn("MEANING CHECK", prompt)
            self.assertIn("checker findings give the dictionary alternatives", prompt)
        self.assertIn("STE RULES", rewrite_prompt(self.lex, "core"))
        self.assertNotIn("STE RULES", rewrite_prompt(self.lex, "lean"))
        self.assertIn("APPROVED WORDS", rewrite_prompt(self.lex, "lean"))
        tiny = rewrite_prompt(self.lex, "tiny")
        self.assertNotIn("APPROVED WORDS", tiny)
        self.assertNotIn("SOFTWARE WORDS", tiny)

    def test_short_judge_prompt_has_the_check_and_its_examples(self):
        judge = judge_prompt(self.lex, "core")
        self.assertTrue(judge.startswith("The rewrites are in ASD-STE100"))
        self.assertIn("MEANING CHECK", judge)
        self.assertIn("Meaning check example 1.", judge)
        self.assertLess(len(judge), len(rewrite_prompt(self.lex, "tiny")))

    def test_traps_adds_the_meaning_traps_to_both_prompts(self):
        traps, tiny = rewrite_prompt(self.lex, "traps"), rewrite_prompt(self.lex, "tiny")
        self.assertIn("MEANING TRAPS", traps)
        self.assertNotIn("MEANING TRAPS", tiny)
        self.assertEqual(traps.replace(traps[traps.index("MEANING TRAPS"):traps.index("WORKED EXAMPLES")], ""), tiny)
        self.assertIn("easy to miss", judge_prompt(self.lex, "traps"))
        self.assertNotIn("easy to miss", judge_prompt(self.lex, "tiny"))

    def test_unknown_variant(self):
        with self.assertRaises(ValueError):
            rewrite_prompt(self.lex, "huge")


class Prefix(unittest.TestCase):
    def test_chatml(self):
        a = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nfirst probe<|im_end|>\n<|im_start|>assistant\n"
        self.assertEqual(synthesize_v5.system_prefix(a, a.replace("first", "second")), "<|im_start|>system\nS<|im_end|>\n")

    def test_harmony(self):
        a = "<|start|>system<|message|>x<|end|><|start|>developer<|message|>S<|end|><|start|>user<|message|>first probe<|end|>"
        self.assertEqual(synthesize_v5.system_prefix(a, a.replace("first", "second")),
                         "<|start|>system<|message|>x<|end|><|start|>developer<|message|>S<|end|>")

    def test_layout_without_a_user_turn(self):
        with self.assertRaises(ValueError):
            synthesize_v5.system_prefix("plain first probe", "plain second probe")


class Runner(unittest.TestCase):
    """One passage through the loop with a fake server: openai backend, no meaning check."""

    def test_openai_backend_records_a_passage(self):
        row = json.loads((synthesize_v5.V4 / "teacher_inputs.jsonl").read_text().splitlines()[2])
        replies = iter([{"choices": [{"message": {"content": json.dumps({"rewrite": "In [[L0]] [[L1]], a new procedure "
                         "to configure logging is available. It uses dictionaries to hold configuration information.",
                                                                         "uncertain": False})}}],
                         "usage": {"prompt_tokens": 400, "completion_tokens": 40,
                                   "prompt_tokens_details": {"cached_tokens": 300}}}])
        calls = []

        def fake_post(url, path, body, timeout):
            calls.append(path)
            return next(replies)

        args = mock.Mock(url="http://x", backend="openai", served_model="x", profile="qwen36", model_id="m",
                         variant="core", parallel=1, no_check=True, timeout=5, warm_timeout=5)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(synthesize_v5, "post", fake_post):
            teacher = synthesize_v5.Teacher(args, Lexicon())
            record = teacher.passage(row, Path(tmp))
            self.assertEqual(calls, ["/v1/chat/completions"])   # no slot restore, no prefix warm-up
            self.assertEqual(record["issues"], [])
            self.assertEqual(record["gates_only"]["attempt"], 1)
            self.assertEqual(teacher.stats["prompt_tokens"], 100)
            self.assertEqual(teacher.stats["cached_tokens"], 300)
            self.assertTrue((Path(tmp) / "accepted.jsonl").exists())


class MergeTest(unittest.TestCase):
    def test_literal_moved_into_another_word(self):
        source = "Codecs: utf-16-be, utf-32-be, for code points (`U+D800` - `U+DFFF`)."
        swapped = "Codecs: utf-16-be, utf-`U+D800`-be, for code points (32 - `U+DFFF`)."
        self.assertTrue(literal_moved(source, swapped))
        self.assertFalse(literal_moved(source, "The code points (`U+D800` - `U+DFFF`) use utf-16-be and utf-32-be."))


if __name__ == "__main__":
    unittest.main()
