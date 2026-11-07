"""Checks for the v2 teacher context and meaning-class gates."""

import json
import unittest

from checker import Lexicon
from gates_v2 import verify_v2
from gen import proper_names
from ste_context import EXAMPLES, V1, build_system_prompt
from synthesize import mask_literals, restore
from synthesize_v2 import system_prefix


def rows(name):
    return [json.loads(line) for line in (V1 / name).read_text().splitlines()]


class GatesV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lex = Lexicon()

    def gate(self, source, target, mode="DESC"):
        issues = verify_v2(source, target, mode, self.lex, False, proper_names(source))
        return [i for i in issues if not i.startswith("checker_errors")]

    def test_may_satisfied_by_can_or_possible(self):
        self.assertEqual(self.gate("The server may close the connection.", "The server can close the connection."), [])
        self.assertEqual(self.gate("The server may close the connection.",
                                   "It is possible that the server closes the connection."), [])

    def test_should_as_advice_or_requirement(self):
        self.assertEqual(self.gate("You should monitor the archive.", "We recommend that you monitor the archive."), [])
        self.assertEqual(self.gate("The rest should be a filename.", "The remaining text must be a filename."), [])

    def test_requirement_not_weakened_to_advice(self):
        self.assertIn("modal_change:advice/obligation",
                      self.gate("You must monitor the archive.", "We recommend that you monitor the archive."))

    def test_may_wish_to_not_weakened_to_permission(self):
        issues = self.gate("You may wish to run the command.", "You can run the command.")
        self.assertIn("modal_change:advice/obligation", issues)

    def test_negation_classes(self):
        self.assertEqual(self.gate("It is not possible to switch the segments.", "You cannot switch the segments."), [])
        self.assertEqual(self.gate("The server sends an unsupported encoding.",
                                   "The server sends an encoding that curl does not support."), [])
        self.assertIn("negation_change", self.gate("Do not delete the log.", "Delete the log.", "PROC"))

    def test_whether_becomes_if(self):
        self.assertEqual(self.gate("The scan checks whether the row is visible.",
                                   "The scan checks if the row is visible."), [])

    def test_scope_word_kept(self):
        self.assertIn("condition_change:only", self.gate("This only limits the connection phase.",
                                                         "This option sets a limit for the connection phase."))

    def test_description_must_not_become_command(self):
        self.assertIn("description_became_instruction",
                      self.gate("The maximum is set with --parallel-max.", "Set the maximum with --parallel-max."))
        self.assertNotIn("description_became_instruction",
                         self.gate("Use --max-time to cap each attempt.", "Use --max-time to set a limit for each attempt.", "PROC"))


class ContextV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lex = Lexicon()
        cls.prompt = build_system_prompt()

    def test_every_worked_example_passes_v2_gates(self):
        sources = {r["id"]: r for r in rows("candidate_sources.jsonl")}
        for sid, target, _ in EXAMPLES:
            row = sources[sid]
            _, literals = mask_literals(row["text"])
            full = restore(target, literals)
            self.assertEqual(verify_v2(row["text"], full, row["mode"], self.lex, False, proper_names(row["text"])), [],
                             full)

    def test_examples_avoid_inputs_controls_and_holdout(self):
        used = {r["id"] for name in ("teacher_inputs.jsonl", "unchanged_controls.jsonl", "untouched_holdout.jsonl")
                for r in rows(name)}
        self.assertFalse({sid for sid, _, _ in EXAMPLES} & used)

    def test_prompt_contains_no_evaluated_passage(self):
        for name in ("teacher_inputs.jsonl", "unchanged_controls.jsonl", "untouched_holdout.jsonl"):
            for r in rows(name):
                self.assertNotIn(mask_literals(r["text"])[0], self.prompt)
                self.assertNotIn(r["text"], self.prompt)

    def test_prompt_has_every_reference_section(self):
        for heading in ("WORKED EXAMPLES", "STE RULES", "APPROVED WORDS.", "SOFTWARE WORDS.", "UNAPPROVED WORDS.", "REMINDER."):
            self.assertIn(heading, self.prompt)

    def test_system_prefix_cut(self):
        rendered = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"
        self.assertEqual(system_prefix(rendered), "<|im_start|>system\nS<|im_end|>\n")
        with self.assertRaises(ValueError):
            system_prefix("<|im_start|>user\nU<|im_end|>\n")


if __name__ == "__main__":
    unittest.main()
