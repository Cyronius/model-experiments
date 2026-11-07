"""Failure-focused checks for the bounded teacher pilot's deterministic gates."""

import json
import unittest
from pathlib import Path

from synthesize import OUT, mask_literals, restore, verify
from select_sources import self_contained
from checker import Lexicon


class PilotGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lex = Lexicon()

    def test_literals_round_trip_on_every_frozen_teacher_source(self):
        rows = [json.loads(line) for line in (OUT / "teacher_inputs.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 36)
        for row in rows:
            self.assertTrue(self_contained(row), row["text"])
            marked, values = mask_literals(row["text"])
            self.assertEqual(restore(marked, values), row["text"])

    def test_missing_literal_rejected(self):
        marked, values = mask_literals("Run --retry for 50 seconds.")
        self.assertTrue(values)
        with self.assertRaises(ValueError):
            restore(marked.replace("[[L0]]", ""), values)

    def test_negation_change_rejected(self):
        issues = verify("Do not delete `log`.", "Delete `log`.", "PROC", self.lex, False)
        self.assertIn("negation_change", issues)

    def test_modal_change_rejected(self):
        issues = verify("You must keep `log`.", "You can keep `log`.", "DESC", self.lex, False)
        self.assertIn("modal_or_condition_change", issues)

    def test_entire_rust_family_untouched(self):
        plan = json.loads((OUT / "synthesis-plan.json").read_text())
        self.assertEqual(plan["splits"]["untouched_holdout"], ["rust_book"])
        inputs = [json.loads(line) for line in (OUT / "teacher_inputs.jsonl").read_text().splitlines()]
        holdout = [json.loads(line) for line in (OUT / "untouched_holdout.jsonl").read_text().splitlines()]
        self.assertFalse({r["id"] for r in inputs} & {r["id"] for r in holdout})
        self.assertTrue(all(r["family"] == "rust_book" for r in holdout))


if __name__ == "__main__":
    unittest.main()
