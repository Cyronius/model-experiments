"""Checks for the v3 gates, hints, feedback, meaning check and synthesis loop."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from checker import Lexicon
from gates_v3 import bad_verb_forms, dropped_words, verify_v3
from gen import proper_names
from ste_context import EXAMPLES, V1
from ste_context_v3 import (JUDGE_EXAMPLES, build_system_prompt_v3, feedback_v3, judge_message, marker_feedback,
                            parse_judge, source_hints_v3)
from synthesize import mask_literals, restore
import synthesize_v3


def rows(name):
    return [json.loads(line) for line in (V1 / name).read_text().splitlines()]


class GatesV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lex = Lexicon()

    def gate(self, source, target, mode="DESC"):
        issues = verify_v3(source, target, mode, self.lex, False, proper_names(source))
        return [i for i in issues if not i.startswith("checker_errors")]

    def test_nothing_else_is_only(self):
        self.assertEqual(self.gate("The name comes from the URL, nothing else.", "The name comes from the URL only."), [])

    def test_lost_can_become_one_negation(self):
        source = "If you define the same value, the predefined name is lost."
        self.assertEqual(self.gate(source, "If you define the same value, you cannot find the predefined name."), [])
        self.assertIn("negation_change", self.gate(source, "If you do not define the same value, you cannot find the name."))

    def test_pasted_alternative_after_be(self):
        source = "If you define the same value, the predefined name is lost."
        self.assertEqual(bad_verb_forms(source, "The predefined name is not find.", self.lex), ["is not find"])
        self.assertEqual(bad_verb_forms(source, "The connection phase is complete.", self.lex), [])

    def test_once_becomes_when(self):
        self.assertEqual(self.gate("Once the archive is full, the server stops.", "When the archive is full, the server stops."), [])

    def test_v2_gates_still_hold(self):
        self.assertIn("negation_change", self.gate("Do not delete the log.", "Delete the log.", "PROC"))
        self.assertIn("modal_change:advice/obligation",
                      self.gate("You must monitor the archive.", "We recommend that you monitor the archive."))
        self.assertIn("condition_change:contrast",
                      self.gate("Code can make formatters, although you could subclass one.",
                                "Code can make formatters. You can subclass one."))

    def test_dropped_words_flag_unlisted_qualifiers(self):
        source = "Task groups combine a task creation API with a convenient and reliable way to wait for all tasks."
        target = "Task groups put together a task creation API and a procedure to wait for all tasks."
        self.assertEqual(dropped_words(source, target, self.lex, proper_names(source)), ["convenient", "reliable"])

    def test_every_worked_example_passes_v3_gates(self):
        sources = {r["id"]: r for r in rows("candidate_sources.jsonl")}
        for sid, target, _ in EXAMPLES:
            row = sources[sid]
            full = restore(target, mask_literals(row["text"])[1])
            self.assertEqual(verify_v3(row["text"], full, row["mode"], self.lex, False, proper_names(row["text"])), [], full)


class ContextV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lex = Lexicon()
        cls.prompt = build_system_prompt_v3(cls.lex)

    def test_sibling_forms_come_before_sense_blind_alternatives(self):
        hints = source_hints_v3("A scan fetches each matching index entry.", "DESC", self.lex)
        self.assertIn("The software verb 'match' is approved in these forms: match, matches, matched", hints)
        self.assertLess(hints.index("match, matches"), hints.index("align"))

    def test_negation_feedback_names_the_added_sentence(self):
        source = "Use stdin in non-blocking mode."
        target = "You use stdin in non-blocking mode. The option is not used."
        note = feedback_v3(["negation_change"], source, target, "DESC", self.lex)
        self.assertIn('"The option is not used."', note)
        self.assertNotIn('"You use stdin', note)

    def test_feedback_reports_repeats_and_differences(self):
        note = feedback_v3(["meaning_check:1"], "a", "b", "DESC", self.lex, differences=["The rewrite drops 'requested'."],
                           repeated=True)
        self.assertIn("(1) The rewrite drops 'requested'.", note)
        self.assertIn("same as an earlier attempt", note)

    def test_marker_feedback_names_the_dropped_marker(self):
        note = marker_feedback("Set [[L0]] to a period.", 2)
        self.assertIn("the input has [[L0]] [[L1]]", note)
        self.assertIn("It drops [[L1]]", note)
        self.assertIn("order changed", marker_feedback("[[L1]] then [[L0]].", 2))

    def test_parse_judge(self):
        self.assertEqual(parse_judge('{"differences": []}'), [])
        self.assertEqual(parse_judge('```json\n{"differences": ["x drops y"]}\n```'), ["x drops y"])
        with self.assertRaises(ValueError):
            parse_judge('{"rewrite": "x", "uncertain": false}')

    def test_judge_examples_avoid_inputs_controls_and_holdout(self):
        used = {r["id"] for name in ("teacher_inputs.jsonl", "unchanged_controls.jsonl", "untouched_holdout.jsonl")
                for r in rows(name)}
        self.assertFalse({sid for sid, _, _ in JUDGE_EXAMPLES} & used)

    def test_prompt_contains_no_evaluated_passage(self):
        for name in ("teacher_inputs.jsonl", "unchanged_controls.jsonl", "untouched_holdout.jsonl"):
            for r in rows(name):
                self.assertNotIn(mask_literals(r["text"])[0], self.prompt)
                self.assertNotIn(r["text"], self.prompt)

    def test_prompt_has_both_tasks(self):
        for heading in ("WORKED EXAMPLES", "MEANING CHECK.", "Meaning check example 5.", "STE RULES", "APPROVED WORDS.",
                        "SOFTWARE WORDS.", "UNAPPROVED WORDS.", "REMINDER."):
            self.assertIn(heading, self.prompt)

    def test_judge_message_lists_replaced_and_missing_words(self):
        source = "Task groups combine a task creation API with a convenient way to wait for all tasks."
        message = judge_message(source, "Task groups put together a task creation API and a procedure to wait for all tasks.",
                                "DESC", self.lex, proper_names(source))
        self.assertTrue(message.startswith("MEANING CHECK\nMode: DESC\n"))
        self.assertIn("replacement): combine, way", message)
        self.assertIn("stays): convenient", message)


class SynthesisLoop(unittest.TestCase):
    """Drive synthesize_v3.main against a fake server: retry, repeat, meaning check, accept."""

    def run_fake(self, source, replies, judge=lambda message: []):
        replies, calls = iter(replies), []

        def fake_post(url, path, body, timeout):
            calls.append(path)
            if path == "/apply-template":
                return {"prompt": "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n" + body["messages"][1]["content"]}
            if path == "/completion":
                return {"tokens_evaluated": 100, "timings": {}}
            if path.startswith("/slots/0"):
                return {"n_restored": 100}
            user = body["messages"][-1]["content"]
            content = (json.dumps({"differences": judge(user)}) if user.startswith("MEANING CHECK")
                       else json.dumps(next(replies)))
            return {"choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 120, "prompt_tokens_details": {"cached_tokens": 100}}, "timings": {}}

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(synthesize_v3, "OUT", Path(tmp)), \
                mock.patch.object(synthesize_v3, "post", fake_post), \
                mock.patch.object(synthesize_v3, "first_rows", lambda limit: [source]), \
                mock.patch.object(sys, "argv", ["synthesize_v3.py", "--limit", "1"]):
            synthesize_v3.main()
            read = lambda name: ([json.loads(l) for l in (Path(tmp) / name).read_text().splitlines()]
                                 if (Path(tmp) / name).exists() else [])
            return read("attempts.jsonl"), read("accepted.jsonl"), read("rejected.jsonl"), calls

    def test_marker_error_gets_a_retry(self):
        source = next(r for r in rows("teacher_inputs.jsonl") if r["id"].startswith("0cc668ae8984"))
        dropped = {"rewrite": "As an alternative, you can set [[L0]] to a single period.", "uncertain": False}
        traces, accepted, rejected, _ = self.run_fake(source, [dropped, dropped, dropped])
        self.assertEqual([t["attempt"] for t in traces], [1, 2, 3])
        self.assertEqual(traces[0]["issues"], ["marker_change"])
        self.assertIn("It drops [[L1]]", traces[0]["feedback"])
        self.assertIn("same as an earlier attempt", traces[1]["feedback"])
        self.assertEqual((len(accepted), len(rejected)), (0, 1))

    def test_meaning_check_difference_goes_back_as_feedback(self):
        source = next(r for r in rows("teacher_inputs.jsonl") if r["id"].startswith("2ef7c21814b7"))
        good = {"rewrite": "The remote filename for saving comes from the given URL only. If the file exists, curl overwrites it.",
                "uncertain": False}
        verdicts = iter([["The rewrite drops 'remote'."], []])
        traces, accepted, _, _ = self.run_fake(source, [good, good], judge=lambda message: next(verdicts))
        self.assertEqual(traces[0]["issues"], ["meaning_check:1"])
        self.assertIn("(1) The rewrite drops 'remote'.", traces[0]["feedback"])
        self.assertEqual(traces[1]["issues"], [])
        self.assertEqual(len(accepted), 1)

    def test_loop(self):
        source = next(r for r in rows("teacher_inputs.jsonl") if r["id"].startswith("2ef7c21814b7"))
        replies = iter([
            # attempt 1 fails the checker (extracted), attempt 2 repeats it, attempt 3 passes and the check is clean
            {"rewrite": "The remote filename for saving is extracted from the given URL only. If the file exists, curl overwrites it.", "uncertain": False},
            {"rewrite": "The remote filename for saving is extracted from the given URL only. If the file exists, curl overwrites it.", "uncertain": False},
            {"rewrite": "The remote filename for saving comes from the given URL only. If the file exists, curl overwrites it.", "uncertain": False},
        ])
        calls = []

        bodies = []

        def fake_post(url, path, body, timeout):
            calls.append(path)
            if path == "/apply-template":
                return {"prompt": "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n" + body["messages"][1]["content"]}
            if path == "/completion":
                return {"tokens_evaluated": 100, "timings": {}}
            if path.startswith("/slots/0"):
                return {"n_restored": 100}
            bodies.append(body)
            user = body["messages"][-1]["content"]
            content = (json.dumps({"differences": []}) if user.startswith("MEANING CHECK")
                       else json.dumps(next(replies)))
            return {"choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 120, "prompt_tokens_details": {"cached_tokens": 100}}, "timings": {}}

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(synthesize_v3, "OUT", Path(tmp)), \
                mock.patch.object(synthesize_v3, "post", fake_post), \
                mock.patch.object(synthesize_v3, "first_rows", lambda limit: [source]), \
                mock.patch.object(sys, "argv", ["synthesize_v3.py", "--limit", "1"]):
            synthesize_v3.main()
            traces = [json.loads(l) for l in (Path(tmp) / "attempts.jsonl").read_text().splitlines()]
            accepted = [json.loads(l) for l in (Path(tmp) / "accepted.jsonl").read_text().splitlines()]
            summary = json.loads((Path(tmp) / "synthesis-summary.json").read_text())
        self.assertEqual([t["attempt"] for t in traces], [1, 2, 3])
        self.assertTrue(traces[1]["repeated"])
        self.assertIn("same as an earlier attempt", traces[1]["feedback"])
        self.assertIn("'extracted' is not an approved word", traces[0]["feedback"])
        self.assertEqual(traces[2]["issues"], [])
        self.assertEqual(traces[2]["meaning_check"]["differences"], [])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["meaning_check"], [])
        self.assertEqual(summary["meaning_checks"], {"requests": 1, "found_differences": 0, "unparsed": 0})
        self.assertEqual(calls.count("/slots/0?action=restore"), 4)
        self.assertEqual([b["temperature"] for b in bodies], [0.0, 0.7, 0.7, 0.0])
        self.assertEqual([b["seed"] for b in bodies], [42, 43, 44, 42])


if __name__ == "__main__":
    unittest.main()
