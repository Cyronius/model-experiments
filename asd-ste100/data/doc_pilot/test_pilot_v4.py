"""Checks for the v4 selection, gates, prompt, feedback and synthesis loop."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from checker import Lexicon
from gates_v4 import dropped_units, marker_problems, restore_any_order, verify_v4
from gen import proper_names
from ste_context import EXAMPLES, V1
from ste_context_v4 import (EXAMPLES_V4, JUDGE_EXAMPLES_V4, build_system_prompt_v4, feedback_v4, judge_examples_v4,
                            judge_message_v4, marker_feedback_v4, source_hints_v4)
from synthesize import mask_literals, restore
from synthesize_v2 import first_rows
import synthesize_v4

V4_INPUTS = synthesize_v4.OUT / "teacher_inputs.jsonl"


def rows(name, folder=V1):
    return [json.loads(line) for line in (folder / name).read_text().splitlines()]


class GatesV4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lex = Lexicon()

    def gate(self, source, target, mode="DESC"):
        issues = verify_v4(source, target, mode, self.lex, False, proper_names(source))
        return [i for i in issues if not i.startswith("checker_errors")]

    def test_advisable_is_advice(self):
        self.assertEqual(self.gate("It is advisable to test the library first.",
                                   "We recommend that you test the library first."), [])

    def test_allowed_to_is_can(self):
        self.assertEqual(self.gate("A transfer that started is allowed to run to the end.",
                                   "A transfer that started can run to the end."), [])
        self.assertIn("modal_change:possibility", self.gate("A transfer that started is allowed to run to the end.",
                                                            "A transfer that started runs to the end."))

    def test_require_is_obligation(self):
        self.assertEqual(self.gate("A custom exporter is required for the queue.",
                                   "A custom exporter is necessary for the queue."), [])

    def test_divided_sentence_can_repeat_a_modal_once(self):
        source = "With timelines, you can recover to any prior state, including states in branches that you left."
        self.assertEqual(self.gate(source, "With timelines, you can restore a prior state. You can also restore a "
                                           "state in a branch that you left."), [])
        self.assertIn("modal_change:possibility", self.gate(
            source, "With timelines, you can restore a prior state. You can restore a state in a branch. You can "
                    "leave a branch."))

    def test_hedges_stay(self):
        source = "The configuration presumably adds some handlers."
        self.assertEqual(self.gate(source, "The configuration probably adds some handlers."), [])
        self.assertEqual(self.gate(source, "The configuration possibly adds some handlers."), [])
        self.assertIn("certainty_change:hedge", self.gate(source, "The configuration adds some handlers."))
        self.assertIn("certainty_change:hedge", self.gate("The configuration adds some handlers.",
                                                          "The configuration probably adds some handlers."))

    def test_may_can_become_possibly(self):
        self.assertEqual(self.gate("The value may be out of date.", "The value is possibly out of date."), [])

    def test_certainty_and_frequency_stay(self):
        source = "Aggressive scans are guaranteed to advance the limit and usually finish quickly."
        self.assertEqual(self.gate(source, "Aggressive scans always move the limit forward and usually finish quickly."), [])
        issues = self.gate(source, "Aggressive scans can move the limit forward and finish quickly.")
        self.assertIn("certainty_change:certainty", issues)
        self.assertIn("certainty_change:frequency", issues)

    def test_units_stay(self):
        self.assertEqual(dropped_units("Set the limit (in bytes per second).", "Set the limit."), ["byte", "second"])
        self.assertEqual(dropped_units("Wait 5 seconds.", "Wait for 5 seconds."), [])
        self.assertIn("unit_dropped:byte", self.gate("The limit is in bytes.", "The limit is a value."))

    def test_markers_can_move_but_not_drop(self):
        literals = ["22", "400"]
        moved = "For codes at [[L1]] or more, curl fails with error code [[L0]]."
        self.assertFalse(marker_problems(moved, 2))
        self.assertEqual(restore_any_order(moved, literals), "For codes at 400 or more, curl fails with error code 22.")
        self.assertTrue(marker_problems("curl fails with error code [[L0]].", 2))
        self.assertTrue(marker_problems("[[L0]] and [[L0]] and [[L1]].", 2))
        self.assertTrue(marker_problems("[[L0]] [[L1]] [[L2]].", 2))

    def test_every_worked_example_passes_v4_gates(self):
        sources = {r["id"]: r for r in rows("candidate_sources.jsonl")}
        for sid, target, _ in EXAMPLES:
            row = sources[sid]
            full = restore(target, mask_literals(row["text"])[1])
            self.assertEqual(verify_v4(row["text"], full, row["mode"], self.lex, False, proper_names(row["text"])), [], full)
        for mode, passage, target, _ in EXAMPLES_V4:
            self.assertEqual(verify_v4(passage, target, mode, self.lex, False, proper_names(passage)), [], target)

    def test_invented_judge_rewrites_pass_the_gates(self):
        for mode, text, rewrite, _ in JUDGE_EXAMPLES_V4:
            self.assertEqual(verify_v4(text, rewrite, mode, self.lex, False, proper_names(text)), [], rewrite)


@unittest.skipUnless(V4_INPUTS.exists(), "run select_sources_v4.py first")
class SelectionV4(unittest.TestCase):
    def test_new_inputs_are_unused_and_the_v1_inputs_follow(self):
        inputs = rows(V4_INPUTS.name, V4_INPUTS.parent)
        self.assertEqual(len({r["id"] for r in inputs}), 132)
        new, old = inputs[:96], inputs[96:]
        self.assertEqual([r["id"] for r in old], [r["id"] for r in first_rows(36)])
        used = {r["id"] for name in ("teacher_inputs.jsonl", "unchanged_controls.jsonl", "untouched_holdout.jsonl")
                for r in rows(name)} | {e[0] for e in EXAMPLES}
        self.assertFalse({r["id"] for r in new} & used)
        self.assertTrue(all(r["pilot_set"] == "v4_new" and r["checker_error_rules"] for r in new))
        self.assertNotIn("rust_book", {r["family"] for r in inputs})


class ContextV4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lex = Lexicon()
        cls.prompt = build_system_prompt_v4(cls.lex)

    def test_prompt_contains_no_evaluated_passage(self):
        evaluated = [r for name in ("teacher_inputs.jsonl", "unchanged_controls.jsonl", "untouched_holdout.jsonl")
                     for r in rows(name)]
        if V4_INPUTS.exists():
            evaluated += rows(V4_INPUTS.name, V4_INPUTS.parent)
        for r in evaluated:
            self.assertNotIn(mask_literals(r["text"])[0], self.prompt)
            self.assertNotIn(r["text"], self.prompt)

    def test_invented_examples_are_in_no_source_passage(self):
        texts = {r["text"] for r in rows("candidate_sources.jsonl")}
        if V4_INPUTS.exists():
            texts |= {r["text"] for r in rows(V4_INPUTS.name, V4_INPUTS.parent)}
        for passage in [e[1] for e in EXAMPLES_V4] + [e[1] for e in JUDGE_EXAMPLES_V4]:
            self.assertNotIn(passage, texts)

    def test_prompt_sections(self):
        for heading in ("KEEP ALL THE INFORMATION.", "Example 12. Mode: DESC", "MEANING CHECK.",
                        "Meaning check example 10.", "STE RULES", "APPROVED WORDS.", "SOFTWARE WORDS.",
                        "UNAPPROVED WORDS.", "REMINDER."):
            self.assertIn(heading, self.prompt)
        self.assertNotIn("in the same order", self.prompt)
        self.assertEqual(len(judge_examples_v4()), 10)

    def test_judge_message_wording(self):
        source = "Task groups combine a task creation API with a convenient way to wait for all tasks."
        message = judge_message_v4(source, "Task groups put together a task creation API and a procedure to wait for "
                                           "all tasks.", "DESC", self.lex, proper_names(source))
        self.assertIn("had to replace: combine, way\n", message)
        self.assertTrue(message.endswith("not in the rewrite: convenient"))

    def test_adverb_hints(self):
        hints = source_hints_v4("A transfer that has already started can run even if it takes a long time.", "DESC",
                                self.lex)
        self.assertIn("'has already started' -> 'is in progress'", hints)
        even = next(line for line in hints.splitlines() if line.startswith("- 'even'"))
        self.assertLess(even.index("'even if X' -> a second sentence"), even.index("If nothing fits"))

    def test_certainty_and_unit_feedback(self):
        source = "The rate is presumably in bytes per second."
        note = feedback_v4(["certainty_change:hedge", "unit_dropped:byte"], source, "The rate is in seconds.", "DESC",
                           self.lex)
        self.assertIn("The hedge words changed. The input has hedge: 'presumably'. Your rewrite has none.", note)
        self.assertIn("drops the unit 'byte'", note)

    def test_marker_feedback_quotes_the_input_words(self):
        marked = "Alternately, the filename [[L0]] (a single period) may be specified instead of [[L1]] to use stdin."
        note = marker_feedback_v4("You can set [[L0]] to a single period.", marked, 2)
        self.assertIn('It drops [[L1]]. In the input, [[L1]] is in "period) may be specified instead of [[L1]] to '
                      'use stdin.". Put [[L1]] back with those words.', note)
        self.assertNotIn("order", note)


class SynthesisLoopV4(unittest.TestCase):
    """Drive synthesize_v4.main against a fake server."""

    def run_fake(self, source, replies, verdicts):
        replies, verdicts, bodies = iter(replies), iter(verdicts), []

        def fake_post(url, path, body, timeout):
            if path == "/apply-template":
                return {"prompt": "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n" + body["messages"][1]["content"]}
            if path == "/completion":
                return {"tokens_evaluated": 100, "timings": {}}
            if path.startswith("/slots/0"):
                return {"n_restored": 100}
            bodies.append(body)
            user = body["messages"][-1]["content"]
            content = (json.dumps({"differences": next(verdicts)}) if user.startswith("MEANING CHECK")
                       else json.dumps(next(replies)))
            return {"choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 120, "prompt_tokens_details": {"cached_tokens": 100}}, "timings": {}}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "synthesis-plan.json").write_text(json.dumps({"splits": {"untouched_holdout": ["rust_book"]},
                                                                 "source_manifest_sha256": "x"}))
            (out / "teacher_inputs.jsonl").write_text(json.dumps(dict(source, pilot_set="v4_new")) + "\n")
            with mock.patch.object(synthesize_v4, "OUT", out), mock.patch.object(synthesize_v4, "post", fake_post), \
                    mock.patch.object(sys, "argv", ["synthesize_v4.py", "--limit", "1"]):
                synthesize_v4.main()
            read = lambda name: ([json.loads(l) for l in (out / name).read_text().splitlines()]
                                 if (out / name).exists() else [])
            return (read("attempts.jsonl"), read("accepted.jsonl"), read("rejected.jsonl"),
                    json.loads((out / "synthesis-summary.json").read_text()), bodies)

    def test_moved_markers_pass_and_gates_only_is_recorded(self):
        source = next(r for r in rows("teacher_inputs.jsonl") if r["id"].startswith("41f812d0ef1f"))
        moved = {"rewrite": "For HTTP transfers that return HTTP response codes at [[L1]] or greater, curl fails with "
                            "error code [[L0]] and does not output a response body.", "uncertain": False}
        traces, accepted, rejected, summary, bodies = self.run_fake(
            source, [moved, moved], [["'at all' -> (dropped): no output at all."], []])
        self.assertEqual(traces[0]["issues"], ["meaning_check:1"])
        self.assertIn("400 or greater", traces[0]["target"])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["gates_only"], {"attempt": 1, "ste": traces[0]["target"]})
        self.assertEqual(summary["gates_only_by_set"], {"v4_new": 1})
        self.assertEqual([b["temperature"] for b in bodies], [0.0, 0.0, 0.7, 0.0])

    def test_dropped_marker_feedback_quotes_the_input(self):
        source = next(r for r in rows("teacher_inputs.jsonl") if r["id"].startswith("0cc668ae8984"))
        dropped = {"rewrite": "As an alternative, you can set [[L0]] to a single period.", "uncertain": False}
        traces, accepted, rejected, _, _ = self.run_fake(source, [dropped] * 3, [])
        self.assertEqual([t["issues"] for t in traces], [["marker_change"]] * 3)
        self.assertIn("In the input, [[L1]] is in", traces[0]["feedback"])
        self.assertEqual(rejected[0]["gates_only"], None)


if __name__ == "__main__":
    unittest.main()
