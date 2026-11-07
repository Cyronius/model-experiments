"""System prompts for the v5 speed runs: the v4 prompt, or shorter ones.

The v4 prompt (about 30k tokens) carries everything: the task, the worked
examples, the meaning check, the rule summaries, the approved, software and
unapproved word lists. Half of it is the unapproved list, which the checker
findings already cover for the words of each passage: every unapproved word
of a passage comes with its dictionary alternatives in the user message.

- full: the v4 prompt, one prefix for the rewrite and the meaning check.
- core: the task, the examples, the rule summaries and the approved and
  software lists (about 12k tokens). The meaning check gets its own prefix.
- lean: core without the rule summaries (about 8k tokens).
- tiny: the task and the examples only (about 3k tokens).
- traps: tiny plus MEANING TRAPS, the meaning changes that passed the checker and
  the meaning check in the v4 and v5 reviews (changed agents, which-clauses made
  into rules, comparisons turned around, one-of lists made into all-of, lost
  version conditions, changed objects, attempts made into facts), with invented
  examples. Its meaning check lists the same changes.
The meaning check prefix of the shorter variants is the short STE note that
the NPU judges used, the meaning check text and its examples.
The prompts are AGPL-derived, so they are written only under ignored data/out.
"""

from __future__ import annotations

import json

from ste_context import LEX, _approved_section, _rules_section, _software_section
from ste_context_v3 import _edit
from ste_context_v4 import TASK_V4, _example_section_v4, _judge_section_v4, build_system_prompt_v4

VARIANTS = ("full", "core", "lean", "tiny", "traps")

# Without the unapproved list, the alternatives come from the checker findings in the user message.
TASK_SHORT = _edit(TASK_V4, "Do not use a word from UNAPPROVED WORDS. Each one has dictionary alternatives, but",
                   "Do not use an unapproved word. The checker findings give the dictionary alternatives of each "
                   "unapproved word in the passage, but")
TASK_SHORT = _edit(TASK_SHORT, "\n\nA user message that starts with \"MEANING CHECK\" asks for a different task. See MEANING CHECK.",
                   "")
# Without the word lists, the model uses the STE words that it knows.
TASK_TINY = _edit(TASK_SHORT, "- words from APPROVED WORDS, in the part of speech and the forms that the list gives,\n"
                              "- words from SOFTWARE WORDS,\n",
                  "- approved STE words, in their approved part of speech and forms (the checker findings name each "
                  "word that is not approved),\n- common computer words (file, run, log, test),\n")

# The v4/v5 reviews found these changes again and again in rewrites that passed every gate. The
# examples are invented, so no passage of a source file is one of them.
MEANING_TRAPS = """MEANING TRAPS. Earlier rewrites passed the checker but changed the meaning in these ways. Check your rewrite for each one:
- The agent. Keep who or what does each action. Before you put a noun for "it" or for the agent of a passive verb, find what the input means: "If the proxy sends the job to a second queue, it does not get the token" means that the second queue does not get the token. It does not mean that the proxy does not get it.
- A clause that tells which items. "Jobs which need more memory" tells which jobs. Do not change it into a rule ("These jobs must have more memory") or into a separate statement.
- A comparison or an order. "The file that changed least recently" is the file that changed the longest time ago, not the file that changed last. "More than the minimum work" is not necessary work.
- A choice. "Either A, B or C" and "one of" mean one item of the list: write "one of these: A, B or C". Do not write "It is A. It is also B."
- A version or a time condition. "Since version 4.2, you can ..." stays "Since version 4.2" or "From version 4.2".
- The exact thing. "The requested page" is the page, not the request. "Very old rows" are not "expired rows" or "used rows".
- An attempt or a hedge. "Tries to keep the cache small" is not "keeps the cache small"."""
JUDGE_TRAPS = """In particular, report these changes, which are easy to miss: a different agent (who does an action, or what "it" is); a clause that tells which items changed into a rule or a separate statement; a comparison or an order turned around ("least recently" -> "last"); one item of a list changed into all of them; a lost version or time condition; a different object ("the requested page" -> "the request"); an attempt or a hedge changed into a fact."""

STE_NOTE = """The rewrites are in ASD-STE100 Simplified Technical English (STE). STE uses only approved words, short sentences, the active voice and simple tenses. It writes modal words by sense ("MODAL WORDS"): may or might -> can or possibly; should (advice) -> "we recommend that you"; should (a requirement) -> must; whether -> if; once -> when; never -> do not; except -> but."""

REWRITE_REMINDER = ("REMINDER. Keep the meaning exactly, then use STE. Keep every literal marker exactly once. Keep every "
                    "word that the checker findings do not name, if it carries meaning. Reply with one JSON object only: "
                    "{\"rewrite\": \"...\", \"uncertain\": false}.")
JUDGE_REMINDER = "REMINDER. For a MEANING CHECK, reply with one JSON object only: {\"differences\": [...]}."


def rewrite_prompt(lex, variant: str) -> str:
    if variant == "full":
        return build_system_prompt_v4(lex)
    rows = [json.loads(l) for l in (LEX / "lexicon.jsonl").open(encoding="utf-8")]
    parts = [TASK_TINY if variant in ("tiny", "traps") else TASK_SHORT]
    if variant == "traps":
        parts.append(MEANING_TRAPS)
    parts.append("WORKED EXAMPLES\n\n" + _example_section_v4())
    if variant == "core":
        parts.append("STE RULES (summaries)\n\n" + _rules_section())
    if variant in ("core", "lean"):
        parts += ["APPROVED WORDS. Each word is approved only as the part of speech of its section, with the forms in "
                  "parentheses.\n\n" + _approved_section(rows),
                  "SOFTWARE WORDS. A project list of computer words, treated as approved, with their forms.\n\n"
                  + _software_section()]
    elif variant not in ("tiny", "traps"):
        raise ValueError(f"unknown prompt variant {variant}")
    return "\n\n".join(parts + [REWRITE_REMINDER])


def judge_prompt(lex, variant: str) -> str:
    if variant == "full":
        return build_system_prompt_v4(lex)
    parts = [STE_NOTE, _judge_section_v4(lex)] + ([JUDGE_TRAPS] if variant == "traps" else []) + [JUDGE_REMINDER]
    return "\n\n".join(parts)
