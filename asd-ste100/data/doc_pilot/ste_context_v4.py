"""STE material for the v4 teacher: the v3 prompt with the fixes from the v3 review of all 36.

- The task text says that a word the checker findings do not name is accepted,
  and lists what a rewrite dropped in v3: units, hedges, certainty and
  frequency words, evaluative adjectives, restrictive clauses, and the scope of
  a negation ("not ... by any" is "none").
- MODAL WORDS adds "it is advisable to", "is allowed to" and the hedges.
- A literal marker can move with its words; the order is no longer checked.
- Three invented worked examples show the v3 drops kept. They are not
  passages from any source file, so no input can be one of them.
- The MEANING CHECK asks for differences that change what a reader knows or
  must do, lists the v3 false alarms as kept meaning (proper -> correct,
  "etc." -> "and other", dropped emphasis words), and wants each difference as
  "'input words' -> 'rewrite words': what changes". Five invented examples
  join the five v3 ones.
- Flagged "even", "just" and "already" get a hint for the adverb sense, which
  the dictionary alternatives do not cover.
The prompt is AGPL-derived, so it is written only under ignored data/out.
"""

from __future__ import annotations

import hashlib
import json

from gates_v4 import CERTAINTY_CLASSES, MODAL_CLASSES_V4, units  # gates_v2 sets the path first
from checker import check, errors
from ste_context import EXAMPLES, ISSUES, LEX, ROOT, V1, _approved_section, _rules_section, _software_section, \
    _unapproved_section
from ste_context_v3 import (JUDGE_EXAMPLES, MODAL_GROUP, TASK_V3, _class_detail, _edit, _negation_detail, _sources,
                            describe_v3, participle, replaced_words)
from gates_v3 import CONDITIONS, dropped_words
from synthesize import SLOT, mask_literals, restore

TASK_V4 = _edit(TASK_V3, "Keep every marker exactly once and in the same order.",
                "Keep every marker exactly once. A marker can move when the word order changes, but keep it with the "
                "words that it belongs to.")
TASK_V4 = _edit(TASK_V4, "Do not add facts, examples, warnings, or explanations. Do not summarize.",
                "Do not add facts, examples, warnings, or explanations. Do not summarize.\n\n"
                "KEEP ALL THE INFORMATION. The checker findings for a passage name every word that you must change. "
                "A word that they do not name is accepted. If it carries meaning, do not remove it or put a simpler "
                "word in its place. Keep in particular:\n"
                "- each number and unit (\"in bytes per second\" -> \"in bytes for each second\");\n"
                "- each hedge, certainty or frequency word (\"presumably\", \"probably\", \"likely\", \"guaranteed\", "
                "\"always\", \"usually\", \"often\", \"sometimes\"), see MODAL WORDS;\n"
                "- each adjective that tells what type or how good (\"convenient\", \"reliable\", \"external\");\n"
                "- each clause that tells which item (\"the directory in which the files are kept\" -> \"the directory that "
                "contains the files\");\n"
                "- the scope of each negation: \"not done by any of the handlers\" means that no handler does it "
                "(\"none of the handlers does it\"). It does not mean \"not done by all of the handlers\".")
TASK_V4 = _edit(TASK_V4, "- may (permission) -> \"can\".",
                "- may (permission), is allowed to, or is permitted to -> \"can\". \"allows you to\" -> \"lets you\" or "
                "\"you can\".")
TASK_V4 = _edit(TASK_V4, "- may wish to, might want to, you are advised to, or should (advice)",
                "- may wish to, might want to, you are advised to, it is advisable to, it is best to, or should (advice)")
TASK_V4 = _edit(TASK_V4, "- could -> \"can\"; keep a hedge such as \"likely\" as \"possibly\".",
                "- could -> \"can\". Keep each hedge (\"presumably\", \"probably\", \"likely\"): keep the word, or "
                "write \"probably\" or \"possibly\". Keep each certainty word: \"is guaranteed to\" -> \"always\", or keep \"guaranteed\". "
                "Keep each frequency word: \"often\" -> \"frequently\"; \"usually\" and \"sometimes\" stay.")

# Invented passages (in no source file): (mode, passage with markers, rewrite, why). Each must pass verify_v4.
EXAMPLES_V4 = [
    ("DESC",
     "The reported rate is presumably an average in kilobytes per second, and it is usually refreshed after each block.",
     "The reported rate is presumably an average in kilobytes for each second. Usually, the rate is refreshed after "
     "each block.",
     "Only 'per' and the sentence length are in the checker findings, so every other word stays. The unit stays: "
     "'per' becomes 'for each'. The hedge 'presumably' and the frequency word 'usually' stay."),
    ("DESC",
     "Dirty buffers are guaranteed to reach the disk before the checkpoint is marked as done, and a backend is "
     "allowed to keep running even if the checkpoint takes longer than planned.",
     "The server always writes the dirty buffers to the disk before it marks the checkpoint as done. A backend can "
     "continue to run. This is also true if the checkpoint is longer than planned.",
     "'are guaranteed to' becomes 'always'. 'is allowed to' becomes 'can'. 'even if' becomes 'This is also true if', "
     "so the result does not change when the condition occurs."),
    ("DESC",
     "The compact format is not supported by any of the bundled exporters, so a simple custom exporter is required "
     "for the queue in which the events are kept.",
     "None of the bundled exporters supports the compact format. Thus, a simple custom exporter is necessary for the "
     "queue that keeps the events.",
     "'not ... by any of' becomes 'None of': no exporter supports the format. 'is required' becomes 'is necessary'. "
     "'simple' and the clause that tells which queue stay."),
]

MEANING_CHECK_V4 = """MEANING CHECK. A user message that starts with "MEANING CHECK" gives an input passage and a rewrite. Do not rewrite. Find each difference that changes what a reader knows or must do:
- a fact, agent, action or object that changes, or that the rewrite drops or adds;
- a changed condition, scope word (all, any, each, only, none), negation, or modal force (can, must, "we recommend that you");
- a dropped or changed quantity, unit, hedge ("probably"), certainty word ("guaranteed", "always"), frequency word ("usually"), contrast or concession ("although"), purpose or reason;
- a dropped qualifier that tells which items, which cases, what type or how good ("external", "unsupported", "convenient");
- a replacement word with a different sense in this passage;
- grammar that makes the meaning unclear.
Do not report these. They keep the meaning:
- a different word or phrase with the same sense in this passage: proper -> correct, appropriate -> applicable, succeeds -> completes correctly, as well as -> and, "etc." -> "and other", nothing else -> only, is lost -> you cannot find, if one is present -> if there is one, "X means that Y" -> Y;
- a dropped emphasis word that adds no fact: even, just, really, simply, perfectly, actually;
- shorter sentences, the active voice with the correct agent, the replacements in MODAL WORDS (may -> can, should -> must or "we recommend that you", whether -> if, once -> when), literal markers.
Before you report a difference, ask: does a reader of the rewrite know or do something different from a reader of the input? If not, do not report it.
The message lists the unapproved input words that the rewrite had to replace, and the input words that are not in the rewrite. They are only places to look. A correct replacement, or the same meaning in other words, is not a difference.
Reply with one JSON object only: {"differences": ["...", "..."]}. Write each difference as "'input words' -> 'rewrite words': what changes", or as "'input words' -> (dropped): what is lost". If the rewrite keeps the full meaning, reply {"differences": []}."""

# The v3 judge examples, with their differences in the v4 format.
JUDGE_DIFFERENCES_V4 = {
    "6f6a806f4cc6929523e56ad40dbb02785a06c36900d08f3e22d250e384893c3e":
        ["'if this option is used' -> (dropped): the rewrite says that curl shows the error in all cases.",
         "'unsupported' -> (dropped): the input is only about an encoding that curl does not support."],
    "1360567876cf0045031c930759ee32824a5d04241e6d6afbc1ba7ebd625e57f9":
        ["'handle' -> 'touch': 'touch' has a different sense. The input means to control the retry delays."],
    "329f02ff2c0ea773552727aede9ddcb60fbf505f1bdacd303919e3a9c1ad770f":
        ["'external' -> (dropped): the input is about a program that is not part of the server.",
         "'if you already use such a program with other server software' -> (dropped): the rewrite drops this condition."],
}
# Invented judge examples: (mode, input with markers, rewrite with markers, differences). Each rewrite passes verify_v4.
JUDGE_EXAMPLES_V4 = [
    ("DESC", "When the copy succeeds, the proper checksum is recorded as well as the name of the file.",
     "When the copy completes correctly, the server records the correct checksum and the name of the file.", []),
    ("DESC", "If the lock file is deleted, the process ID is lost, so monitors, schedulers, etc. cannot stop the process.",
     "If the lock file is erased, you cannot find the process ID. Thus, monitors, schedulers and other tools cannot "
     "stop the process.", []),
    ("DESC", "You can enable debug output for whole modules or even just single functions, but you must know perfectly "
             "well which functions are slow.",
     "You can start the debug output for full modules or for single functions. But you must know which functions "
     "are slow.",
     []),
    ("DESC", "The limits are not enforced by any of the default handlers, only by the handler in which the filter is installed.",
     "The default handlers do not obey all of the limits. Only the handler obeys them.",
     ["'not enforced by any of the default handlers' -> 'do not obey all of the limits': the input says that no default "
      "handler obeys the limits. The rewrite says that some limits are not obeyed.",
      "'the handler in which the filter is installed' -> 'the handler': the rewrite drops which handler."]),
    ("DESC", "A convenient and reliable procedure to rotate the files is to let the scheduler send the signal.",
     "A procedure to turn the files is to let the scheduler send the signal.",
     ["'convenient and reliable' -> (dropped): the input says that this procedure is easy to use and that it operates "
      "correctly every time.",
      "'rotate' -> 'turn': 'turn' has a different sense. The input means to replace old log files with new ones."]),
]


def judge_message_v4(source: str, target: str, mode: str, lex, extra_allowed=()) -> str:
    marked_source, _ = mask_literals(source)
    marked_target, _ = mask_literals(target)
    replaced = replaced_words(source, mode, lex, extra_allowed)
    missing = dropped_words(source, target, lex, extra_allowed)
    return (f"MEANING CHECK\nMode: {mode}\nInput: {marked_source}\nRewrite: {marked_target}\n"
            f"Unapproved input words that the rewrite had to replace: {', '.join(replaced) or 'none'}\n"
            f"Input words that are not in the rewrite: {', '.join(missing) or 'none'}")


def _example_section_v4() -> str:
    sources, blocks = _sources(), []
    for sid, target, why in EXAMPLES:
        row = sources[sid]
        blocks.append((row["mode"], mask_literals(row["text"])[0], target, why))
    blocks += EXAMPLES_V4
    return "\n\n".join(f"Example {n}. Mode: {mode}\nPassage: {passage}\nReply: "
                       + json.dumps({"rewrite": target, "uncertain": False}, ensure_ascii=False) + f"\nWhy: {why}"
                       for n, (mode, passage, target, why) in enumerate(blocks, 1))


def judge_examples_v4() -> list[tuple[str, str, str, list[str]]]:
    """(mode, input, rewrite, differences) for every judge example, as plain text with literals restored."""
    sources, out = _sources(), []
    for sid, rewrite, differences in JUDGE_EXAMPLES:
        row = sources[sid]
        out.append((row["mode"], row["text"], restore(rewrite, mask_literals(row["text"])[1]),
                    JUDGE_DIFFERENCES_V4.get(sid, differences)))
    return out + [(mode, text, rewrite, differences) for mode, text, rewrite, differences in JUDGE_EXAMPLES_V4]


def _judge_section_v4(lex) -> str:
    blocks = [f"Meaning check example {n}.\n{judge_message_v4(text, rewrite, mode, lex)}\nReply: "
              + json.dumps({"differences": differences}, ensure_ascii=False)
              for n, (mode, text, rewrite, differences) in enumerate(judge_examples_v4(), 1)]
    return MEANING_CHECK_V4 + "\n\n" + "\n\n".join(blocks)


def build_system_prompt_v4(lex) -> str:
    rows = [json.loads(l) for l in (LEX / "lexicon.jsonl").open(encoding="utf-8")]
    return "\n\n".join([
        TASK_V4,
        "WORKED EXAMPLES\n\n" + _example_section_v4(),
        _judge_section_v4(lex),
        "STE RULES (summaries)\n\n" + _rules_section(),
        "APPROVED WORDS. Each word is approved only as the part of speech of its section, with the forms in parentheses.\n\n"
        + _approved_section(rows),
        "SOFTWARE WORDS. A project list of computer words, treated as approved, with their forms.\n\n" + _software_section(),
        "UNAPPROVED WORDS. Do not use these words as the part of speech shown. Dictionary alternatives follow each word. "
        "Each alternative is for one sense of the word and is in its base form.\n\n" + _unapproved_section(rows),
        "REMINDER. Keep the meaning exactly, then use STE. Keep every literal marker exactly once. Keep every word that "
        "the checker findings do not name, if it carries meaning. For a rewrite, reply with one JSON object only: "
        "{\"rewrite\": \"...\", \"uncertain\": false}. For a MEANING CHECK, reply with one JSON object only: "
        "{\"differences\": [...]}.",
    ])


def context_manifest_v4(prompt: str) -> dict:
    files = [LEX / "lexicon.jsonl", LEX / "rules.json", LEX / "tech_words.txt", V1 / "candidate_sources.jsonl"]
    return {"system_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "system_prompt_chars": len(prompt),
            "example_source_ids": [e[0] for e in EXAMPLES],
            "judge_example_source_ids": [e[0] for e in JUDGE_EXAMPLES],
            "invented_examples": len(EXAMPLES_V4), "invented_judge_examples": len(JUDGE_EXAMPLES_V4),
            "input_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}


# The dictionary lists these words for a different sense (even: adjective, smooth).
ADVERB_HINTS = {
    "even": "As an adverb, keep its meaning in a different way: 'even if X' -> a second sentence 'This is also true "
            "if X'; 'even X' -> 'also X'. As emphasis only ('even just'), you can drop it.",
    "just": "'just' (only) -> 'only'; 'just' (immediately) -> 'immediately'. As emphasis only, you can drop it.",
    "already": "'has already started' -> 'is in progress' or 'started before'. If the meaning stays without it, drop it "
               "('if the file already exists' -> 'if the file exists').",
}


def describe_v4(violation, lex) -> str:
    text = describe_v3(violation, lex)
    hint = ADVERB_HINTS.get(violation.text.lower()) if violation.rule in ("R1.1", "R1.2") else None
    return text.replace(" If nothing fits,", f" {hint} If nothing fits,") if hint else text


def source_hints_v4(text: str, mode: str, lex, extra_allowed=()) -> str:
    _, literals = mask_literals(text)
    found = [v for v in errors(check(text, lex, mode, extra_allowed=set(extra_allowed)))
             if not any(v.text in lit for lit in literals)]
    if not found:
        return "The checker finds no errors outside the literals. Change only what STE needs."
    return "\n".join("- " + line for line in dict.fromkeys(describe_v4(v, lex) for v in found))


MODAL_ISSUES_V4 = {
    "modal_change:possibility": "Possibility or permission changed. Each may, might, can, could, possible, allow, "
                                "permit or let in the input must stay as can, possible, possibly or lets, and you must "
                                "not add one.",
}
CERTAINTY_HELP = {
    "hedge": "Keep each hedge as 'probably' or 'possibly', and do not add one.",
    "certainty": "Keep 'guaranteed', or write it as 'always'. Keep 'always'. Do not add one.",
    "frequency": "Keep 'usually' and 'sometimes'; 'often' becomes 'frequently'. Do not add one.",
}


def feedback_v4(issues: list[str], source: str, target: str, mode: str, lex, extra_allowed=(),
                differences=(), repeated: bool = False) -> str:
    lines = []
    for issue in issues:
        if issue.startswith("checker_errors"):
            lines += [describe_v4(v, lex) for v in errors(check(target, lex, mode, extra_allowed=set(extra_allowed)))]
        elif issue == "negation_change":
            lines.append(_negation_detail(source, target))
        elif issue.startswith("modal_change:"):
            group = MODAL_GROUP[issue.split(":", 1)[1]]
            lines.append(MODAL_ISSUES_V4.get(issue, ISSUES.get(issue, issue)) + " "
                         + _class_detail(source, target, MODAL_CLASSES_V4, group))
        elif issue.startswith("certainty_change:"):
            name = issue.split(":", 1)[1]
            lines.append(f"The {name} words changed. " + _class_detail(source, target, CERTAINTY_CLASSES, [name])
                         + " " + CERTAINTY_HELP[name])
        elif issue.startswith("unit_dropped:"):
            lost = issue.split(":", 1)[1].split(",")
            lines.append(f"Your rewrite drops the unit {', '.join(repr(u) for u in lost)}. The input has "
                         f"{', '.join(repr(u) for u in units(source))}. Keep each unit (\"per second\" -> \"for each second\").")
        elif issue.startswith("condition_change:"):
            word = issue.split(":", 1)[1]
            lines.append(f"The {word} words changed. " + _class_detail(source, target, CONDITIONS, [word])
                         + " Keep each one (whether becomes if; once becomes when; a contrast stays as but or although).")
        elif issue.startswith("verb_form:"):
            for phrase in issue.split(":", 1)[1].split(","):
                verb = phrase.split()[-1]
                lines.append(f"'{phrase}' is not correct English. A dictionary alternative is a base form: change it to "
                             f"fit the sentence, for example an active sentence ('you cannot {verb}')"
                             + (f" or '{participle(verb)}' after is or are" if participle(verb) else "") + ".")
        elif issue.startswith("meaning_check"):
            lines.append("A meaning check compared your rewrite with the input and found these differences: "
                         + " ".join(f"({n}) {d}" for n, d in enumerate(differences, 1))
                         + " Correct each one and keep the rewrite in STE.")
        else:
            lines.append(ISSUES.get(issue, issue))
    if repeated:
        lines.append("Your rewrite is the same as an earlier attempt that failed these checks. Write a different rewrite.")
    return "\n".join("- " + l for l in dict.fromkeys(lines))


def _context(marked: str, marker: str, width: int = 6) -> str:
    tokens = marked.split()
    for i, token in enumerate(tokens):
        if marker in token:
            return " ".join(tokens[max(0, i - width):i + width + 1])
    return marker


def marker_feedback_v4(rewrite: str, marked_source: str, count: int) -> str:
    """Name each missing, repeated or extra marker, and quote the input words around a missing one."""
    found, expected = SLOT.findall(rewrite), [f"[[L{i}]]" for i in range(count)]
    missing = [m for m in expected if m not in found]
    repeated = sorted({m for m in found if found.count(m) > 1})
    extra = [m for m in dict.fromkeys(found) if m not in expected]
    parts = [f"Literal markers: the input has {' '.join(expected) or 'none'}. Your rewrite has {' '.join(found) or 'none'}."]
    for m in missing:
        parts.append(f"It drops {m}. In the input, {m} is in \"{_context(marked_source, m)}\". Put {m} back with "
                     "those words.")
    if repeated:
        parts.append(f"It repeats {' '.join(repeated)}. Use each marker once.")
    if extra:
        parts.append(f"It adds {' '.join(extra)}, which the input does not have.")
    return " ".join(parts)
