"""STE material for the v3 teacher: the v2 prompt with the fixes from the v2 review.

- The task text says that a dictionary alternative is for one sense and is a
  base form, and that every qualifier keeps its meaning.
- A MEANING CHECK task in the same system prompt lets the teacher compare a
  passing rewrite with its input before it is accepted. It is the same model,
  so it is a second look, not an independent judge.
- A flagged word that is an -ing or -ed form of an approved or software word
  ("matching", software verb "match") gets those forms in its hint, before the
  sense-blind dictionary alternatives.
- Retry feedback names the words that each failed gate counted, and says when
  a retry repeats an earlier attempt.
The prompt is AGPL-derived, so it is written only under ignored data/out.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "data"), str(ROOT / "checker"), str(ROOT / "tokenizer"), str(Path(__file__).parent)]
from checker import check, errors, split_sentences  # noqa: E402
from gates_v2 import MODAL_CLASSES, CONDITIONS  # noqa: E402
from gates_v3 import (ROWS, TECH, class_hits, dropped_words, lemma_candidates, negation_hits,  # noqa: E402
                      negation_slack, participle)
from gen import plain  # noqa: E402
from ste_context import (EXAMPLES, ISSUES, LEX, POS, TASK, V1, _approved_section, _example_section,  # noqa: E402
                         _rules_section, _software_section, _unapproved_section)
from synthesize import SLOT, mask_literals, restore  # noqa: E402


def _edit(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"v2 task text changed: {old[:40]!r}")
    return text.replace(old, new)


TASK_V3 = _edit(TASK, "Keep qualifiers that limit scope: an adjective or phrase that says which items, which cases or how many.",
                "Keep every qualifier: each adjective, adverb or phrase that says which items, which cases, how many, "
                "what type or how good (for example \"temporary\", \"optional\", \"probably\"). If a qualifier is an "
                "unapproved word, keep its meaning with approved words. Words that are in no list (technical nouns "
                "and some adjectives) are accepted by the checker: keep them when they carry meaning.")
TASK_V3 = _edit(TASK_V3, "Each one has dictionary alternatives, but an alternative is for one sense of the word. Use an alternative "
                "only if it keeps the meaning in this passage. If no alternative fits, write the sentence in a different way (rule R9.1).",
                "Each one has dictionary alternatives, but an alternative is for one sense of the word, and it is in its "
                "base form. Use an alternative only if it has the same sense in this passage, and change its form to fit "
                "the sentence (\"curl abandons the transfer\" -> \"curl stops the transfer\"). If the unapproved word is an "
                "-ing or -ed form of an approved word or a software word, use an approved form of that word "
                "(\"the parsing step\" -> \"the step that parses the text\"). If no alternative has the same sense, keep "
                "the meaning with other approved words, or write the sentence in a different way (rule R9.1). Never drop "
                "the meaning of an unapproved word.")
TASK_V3 = _edit(TASK_V3, 'Reply with one JSON object only: {"rewrite": "...", "uncertain": false}.',
                'Reply with one JSON object only: {"rewrite": "...", "uncertain": false}.\n\n'
                'A user message that starts with "MEANING CHECK" asks for a different task. See MEANING CHECK.')

MEANING_CHECK = """MEANING CHECK. A user message that starts with "MEANING CHECK" gives an input passage and a rewrite. Do not rewrite. Compare the rewrite with the input and find each difference in meaning:
- information in the input that the rewrite drops, for example an adjective, adverb or phrase that tells which items, which cases, how many, what type or how good;
- information that the rewrite adds;
- a replacement word with a different sense from the input word (for example a dictionary alternative for a different sense);
- a changed agent, object, quantity, condition, scope word, negation, hedge ("probably", "usually"), contrast or concession ("although"), purpose or reason;
- grammar that makes the meaning unclear or different.
These changes keep the meaning, so do not report them: an approved word with the same sense in place of an unapproved word; a long sentence divided into shorter ones; the active voice with the correct agent in place of the passive voice; the replacements in MODAL WORDS (may -> can, should -> must or "we recommend that you", whether -> if); "nothing else" -> "only"; literal markers.
The message lists the unapproved input words that the rewrite replaced, and the input words that are not in the rewrite. Examine each of them.
Reply with one JSON object only: {"differences": ["...", "..."]}. Write each difference as one short sentence that names the words. If the rewrite keeps the full meaning, reply {"differences": []}."""

# (worked example source id, rewrite with markers, differences). The ids are the
# v2 worked examples, so they stay out of the teacher inputs, controls and holdout.
JUDGE_EXAMPLES = [
    ("b254c78195d7cc7e3d103e7b912bbed3768fb411f389fb0e55a4f37844168520",
     "This option sets a limit only for the connection phase. If curl connects in the specified period, it continues. "
     "If curl does not connect in that period, curl stops.", []),
    ("060a41ccedcf8202f58bfb66cc8736e76b0aa72642e87e8cd387fe50b712435d",
     "Use [[L0]] to also set a maximum time for each time that curl tries a transfer.", []),
    ("6f6a806f4cc6929523e56ad40dbb02785a06c36900d08f3e22d250e384893c3e",
     "If the server sends an encoding, curl shows an error.",
     ["The rewrite drops the condition 'if this option is used'.",
      "The rewrite drops 'unsupported': the input is only about an encoding that curl does not support."]),
    ("1360567876cf0045031c930759ee32824a5d04241e6d6afbc1ba7ebd625e57f9",
     "To touch the retry intervals of the server that can be longer than you expect, you can use [[L0]] to set a limit "
     "for the total retry time.",
     ["'touch' has a different sense from 'handle': the input means to control the retry delays, not to touch them."]),
    ("329f02ff2c0ea773552727aede9ddcb60fbf505f1bdacd303919e3a9c1ad770f",
     "As an alternative, you can use a log rotation program.",
     ["The rewrite drops 'external'.",
      "The rewrite drops the condition: if you already use such a program with other server software."]),
]


def _sources() -> dict[str, dict]:
    return {json.loads(l)["id"]: json.loads(l) for l in (V1 / "candidate_sources.jsonl").read_text(encoding="utf-8").splitlines()}


def replaced_words(source: str, mode: str, lex, extra_allowed=()) -> list[str]:
    """Rejected source words, including those the checker downgrades to info ('to handle')."""
    _, literals = mask_literals(source)
    return list(dict.fromkeys(v.text.lower() for v in check(source, lex, mode, extra_allowed=set(extra_allowed))
                              if v.rule in ("R1.1", "R1.2") and not any(v.text in lit for lit in literals)
                              and any(c in lex.rejected for c in lemma_candidates(v.text))))


def judge_message(source: str, target: str, mode: str, lex, extra_allowed=()) -> str:
    marked_source, _ = mask_literals(source)
    marked_target, _ = mask_literals(target)
    replaced = replaced_words(source, mode, lex, extra_allowed)
    missing = dropped_words(source, target, lex, extra_allowed)
    return (f"MEANING CHECK\nMode: {mode}\nInput: {marked_source}\nRewrite: {marked_target}\n"
            f"Unapproved input words that the rewrite replaced (check the sense of each replacement): "
            f"{', '.join(replaced) or 'none'}\n"
            f"Input words that are not in the rewrite (check that their meaning stays): {', '.join(missing) or 'none'}")


def _judge_section(lex) -> str:
    sources, blocks = _sources(), []
    for n, (sid, rewrite, differences) in enumerate(JUDGE_EXAMPLES, 1):
        row = sources[sid]
        target = restore(rewrite, mask_literals(row["text"])[1])
        reply = json.dumps({"differences": differences}, ensure_ascii=False)
        blocks.append(f"Meaning check example {n}.\n{judge_message(row['text'], target, row['mode'], lex)}\nReply: {reply}")
    return MEANING_CHECK + "\n\n" + "\n\n".join(blocks)


def build_system_prompt_v3(lex) -> str:
    rows = [json.loads(l) for l in (LEX / "lexicon.jsonl").open(encoding="utf-8")]
    return "\n\n".join([
        TASK_V3,
        "WORKED EXAMPLES\n\n" + _example_section(),
        _judge_section(lex),
        "STE RULES (summaries)\n\n" + _rules_section(),
        "APPROVED WORDS. Each word is approved only as the part of speech of its section, with the forms in parentheses.\n\n"
        + _approved_section(rows),
        "SOFTWARE WORDS. A project list of computer words, treated as approved, with their forms.\n\n" + _software_section(),
        "UNAPPROVED WORDS. Do not use these words as the part of speech shown. Dictionary alternatives follow each word. "
        "Each alternative is for one sense of the word and is in its base form.\n\n" + _unapproved_section(rows),
        "REMINDER. Keep the meaning exactly, then use STE. Keep every literal marker in order. For a rewrite, reply with "
        "one JSON object only: {\"rewrite\": \"...\", \"uncertain\": false}. For a MEANING CHECK, reply with one JSON "
        "object only: {\"differences\": [...]}.",
    ])


def context_manifest_v3(prompt: str) -> dict:
    files = [LEX / "lexicon.jsonl", LEX / "rules.json", LEX / "tech_words.txt", V1 / "candidate_sources.jsonl"]
    return {"system_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "system_prompt_chars": len(prompt),
            "example_source_ids": [e[0] for e in EXAMPLES],
            "judge_example_source_ids": [e[0] for e in JUDGE_EXAMPLES],
            "input_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}


def _approved_forms() -> dict[str, tuple[str, list[str]]]:
    forms: dict[str, tuple[str, list[str]]] = {}
    for r in ROWS:
        if r["status"] == "approved":
            forms.setdefault(r["lemma"], (POS.get(r["pos"], r["pos"]), r["forms"]))
    for name, lines in TECH.items():
        kind = "software " + ("verb" if name.startswith("verb") else "noun" if name.startswith("noun")
                              else "adjective" if name.startswith("adjective") else "word")
        for line in lines:
            forms.setdefault(line[0], (kind, line))
    return forms


APPROVED_FORMS = _approved_forms()


def sibling_forms(word: str) -> tuple[str, str, list[str]] | None:
    """An approved or software word that the flagged word is a form of: ('match', 'software verb', [...])."""
    w = word.lower()
    for lemma in lemma_candidates(w)[1:]:
        if lemma in APPROVED_FORMS and w not in APPROVED_FORMS[lemma][1]:
            kind, forms = APPROVED_FORMS[lemma]
            return lemma, kind, forms
    return None


def _alternatives(lex, word: str) -> str:
    for cand in lemma_candidates(word):
        if cand in lex.rejected:
            return ", ".join(dict.fromkeys(lex.rejected[cand]))
    return ""


def describe_v3(violation, lex) -> str:
    if violation.rule not in ("R1.1", "R1.2"):
        return plain(violation)
    alts, sibling = _alternatives(lex, violation.text), sibling_forms(violation.text)
    parts = [f"'{violation.text}' is not an approved word."]
    if sibling:
        lemma, kind, forms = sibling
        parts.append(f"The {kind} '{lemma}' is approved in these forms: {', '.join(forms)}. "
                     "Use one of them if it has the same sense"
                     + (" (for example in a clause with 'that')." if kind.endswith("verb") else "."))
    if alts:
        parts.append(("Other dictionary alternatives" if sibling else "Dictionary alternatives")
                     + f", each for one sense and in its base form: {alts}. Use one only if it has the same sense here, "
                     "and change its form to fit the sentence.")
    parts.append("If nothing fits, keep the meaning with other approved words. Do not drop the meaning.")
    return " ".join(parts)


def source_hints_v3(text: str, mode: str, lex, extra_allowed=()) -> str:
    _, literals = mask_literals(text)
    found = [v for v in errors(check(text, lex, mode, extra_allowed=set(extra_allowed)))
             if not any(v.text in lit for lit in literals)]
    if not found:
        return "The checker finds no errors outside the literals. Change only what STE needs."
    return "\n".join("- " + line for line in dict.fromkeys(describe_v3(v, lex) for v in found))


def _quoted(items) -> str:
    items = list(items)
    return ", ".join(f"'{w}'" for w in items) if items else "none"


def _negation_detail(source: str, target: str) -> str:
    s, t, slack = negation_hits(source), negation_hits(target), negation_slack(source)
    lines = [f"Negations: the input has {_quoted(s)}. Your rewrite has {_quoted(t)}."]
    added = Counter(t) - Counter(s)
    if len(t) > len(s) + len(slack):
        sentences = [x for x in split_sentences(target) if any(a in negation_hits(x) for a in added)]
        lines.append("Your rewrite adds a negation that the input does not state, in: "
                     + " ".join(f'"{x}"' for x in sentences) + " Remove it, or write that sentence again.")
    else:
        lines.append(f"Your rewrite drops {_quoted((Counter(s) - Counter(t)).elements())}. Keep each negation of the input "
                     "(for example, 'unsupported' becomes 'that ... does not support').")
    if slack:
        lines.append(f"The input word {_quoted(slack)} can become one 'not' (for example 'cannot find').")
    return " ".join(lines)


def _class_detail(source: str, target: str, classes, names: list[str]) -> str:
    s, t = class_hits(source, classes), class_hits(target, classes)
    label = lambda c: c.replace("_", " ")
    show = lambda hits: "; ".join(f"{label(c)}: {_quoted(hits[c])}" for c in names if hits[c]) or "none"
    return f"The input has {show(s)}. Your rewrite has {show(t)}."


MODAL_GROUP = {"possibility": ["possibility"], "future": ["future"],
               "advice/obligation": ["advice", "obligation", "advice_or_obligation"]}


def feedback_v3(issues: list[str], source: str, target: str, mode: str, lex, extra_allowed=(),
                differences=(), repeated: bool = False) -> str:
    lines = []
    for issue in issues:
        if issue.startswith("checker_errors"):
            lines += [describe_v3(v, lex) for v in errors(check(target, lex, mode, extra_allowed=set(extra_allowed)))]
        elif issue == "negation_change":
            lines.append(_negation_detail(source, target))
        elif issue.startswith("modal_change:"):
            group = MODAL_GROUP[issue.split(":", 1)[1]]
            lines.append(ISSUES.get(issue, issue) + " " + _class_detail(source, target, MODAL_CLASSES, group))
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


def marker_feedback(rewrite: str, count: int) -> str:
    """Feedback for a rewrite whose literal markers do not match the input (v1 and v2 stopped without a retry)."""
    found, expected = SLOT.findall(rewrite), [f"[[L{i}]]" for i in range(count)]
    missing = [m for m in expected if m not in found]
    repeated = sorted({m for m in found if found.count(m) > 1})
    extra = [m for m in dict.fromkeys(found) if m not in expected]
    parts = [f"Literal markers: the input has {' '.join(expected) or 'none'}, in this order. "
             f"Your rewrite has {' '.join(found) or 'none'}."]
    if missing:
        parts.append(f"It drops {' '.join(missing)}. Put each one back, with the meaning it has in the input.")
    if repeated:
        parts.append(f"It repeats {' '.join(repeated)}. Use each marker once.")
    if extra:
        parts.append(f"It adds {' '.join(extra)}, which the input does not have.")
    if not (missing or repeated or extra):
        parts.append("The order changed. Keep the markers in the input order.")
    return " ".join(parts)


def parse_judge(raw: str) -> list[str]:
    value = raw.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I).strip()
    obj = json.loads(value)
    if set(obj) != {"differences"} or not isinstance(obj["differences"], list) \
            or not all(isinstance(d, str) for d in obj["differences"]):
        raise ValueError("wrong meaning-check JSON schema")
    return [d.strip() for d in obj["differences"] if d.strip()]
