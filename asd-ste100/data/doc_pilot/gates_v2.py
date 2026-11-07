"""Meaning-class gates for the v2 teacher pilot.

v1 required the same modal and negation tokens in source and target. The STE
checker rejects may, should, would and never, so a compliant rewrite of a
passage that used them could never pass. v2 counts what these words do: a
source `may` is satisfied by `can` or `possibly`, `it is not possible` by
`cannot`, `whether` by `if`. The lexicon's own alternatives define the pairs.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "data"), str(ROOT / "checker"), str(ROOT / "tokenizer"), str(Path(__file__).parent)]
from checker import check, errors, split_sentences  # noqa: E402
from gen import first_word, sane  # noqa: E402
from quality_gates import target_issues, unsafe_changes  # noqa: E402
from synthesize import NUMBER, OPTION  # noqa: E402

# Ordered: phrases are consumed before the single words inside them.
MODAL_CLASSES = [
    ("advice", r"may wish to|might wish to|may want to|might want to|ought to|it is a good idea|"
               r"(?:it is|it's) better to|recommend(?:s|ed)?|advised?|advises"),
    ("obligation", r"must|has to|have to|had to|needs? to|needed to|needs?|needed|necessary"),
    ("possibility", r"cannot|can't|can|could|couldn't|may|might|possible|possibly|impossible|unable|able"),
    ("future", r"will|won't|shall"),
    # `should` is advice or obligation by sense; either satisfies it.
    ("advice_or_obligation", r"should|shouldn't"),
]
# `would` is not counted: STE replaces it with can, the present tense, or nothing.
CONDITIONS = [
    ("if", r"in case|whether|if"),
    ("when", r"when"),
    ("unless", r"unless"),
    ("before", r"before"),
    ("after", r"after"),
    ("only", r"only"),
    ("at least", r"at least"),
    ("at most", r"at most"),
    # The checker rejects `except` without an alternative and `but` is the usual
    # rewrite, so contrast words are counted together.
    ("contrast", r"except|but|however|although|though|whereas|unlike"),
]
NEGATORS = re.compile(r"\b(?:not|no|never|none|nothing|nobody|neither|nor|without|cannot|\w+n't)\b", re.I)
NEG_PREFIX = re.compile(r"\b(?:un|non)-?[a-z]{4,}\b|\b(?:impossible|incorrect|invalid|inactive|incomplete|insufficient)\b", re.I)
NOT_NEGATIVE = set("""under underneath undergo undertake understand understands understood until unit units unite
united union unique universal universe unix uniform unicode unless unlike""".split())


def _counts(text: str, classes) -> Counter:
    rest, found = text.lower(), Counter()
    for name, pattern in classes:
        regex = re.compile(rf"\b(?:{pattern})\b")
        found[name] += len(regex.findall(rest))
        rest = regex.sub(" ", rest)
    return found


def negations(text: str) -> int:
    prefixed = [w for w in NEG_PREFIX.findall(text) if w.lower().replace("-", "") not in NOT_NEGATIVE]
    return len(NEGATORS.findall(text)) + len(prefixed)


def modal_change(source: str, target: str) -> list[str]:
    s, t = _counts(source, MODAL_CLASSES), _counts(target, MODAL_CLASSES)
    changed = [c for c in ("possibility", "future") if s[c] != t[c]]
    pool = ("advice", "obligation", "advice_or_obligation")
    if sum(t[c] for c in pool) != sum(s[c] for c in pool) or any(t[c] < s[c] for c in pool[:2]):
        changed.append("advice/obligation")
    return changed


def condition_change(source: str, target: str) -> list[str]:
    s, t = _counts(source, CONDITIONS), _counts(target, CONDITIONS)
    return [name for name, _ in CONDITIONS if s[name] != t[name]]


def _base_verbs() -> set[str]:
    lex_dir = ROOT / "lexicon"
    verbs = {json.loads(l)["lemma"] for l in (lex_dir / "lexicon.jsonl").open(encoding="utf-8")
             if json.loads(l)["pos"] == "v"}
    section = ""
    for line in (lex_dir / "tech_words.txt").open(encoding="utf-8"):
        if line.startswith("#"):
            section = line.strip("# \n").lower()
        elif line.split() and (section.startswith("verb") or ("command" in section and len(line.split()) >= 3)):
            verbs.add(line.split()[0])
    return verbs


BASE_VERBS = _base_verbs()


def imperatives(text: str) -> int:
    return sum(1 for s in split_sentences(text) if first_word(s) in BASE_VERBS)


def verify_v2(source: str, target: str, mode: str, lex, uncertain: bool, extra_allowed=()) -> list[str]:
    problems = []
    if uncertain:
        problems.append("teacher_uncertain")
    if not target or target == source:
        problems.append("unchanged_or_empty")
    if not sane(source, target, lex, mode):
        problems.append("source_target_sanity")
    if unsafe_changes(source, target):
        problems.append("protected_change")
    if Counter(NUMBER.findall(source)) != Counter(NUMBER.findall(target)):
        problems.append("number_change")
    if Counter(OPTION.findall(source)) != Counter(OPTION.findall(target)):
        problems.append("option_change")
    if negations(source) != negations(target):
        problems.append("negation_change")
    problems += ["modal_change:" + c for c in modal_change(source, target)]
    problems += ["condition_change:" + c for c in condition_change(source, target)]
    if mode == "DESC" and imperatives(target) > imperatives(source):
        problems.append("description_became_instruction")
    ratio = len(target.split()) / max(1, len(source.split()))
    if not 0.55 <= ratio <= 2.0:
        problems.append("length_ratio")
    if any(issue["action"] == "exclude" for issue in target_issues(target)):
        problems.append("damaged_target")
    violations = errors(check(target, lex, mode, extra_allowed=set(extra_allowed)))
    if violations:
        problems.append("checker_errors:" + ",".join(v.rule for v in violations[:8]))
    return list(dict.fromkeys(problems))
