"""Gates for the v4 teacher pilot: the v3 gates plus the fixes from the v3 review of all 36.

- "advisable", "preferable" and "it is best to" are advice words. "require" is
  an obligation word: the dictionary gives "necessary" for it.
- "allow", "permit" and "let" are possibility words: "is allowed to run" is
  "can run", and "lets you" stays "lets you".
- A sentence that the rewrite divides can repeat its modal word ("you can
  restore ... You can also restore ..."): each extra sentence in the target
  allows one more of a modal class that the source has.
- Hedge, certainty and frequency words are meaning classes, as the modal words
  are: "presumably", "guaranteed" and "usually" must stay. "possibly" is the
  STE hedge (MODAL WORDS), so it satisfies a hedge and is not an added possibility.
- Each unit word of the source ("bytes", "seconds") must stay.
- Literal markers can change order. `restore` replaces each marker by its
  number, so only a missing, repeated or extra marker is an error.
"""

from __future__ import annotations

import re
from collections import Counter

# gates_v2 puts data/, checker/ and tokenizer/ on the path.
from gates_v2 import MODAL_CLASSES, imperatives
from gates_v3 import bad_verb_forms, class_hits, condition_change, negation_change, negation_slack, words
from checker import check, errors, split_sentences
from gen import sane
from quality_gates import target_issues, unsafe_changes
from synthesize import NUMBER, OPTION, SLOT

_V2 = dict(MODAL_CLASSES)
MODAL_CLASSES_V4 = [
    ("advice", _V2["advice"] + r"|advisable|preferable|(?:it is|it's) best to"),
    # 'require' is unapproved and its dictionary alternative is 'necessary'.
    ("obligation", _V2["obligation"] + r"|requires?|required|requiring"),
    ("possibility", _V2["possibility"] + r"|allow|allows|allowed|allowing|permit|permits|permitted|let|lets"),
    ("future", _V2["future"]),
    ("advice_or_obligation", _V2["advice_or_obligation"]),
]
CERTAINTY_CLASSES = [
    ("hedge", r"presumably|probably|probable|likely|unlikely|perhaps|maybe|apparently|possibly"),
    ("certainty", r"guaranteed|guarantees?|certainly|definitely|surely|always"),
    ("frequency", r"usually|generally|typically|often|frequently|sometimes|occasionally|rarely|seldom|commonly|"
                  r"in most cases|in general"),
]
UNITS = {"bit", "byte", "kilobyte", "megabyte", "gigabyte", "terabyte", "kb", "mb", "gb", "tb", "kib", "mib", "gib",
         "nanosecond", "microsecond", "millisecond", "second", "minute", "hour", "day", "week", "percent"}


def extra_sentences(source: str, target: str) -> int:
    return max(0, len(split_sentences(target)) - len(split_sentences(source)))


def _hits(text: str, classes) -> Counter:
    return Counter({k: len(v) for k, v in class_hits(text, classes).items()})


def possibly_for_hedge(source: str, target: str) -> int:
    """Target 'possibly' that stands for a source hedge such as 'likely' (MODAL WORDS)."""
    hedges = len(re.findall(r"\b(?:presumably|probably|probable|likely|perhaps|maybe|apparently)\b", source, re.I))
    return min(hedges, len(re.findall(r"\bpossibly\b", target, re.I)))


def modal_change(source: str, target: str) -> list[str]:
    s, t = _hits(source, MODAL_CLASSES_V4), _hits(target, MODAL_CLASSES_V4)
    split = extra_sentences(source, target)
    # 'lost' -> 'you cannot find': a negating dictionary word may become 'cannot'.
    cannot = min(len(negation_slack(source)), len(re.findall(r"\b(?:cannot|can't)\b", target, re.I)))
    # A divided sentence repeats each modal word of the source at most once.
    slack = {"possibility": cannot + possibly_for_hedge(source, target) + min(split, s["possibility"]),
             "future": min(split, s["future"])}
    changed = [c for c in ("possibility", "future") if not s[c] <= t[c] <= s[c] + slack[c]]
    pool = ("advice", "obligation", "advice_or_obligation")
    total_s, total_t = sum(s[c] for c in pool), sum(t[c] for c in pool)
    if not total_s <= total_t <= total_s + min(split, total_s) or any(t[c] < s[c] for c in pool[:2]):
        changed.append("advice/obligation")
    return changed


def certainty_change(source: str, target: str) -> list[str]:
    s, t = _hits(source, CERTAINTY_CLASSES), _hits(target, CERTAINTY_CLASSES)
    split = extra_sentences(source, target)
    # A source 'may' can become the hedge 'possibly'.
    may = len(re.findall(r"\b(?:may|might|could)\b", source, re.I))
    slack = {"hedge": may, "certainty": min(split, s["certainty"]), "frequency": min(split, s["frequency"])}
    return [c for c, _ in CERTAINTY_CLASSES if not s[c] <= t[c] <= s[c] + slack[c]]


def units(text: str) -> list[str]:
    found = []
    for w in words(text):
        unit = w if w in UNITS else w[:-1] if w.endswith("s") and w[:-1] in UNITS else None
        if unit:
            found.append(unit)
    return list(dict.fromkeys(found))


def dropped_units(source: str, target: str) -> list[str]:
    kept = set(units(target))
    return [u for u in units(source) if u not in kept]


def marker_problems(rewrite: str, count: int) -> bool:
    return Counter(SLOT.findall(rewrite)) != Counter(f"[[L{i}]]" for i in range(count))


def restore_any_order(marked: str, literals: list[str]) -> str:
    if marker_problems(marked, len(literals)):
        raise ValueError("literal placeholders missing, repeated, or extra")
    return SLOT.sub(lambda m: literals[int(m[0][3:-2])], marked)


def verify_v4(source: str, target: str, mode: str, lex, uncertain: bool, extra_allowed=()) -> list[str]:
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
    if negation_change(source, target):
        problems.append("negation_change")
    problems += ["modal_change:" + c for c in modal_change(source, target)]
    problems += ["condition_change:" + c for c in condition_change(source, target)]
    problems += ["certainty_change:" + c for c in certainty_change(source, target)]
    lost = dropped_units(source, target)
    if lost:
        problems.append("unit_dropped:" + ",".join(lost))
    if mode == "DESC" and imperatives(target) > imperatives(source):
        problems.append("description_became_instruction")
    ratio = len(target.split()) / max(1, len(source.split()))
    if not 0.55 <= ratio <= 2.0:
        problems.append("length_ratio")
    if any(issue["action"] == "exclude" for issue in target_issues(target)):
        problems.append("damaged_target")
    forms = bad_verb_forms(source, target, lex)
    if forms:
        problems.append("verb_form:" + ",".join(forms))
    violations = errors(check(target, lex, mode, extra_allowed=set(extra_allowed)))
    if violations:
        problems.append("checker_errors:" + ",".join(v.rule for v in violations[:8]))
    return list(dict.fromkeys(problems))
