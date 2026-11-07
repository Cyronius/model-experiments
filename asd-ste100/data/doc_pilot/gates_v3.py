"""Gates for the v3 teacher pilot: the v2 meaning classes plus the fixes from the v2 review.

- "nothing else", "no other" and "nothing but" mean "only". They count as the
  scope word, not as a negation.
- A source word whose dictionary alternatives contain a negation (lost -> not
  find, fail -> if ... not) may become one explicit negation in the target.
  It may also be "cannot" ("you cannot find") without counting as added possibility.
- A source "once" (conjunction) may become "when".
- A dictionary alternative pasted as a bare verb after a form of "be"
  ("is not find" for "lost") is damage.
- `dropped_words` lists source words that the checker accepts but the target
  does not keep (unlisted adjectives such as "convenient", technical and
  software nouns). It is not a gate: on the v1/v2 attempts it also flagged
  correct paraphrases ("individual" -> "each"), so the meaning check gets the
  list to judge, and the record keeps it as a review flag.
The helpers return the words they found, so the retry feedback can name them.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEX_DIR = ROOT / "lexicon"
sys.path[:0] = [str(ROOT / "data"), str(ROOT / "checker"), str(ROOT / "tokenizer"), str(Path(__file__).parent)]
from checker import PROTECT_RE, WORD_RE, check, errors, plural_forms  # noqa: E402
from gates_v2 import CONDITIONS, MODAL_CLASSES, NEG_PREFIX, NEGATORS, NOT_NEGATIVE, imperatives  # noqa: E402
from gen import sane  # noqa: E402
from quality_gates import target_issues, unsafe_changes  # noqa: E402
from synthesize import NUMBER, OPTION, SLOT, mask_literals  # noqa: E402

ONLY_EQUIV = re.compile(r"\b(?:nothing else|nobody else|no one else|no other|none other than|nothing but)\b", re.I)
NEGATING_ALT = re.compile(r"\b(?:not|no|none|without)\b")
# The dictionary gives 'already' the alternative 'no other', which is a different sense.
NO_NEGATION_SLACK = {"already"}
BE_FORMS = {"is", "are", "was", "were", "be"}
PLACEHOLDER = "\x00p\x00"


def _tech_sections() -> dict[str, list[list[str]]]:
    sections: dict[str, list[list[str]]] = {}
    name = ""
    for line in (LEX_DIR / "tech_words.txt").open(encoding="utf-8"):
        if line.startswith("#"):
            name = line.strip("# \n").lower()
        elif line.split():
            sections.setdefault(name, []).append(line.split())
    return sections


ROWS = [json.loads(line) for line in (LEX_DIR / "lexicon.jsonl").open(encoding="utf-8")]
TECH = _tech_sections()
TECH_NOUNS = {form for name, lines in TECH.items() if name.startswith("noun") for line in lines for form in line}
NEGATING = {r["lemma"] for r in ROWS if r["status"] == "rejected" and r["lemma"] not in NO_NEGATION_SLACK
            and any(NEGATING_ALT.search(a["lemma"]) for a in r["alternatives"])}


def _verb_forms() -> dict[str, set[str]]:
    forms: dict[str, set[str]] = {}
    for r in ROWS:
        if r["status"] == "approved" and r["pos"] == "v":
            forms.setdefault(r["lemma"], set()).update(r["forms"])
    for name, lines in TECH.items():
        if name.startswith("verb"):
            for line in lines:
                forms.setdefault(line[0], set()).update(line)
    return forms


VERB_FORMS = _verb_forms()
NON_VERB_LEMMAS = ({r["lemma"] for r in ROWS if r["status"] == "approved" and r["pos"] != "v"} | TECH_NOUNS
                   | {f for name, lines in TECH.items() if name.startswith("adjective") for line in lines for f in line})
# Verbs whose base form cannot follow "is": they have a separate past participle.
BARE_VERBS = {v for v, fs in VERB_FORMS.items() if v not in NON_VERB_LEMMAS | {"be", "can", "do", "have"}
              and any(f != v and f not in plural_forms(v) for f in fs)}
# Modal, condition, negation and contraction words have their own gates.
CLASS_WORDS = ({w for _, pattern in MODAL_CLASSES + CONDITIONS for w in re.findall(r"[a-z']+", pattern)}
               | {"else", "other", "wish", "want", "though", "whereas", "unlike", "otherwise"})


def normalize(text: str) -> str:
    return ONLY_EQUIV.sub("only", text)


def words(text: str, placeholders: bool = False) -> list[str]:
    """Lowercase words outside literals and protected spans, tokenized as the checker does."""
    masked = PROTECT_RE.sub(f" {PLACEHOLDER} ", SLOT.sub(f" {PLACEHOLDER} ", mask_literals(text)[0]))
    found = [w.lower() for w in WORD_RE.findall(masked.replace(PLACEHOLDER, "\x00P\x00"))]
    return [w for w in found if placeholders or (w != PLACEHOLDER and not w.replace(".", "").isdigit())]


def lemma_candidates(word: str) -> list[str]:
    w = word.lower()
    out = [w]
    if w.endswith("ies"):
        out.append(w[:-3] + "y")
    if w.endswith("es"):
        out.append(w[:-2])
    if w.endswith("s"):
        out.append(w[:-1])
    if w.endswith("ed"):
        out += [w[:-2], w[:-1], w[:-3]]
    if w.endswith("ing"):
        out += [w[:-3], w[:-3] + "e", w[:-4]]
    return list(dict.fromkeys(c for c in out if len(c) >= 2))


def class_hits(text: str, classes) -> dict[str, list[str]]:
    rest, found = normalize(text).lower(), {}
    for name, pattern in classes:
        regex = re.compile(rf"\b(?:{pattern})\b")
        found[name] = regex.findall(rest)
        rest = regex.sub(" ", rest)
    return found


def negation_hits(text: str) -> list[str]:
    text = normalize(text)
    prefixed = [w for w in NEG_PREFIX.findall(text) if w.lower().replace("-", "") not in NOT_NEGATIVE]
    return [w.lower() for w in NEGATORS.findall(text)] + [w.lower() for w in prefixed]


def negation_slack(source: str) -> list[str]:
    """Source words that the dictionary lets become one explicit negation."""
    toks = words(normalize(source))
    flat = " " + " ".join(toks) + " "
    found = [w for w in toks if not negation_hits(w) and any(c in NEGATING for c in lemma_candidates(w))]
    return found + [p for p in sorted(NEGATING) if " " in p and f" {p} " in flat]


def negation_change(source: str, target: str) -> bool:
    s, t = len(negation_hits(source)), len(negation_hits(target))
    return not s <= t <= s + len(negation_slack(source))


def modal_change(source: str, target: str) -> list[str]:
    s = Counter({k: len(v) for k, v in class_hits(source, MODAL_CLASSES).items()})
    t = Counter({k: len(v) for k, v in class_hits(target, MODAL_CLASSES).items()})
    # 'lost' -> 'you cannot find': a negating dictionary word may become 'cannot'.
    cannot = min(len(negation_slack(source)), len(re.findall(r"\b(?:cannot|can't)\b", target, re.I)))
    changed = [c for c in ("possibility", "future")
               if not s[c] <= t[c] <= s[c] + (cannot if c == "possibility" else 0)]
    pool = ("advice", "obligation", "advice_or_obligation")
    if sum(t[c] for c in pool) != sum(s[c] for c in pool) or any(t[c] < s[c] for c in pool[:2]):
        changed.append("advice/obligation")
    return changed


def condition_change(source: str, target: str) -> list[str]:
    s, t = class_hits(source, CONDITIONS), class_hits(target, CONDITIONS)
    once = len(re.findall(r"\bonce\b", source, re.I))
    return [name for name, _ in CONDITIONS
            if not len(s[name]) <= len(t[name]) <= len(s[name]) + (once if name == "when" else 0)]


def _stem(word: str) -> str:
    w = word.lower()
    for suffix in ("ies", "es", "s", "ed", "ing", "ly"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3 and not (suffix == "s" and w.endswith(("ss", "us", "is"))):
            w = w[:-len(suffix)] + ("y" if suffix == "ies" else "")
            break
    return w.rstrip("e")


def _same_stem(a: str, b: str) -> bool:
    return a == b or (min(len(a), len(b)) >= 5 and (a.startswith(b) or b.startswith(a)))


def keep_words(source: str, lex, extra_allowed=()) -> list[str]:
    """Source words that the checker accepts and an STE rewrite can keep."""
    extra = {w.lower() for w in extra_allowed}
    out = []
    for w in words(source):
        if len(w) < 3 or w in extra or w in CLASS_WORDS or "'" in w or negation_hits(w) == [w] or lex.suggest(w):
            continue
        if w in TECH_NOUNS or (w not in lex.allowed and not w.endswith(("ed", "ing"))):
            out.append(w)
    return list(dict.fromkeys(out))


def dropped_words(source: str, target: str, lex, extra_allowed=()) -> list[str]:
    stems = {_stem(w) for w in words(target)}
    dropped = []
    for w in keep_words(source, lex, extra_allowed):
        options = {_stem(w)}
        if NEG_PREFIX.fullmatch(w):
            options.add(_stem(re.sub(r"^(?:un|non)-?", "", w)))
        if not any(_same_stem(o, s) for o in options for s in stems):
            dropped.append(w)
    return dropped


def participle(verb: str) -> str:
    extra = sorted(f for f in VERB_FORMS.get(verb, ()) if f != verb and f not in plural_forms(verb))
    return extra[-1] if extra else ""


def alternative_verbs(source: str, lex) -> set[str]:
    """Bare verbs in the dictionary alternatives of the source's rejected words."""
    found = set()
    for w in words(source):
        for c in lemma_candidates(w):
            for alt in lex.rejected.get(c, ()):
                found.update(x for x in re.findall(r"[a-z]+", alt) if x in BARE_VERBS)
    return found


def bad_verb_forms(source: str, target: str, lex) -> list[str]:
    """'is not find': a pasted alternative after a form of 'be' (other bare verbs there can be adjectives)."""
    verbs = alternative_verbs(source, lex)
    low = words(target, placeholders=True)
    hits = []
    for i, w in enumerate(low[:-1]):
        if w in BE_FORMS:
            j = i + 2 if low[i + 1] == "not" and i + 2 < len(low) else i + 1
            if low[j] in verbs:
                hits.append(" ".join(low[i:j + 1]))
    return hits


def verify_v3(source: str, target: str, mode: str, lex, uncertain: bool, extra_allowed=()) -> list[str]:
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
