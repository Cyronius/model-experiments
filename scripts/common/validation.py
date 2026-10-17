"""Title validation and normalization shared by generation, eval, and bake-off."""

import re
import unicodedata

MIN_WORDS, MAX_WORDS = 3, 8

BANNED_PREFIX_RE = re.compile(r"^\s*(title\s*:|here('s| is)|sure[,!]|a title)", re.I)
QUOTE_CHARS = "\"'“”‘’`"
# any quote mark except an apostrophe inside a word (Don't, Children's)
STRAY_QUOTE_RE = re.compile(r"[\"“”‘`]|(?<!\w)['’]|['’](?!\w)")
SMALL_WORDS = {"a", "an", "the", "and", "or", "nor", "but", "for", "of", "in", "on", "at", "to", "vs", "via", "with"}


def keeps_case(word: str) -> bool:
    """Acronyms, versions and inner-capital words (SQL, v2, iPhone, mRNA) are never re-cased."""
    return any(c.isdigit() for c in word) or any(c.isupper() for c in word[1:])


def has_emoji(text: str) -> bool:
    return any(unicodedata.category(c) == "So" or 0x1F000 <= ord(c) <= 0x1FAFF for c in text)


def clean_title(raw: str, clamp: bool = True) -> str:
    """Normalize a candidate title: strip wrappers, quotes, trailing punctuation, Title Case,
    clamp to MAX_WORDS. Single source of truth for the word-count clamp -- callers should not
    duplicate a `[:8]`/`.slice(0, 8)` step of their own. Teacher titling passes clamp=False so an
    over-long title fails validate_title and is retried instead of being cut mid-phrase."""
    t = raw.strip().replace("’", "'")
    t = re.sub(r"</?think>.*?(</think>|$)", "", t, flags=re.S).strip()  # qwen think tags
    t = t.splitlines()[0].strip() if t else t
    t = t.strip(QUOTE_CHARS + " ")
    t = re.sub(r"^\s*title\s*:\s*", "", t, flags=re.I)
    # strip leaked format instructions, e.g. "3-8 Words:" or "3-8 Word Title:"
    t = re.sub(r"^\s*\d+\s*-\s*\d+\s*w(?:or)?ds?(\s+title)?\s*:?\s*", "", t, flags=re.I)
    t = re.sub(r"[.!?:;,\s]+$", "", t)
    t = re.sub(r"\*+", "", t)  # markdown bold
    t = STRAY_QUOTE_RE.sub("", t)  # inner quotes: strip rather than reject
    t = re.sub(r"\s{2,}", " ", t).strip()
    words = t.split()[:MAX_WORDS] if clamp else t.split()
    cased = [
        w if keeps_case(w)
        else (w.lower() if w.lower() in SMALL_WORDS and 0 < i < len(words) - 1 else w[:1].upper() + w[1:])
        for i, w in enumerate(words)
    ]
    return " ".join(cased)


def validate_title(title: str, prompt: str) -> str | None:
    """Return None if valid, else a short reason string."""
    if not title:
        return "empty"
    words = title.split()
    if len(words) < MIN_WORDS:
        return f"too short ({len(words)} words)"
    if len(words) > MAX_WORDS:
        return f"too long ({len(words)} words)"
    if STRAY_QUOTE_RE.search(title):
        return "contains quotes"
    if "\n" in title:
        return "contains newline"
    if has_emoji(title):
        return "contains emoji"
    if BANNED_PREFIX_RE.match(title):
        return "banned prefix"
    if re.match(r"^\d+\s*-\s*\d+\b", title):
        return "leaked format range"
    # only a defect when copying from a long paste; on short prompts the natural
    # title often IS a substring of the prompt
    if len(prompt) > 200 and len(title) > 20 and title.lower() in prompt.lower():
        return "verbatim prompt substring"
    return None
