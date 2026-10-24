"""Lossless surface tokens for data alignment; this is not the model's BPE tokenizer."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
import unicodedata


# Longest spans go first. Sentence punctuation is outside URLs and paths.
PROTECTED_RE = re.compile(
    r"```[\s\S]*?```|`[^`\n]+`"
    r"|https?://[^\s<>`]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"
    r"|(?:[A-Za-z]:[\\/]|\.{0,2}/|\\\\)[^\s<>`]+"
    r"|\b[\w.-]+(?:/[\w.-]+)+"
    r"|\b\w+(?:\.\w+)+\b%?"
    r"|\b\w*_\w*\b|\b[a-z]+[A-Z]\w*\b"
    r"|\b\w*\d\w*(?:[.,:]\d+)*(?:%|\b)"
)
WORD_RE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*|\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    protected: bool = False


def tokenize(text: str) -> list[Token]:
    tokens = []
    end = 0
    for match in PROTECTED_RE.finditer(text):
        start, stop = match.span()
        # Keep closing sentence marks out of an unquoted URL/path.
        if not match.group().startswith('`'):
            stop = start + len(match.group().rstrip('.,;:!?)]}'))
        if stop <= start:
            continue
        tokens.extend(Token(m.group(), m.start(), m.end())
                      for m in WORD_RE.finditer(text, end, start))
        tokens.append(Token(text[start:stop], start, stop, True))
        end = stop
    tokens.extend(Token(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(text, end))
    return tokens


def protected_values(text: str) -> Counter:
    return Counter(t.text for t in tokenize(text) if t.protected)


def normalized(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def replace(text: str, start: int, end: int, value: str) -> str:
    return text[:start] + value + text[end:]


def match_case(old: str, new: str) -> str:
    if old.isupper():
        return new.upper()
    if old[:1].isupper():
        return new[:1].upper() + new[1:]
    return new
