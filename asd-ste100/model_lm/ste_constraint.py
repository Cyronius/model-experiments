"""Restrict LM decoding to STE words plus words copied from the source.

A word is a maximal run of letters, digits, `_` and apostrophes. Every completed word in the output must be
an STE output word (any case; from the student vocabulary data/out/fixes-v2/vocabulary) or a word of the
source (exact case, or with only the first letter's case changed) that the checker accepts on its own
(`copy_ok`): code, names, numbers and words the dictionary does not reject can be copied, a rejected word
such as "additional" cannot. Characters outside words (spaces,
punctuation, backticks) are free. The check is on characters, so it does not depend on how the LM
tokenizes a word.

Constraint v2 (generate.py --constraint v2) adds four things: words inside the source's protected spans
(code, paths, identifiers) can always be copied; STE output words that the checker rejects on their own
(may, any, should, ...) are left out by the caller; a word may not complete a rejected multiword
phrase (such as, according to, have to); and STE words must be lower case, Title case or UPPER case
(`strict_case`; v1 took any mix, so "qualitIeS" passed as "qualities").

At each step only the top-k candidates are checked (then a wider top-k); every other token is masked.
"""
from __future__ import annotations

import bisect
import json
import re
from pathlib import Path

import torch

WORD_CHARS = re.compile(r"[A-Za-z0-9_'’]+")
SPECIAL = re.compile(r"^(PAD|NULL|SEP|PROC|DESC|[A-Z]+_\d+)$")


def ste_words(vocab_dir: Path) -> set[str]:
    out = json.loads((Path(vocab_dir) / 'vocab.json').read_text())['output_tokens']
    return {w.lower() for w in out if not SPECIAL.match(w) and WORD_CHARS.fullmatch(w)}


def case_ok(w: str) -> bool:
    """Is w (a word or the start of one) lower case, Title case or UPPER case?"""
    return w == w.lower() or w == w.upper() or w == w[:1].upper() + w[1:].lower()


def case_variants(word: str) -> set[str]:
    v = {word}
    if word[:1].isalpha():
        v.add(word[0].upper() + word[1:])
        v.add(word[0].lower() + word[1:])
    return v


class Allowed:
    """The allowed words for one request: the STE set (case-free) plus source words (case-kept)."""

    def __init__(self, ste_sorted: list[str], ste_set: set[str], source: str, copy_ok=None,
                 protected: list[str] = (), strict_case: bool = False):
        self.ste_sorted, self.ste_set, self.strict_case = ste_sorted, ste_set, strict_case
        self.source = set()
        for w in WORD_CHARS.findall(source):
            if copy_ok is None or w.lower() in ste_set or copy_ok(w):
                self.source |= case_variants(w)
        for span in protected:
            self.source |= set(WORD_CHARS.findall(span))
        self.source_sorted = sorted(self.source)

    def complete(self, w: str) -> bool:
        return (w.lower() in self.ste_set and (not self.strict_case or case_ok(w))) or w in self.source

    def prefix(self, p: str) -> bool:
        low = p.lower()
        i = bisect.bisect_left(self.ste_sorted, low)
        if (i < len(self.ste_sorted) and self.ste_sorted[i].startswith(low)
                and (not self.strict_case or case_ok(p))):
            return True
        i = bisect.bisect_left(self.source_sorted, p)
        return i < len(self.source_sorted) and self.source_sorted[i].startswith(p)


def tail_word(text: str) -> str:
    """The unfinished word at the end of text ('' when text ends outside a word)."""
    m = re.search(r"[A-Za-z0-9_'’]+$", text)
    return m.group() if m else ''


def completed_words(allowed: Allowed, partial: str, piece: str) -> list[str] | None:
    """The words that `piece` finishes after unfinished word `partial`, or None if it cannot follow."""
    s = partial + piece
    runs = list(WORD_CHARS.finditer(s))
    done = []
    for j, m in enumerate(runs):
        last_open = j == len(runs) - 1 and m.end() == len(s)
        if last_open:
            if not allowed.prefix(m.group()):
                return None
        elif not allowed.complete(m.group()):
            return None
        else:
            done.append(m.group())
    return done


def extends_ok(allowed: Allowed, partial: str, piece: str) -> bool:
    """Can `piece` follow text whose unfinished last word is `partial`?"""
    return completed_words(allowed, partial, piece) is not None


def ends_phrase(before: list[str], done: list[str], phrases: set[tuple[str, ...]], longest: int) -> bool:
    """Does one of the newly completed words `done` (after words `before`) end a rejected phrase?"""
    seq = [w.lower() for w in before[-longest:] + done]
    for end in range(len(seq) - len(done) + 1, len(seq) + 1):
        for n in range(2, longest + 1):
            if end - n >= 0 and tuple(seq[end - n:end]) in phrases:
                return True
    return False


class SteLogitsProcessor:
    """HF LogitsProcessor. `sources[i]` is the source text of batch row i; prompts are left-padded to prompt_len."""

    def __init__(self, tokenizer, ste: set[str], sources: list[str], prompt_len: int, stop_ids: list[int],
                 k: int = 64, wide_k: int = 2048, copy_ok=None, protected: list[list[str]] | None = None,
                 phrases: set[tuple[str, ...]] | None = None, strict_case: bool = False):
        self.tok_text = token_texts(tokenizer)
        # added tokens (<think>, </think>, tool and media markers) are never text
        self.special = set(tokenizer.all_special_ids) | set(tokenizer.added_tokens_decoder)
        ste_sorted = sorted(ste)
        self.allowed = [Allowed(ste_sorted, ste, s, copy_ok, protected[n] if protected else (), strict_case)
                        for n, s in enumerate(sources)]
        self.phrases = phrases or set()
        self.longest = max((len(p) for p in self.phrases), default=0)
        self.text = [''] * len(sources)
        self.prompt_len, self.stop_ids, self.k, self.wide_k = prompt_len, list(stop_ids), k, wide_k
        self.seen = prompt_len
        self.fallbacks = [0] * len(sources)

    def _fits(self, i: int, partial: str, piece: str) -> bool:
        done = completed_words(self.allowed[i], partial, piece)
        if done is None:
            return False
        if done and self.phrases:
            before = WORD_CHARS.findall(self.text[i][:len(self.text[i]) - len(partial)])
            return not ends_phrase(before, done, self.phrases, self.longest)
        return True

    def _ok(self, i: int, partial: str, t: int) -> bool:
        if t in self.stop_ids:
            if t in self.special:
                return self._fits(i, partial, ' ')
            # a line-break token ends the answer: what comes before the break must still fit
            return self._fits(i, partial, self.tok_text[t].split('\n')[0] + ' ')
        if t in self.special:
            return False
        piece = self.tok_text[t]
        if not piece or '�' in piece:
            return False
        return self._fits(i, partial, piece)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        length = input_ids.shape[1]
        if length > self.seen:
            for i in range(input_ids.shape[0]):
                for t in input_ids[i, self.seen:length].tolist():
                    if t not in self.special:
                        self.text[i] += self.tok_text[t]
            self.seen = length
        out = torch.full_like(scores, float('-inf'))
        for i in range(scores.shape[0]):
            partial = tail_word(self.text[i])
            chosen = []
            for k in (self.k, self.wide_k):
                cand = torch.topk(scores[i], min(k, scores.shape[1])).indices.tolist()
                chosen = [t for t in cand if self._ok(i, partial, t)]
                if chosen:
                    break
            if not chosen:
                self.fallbacks[i] += 1
                chosen = self.stop_ids[:1]
            idx = torch.tensor(chosen, device=scores.device)
            out[i, idx] = scores[i, idx]
        return out


_TEXT_CACHE: dict[int, list[str]] = {}


def token_texts(tokenizer) -> list[str]:
    key = id(tokenizer)
    if key not in _TEXT_CACHE:
        _TEXT_CACHE[key] = [tokenizer.decode([i]) for i in range(len(tokenizer))]
    return _TEXT_CACHE[key]
