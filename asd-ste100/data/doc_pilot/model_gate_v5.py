"""The student model's contract as a gate: can the model's tokenizer encode the pair?

The model writes only its closed output vocabulary (approved STE words, function words,
punctuation) and copies every protected or unknown input span through a slot, as many
times as the input has it. `ModelGate.problems` returns ([], "") when the pair encodes,
or the issues and a feedback note for the rewriter that names each word to change. It
reads all the problems of a rewrite, not only the first one the encoder raises. A
dropped plain word (model_dropped_word) gets no feedback: by the handoff such words are
kept as they are, and those pairs are left out. Needs
the `tokenizers` package (tokenizer/requirements.txt).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tokenizer")]
import re  # noqa: E402
from ste_tokenizer import EncodingError, SteTokenizer, article_a, case_id, model_tokens  # noqa: E402

TOKENIZER = ROOT / "data/out/fixes-v2/vocabulary"   # v2/v4's vocabulary plus the October 30 words
# Problems of the input that no rewrite can fix.
SOURCE_SIDE = ("empty_source", "slot_overflow", "source_bucket_overflow")


def suggest(lex, word: str) -> str:
    """The dictionary's alternatives for a word form. Wider than Lexicon.suggest, which
    keeps its narrow matching for the checker's calibration: this also finds stored ->
    store, modified -> modify, creating -> create and scanned -> scan."""
    found = lex.suggest(word)
    if found:
        return found
    cands = []
    if word.endswith("ed"):
        cands += [word[:-1], word[:-3] + "y" if word.endswith("ied") else word[:-3] if word[-3:-2] == word[-4:-3] else ""]
    if word.endswith("ing"):
        cands += [word[:-3] + "e", word[:-4] if word[-4:-3] == word[-5:-4] else ""]
    for cand in cands:
        if cand in lex.rejected:
            return " / ".join(sorted(set(lex.rejected[cand]))[:3])
    return ""


class ModelGate:
    def __init__(self, lex, directory: Path = TOKENIZER):
        self.lex, self.tok = lex, SteTokenizer(Path(directory))

    def problems(self, source: str, target: str, mode: str) -> tuple[list[str], str]:
        try:
            self.tok.encode_pair({"src": source, "ste": target, "mode": mode}, slot_probability=1.0)
            return [], ""
        except EncodingError as exc:
            raised = str(exc)
        if raised.startswith(SOURCE_SIDE):
            return [f"model_input:{raised}"], ""
        slots = {s["surface"]: s["count"] for s in self.tok.encode_source(source, mode)["slots"]}
        used, outside, unseen, recased = Counter(), [], [], []
        lower = {s.lower(): s for s in slots}
        tokens = model_tokens(target)
        for i, token in enumerate(tokens):
            if article_a(tokens, i):
                continue
            if token.text in slots:
                used[token.text] += 1
            elif self.tok.slot_kind(token) and token.text.lower() in lower:
                recased.append((token.text, lower[token.text.lower()]))
            elif self.tok.slot_kind(token):
                unseen.append(token.text)
            elif token.text.lower() not in self.tok.output_ids:
                outside.append(token.text.lower())
        # A plain word that the dictionary does not list is copied, and the handoff keeps such
        # words as they are: a rewrite that drops one cannot be learned, and is not repaired.
        plain = lambda s: re.fullmatch(r"[A-Za-z]+(?:'s)?", s) and case_id(s.split("'")[0]) in (0, 1)
        dropped = [s for s, n in slots.items() if used[s] < n and plain(s)]
        counts = [(s, n, used[s]) for s, n in slots.items() if used[s] != n and s not in dropped]
        issues, lines = [], []
        if outside:
            words = list(dict.fromkeys(outside))
            issues.append("model_vocabulary:" + ",".join(words))
            named = [f"{w} (use: {suggest(self.lex, w)})" if suggest(self.lex, w) else w for w in words]
            lines.append("These words are not in the STE word list: " + ", ".join(named)
                         + ". Use approved words for them.")
            if any(w.endswith("ing") for w in words):
                lines.append("STE does not use -ing verb forms. Do not put another -ing word in their place: "
                             "write the clause with a plain verb (for example 'to get', 'when it gets', "
                             "'it gets').")
        if unseen:
            words = list(dict.fromkeys(unseen))
            issues.append("model_unseen:" + ",".join(words))
            lines.append("These words are not in the input and not in the STE word list: " + ", ".join(words)
                         + ". Do not add them. Use approved words, or the input's own words.")
        if recased:
            issues.append("model_case:" + ",".join(w for w, _ in recased))
            lines.append("Write these input words with the same capital letters as the input: "
                         + ", ".join(f"{w} (input: {s})" for w, s in recased)
                         + ". Do not start a sentence with them if that changes the capital letters.")
        if counts:
            issues.append("model_copy_count:" + ",".join(s for s, _, _ in counts))
            lines.append("Keep each of these input words or literals unchanged, and use it as many times as the "
                         "input does (not more: use a pronoun or join the sentences): "
                         + ", ".join(f"{s} (input {n}, rewrite {m})" for s, n, m in counts) + ".")
        if dropped:
            issues.append("model_dropped_word:" + ",".join(dropped))
        if not issues:
            issues = [f"model_{raised}"]
            lines.append("Make the rewrite shorter." if "bucket" in raised else
                         "Keep the input's literals and words exactly as they are written.")
        return issues, "\n".join("- " + line for line in lines)
