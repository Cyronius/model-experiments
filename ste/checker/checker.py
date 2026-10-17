"""Deterministic STE checker, tier A rules only.

Implemented: R1.1/R1.2 (word not approved), R4.2 (contractions), R5.1 and R6.3 (sentence
length), R6.6 (paragraph length), R8.1 (semicolon), R3.4 (perfect tense), R3.6 (passive,
PROC only). Not implemented: noun clusters (R2.1), one instruction per sentence (R5.2),
articles (R4.5), and everything in tier B or C. Meaning is never checked here.

Severity: a word is an error only when it is a known rejected word (R1.1) or a form the
dictionary does not give (R3.1). A word the dictionary does not list at all is "info": it may be
a technical noun, which R1.5 allows, and this checker cannot tell nouns from other words.

Protected spans (backticks, URLs, paths, identifiers, anything with a digit, listed
technical words) count as one allowed word.
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass
from pathlib import Path

LEX_DIR = Path(__file__).parent.parent / "lexicon"
BE = {"is", "are", "was", "were", "be", "been", "being", "am"}
BAD_BE = {"been", "being", "am"}
DET = {"the", "a", "an", "this", "that", "these", "those", "its", "their", "your", "each", "every", "any", "no", "of", "from", "with", "for", "by", "to"}
CONTRACTION_RE = re.compile(r"n't$|'re$|'ll$|'ve$|'d$|'m$")
HAVE = {"has", "have", "had"}
PROTECT_RE = re.compile(r"`[^`]*`|https?://\S+|\S*[/\\]\S*|\b\w*[_]\w*\b|\b[a-z]+[A-Z]\w*\b|\b\w*\d\w*\b")
WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*|\d+(?:\.\d+)?|\x00P\x00")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\x00`\"(])|\n+")


@dataclass
class Violation:
    rule: str
    sentence: int
    text: str
    severity: str = "error"
    suggestion: str = ""


def plural_forms(w: str) -> set[str]:
    out = {w + "s"}
    if w.endswith(("s", "x", "z", "ch", "sh")): out.add(w + "es")
    if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou": out.add(w[:-1] + "ies")
    return out


class Lexicon:
    def __init__(self, d: Path = LEX_DIR):
        rows = [json.loads(l) for l in (d / "lexicon.jsonl").open(encoding="utf-8")]
        self.allowed: set[str] = set()
        self.rejected: dict[str, list[str]] = {}
        self.rejected_pos: dict[str, str] = {}
        self.participles: set[str] = set()
        for r in rows:
            if r["status"] == "approved":
                self.allowed.update(r["forms"])
                if r["pos"] == "n":
                    for f in r["forms"]: self.allowed.update(plural_forms(f))
                if r["pos"] == "v" and len(r["forms"]) >= 4:
                    self.participles.add(r["forms"][-1])
                if r["pos"] == "v":
                    self.participles.update(f for f in r["forms"] if f.endswith(("ed", "en", "wn")))
            else:
                self.rejected.setdefault(r["lemma"], []).extend(a["lemma"] for a in r["alternatives"])
                self.rejected_pos[r["lemma"]] = r["pos"]
        self.phrases: dict[str, list[str]] = {k: v for k, v in self.rejected.items() if " " in k}
        self.technical: set[str] = set()
        tp = d / "technical.jsonl"
        if tp.exists():
            for l in tp.open(encoding="utf-8"):
                self.technical.update(re.findall(r"[a-z]+", json.loads(l)["name"]))

    def suggest(self, w: str) -> str:
        for cand in (w, w.rstrip("s"), w[:-2] if w.endswith("ed") else w, w[:-3] if w.endswith("ing") else w):
            if cand in self.rejected: return " / ".join(sorted(set(self.rejected[cand]))[:3])
        return ""


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(text.strip()) if s.strip()]


def check(text: str, lex: Lexicon, mode: str = "PROC", allow_technical: bool = True,
          extra_allowed: set[str] | None = None) -> list[Violation]:
    extra = extra_allowed or set()
    vs: list[Violation] = []
    sents = split_sentences(text)
    if mode == "DESC" and len(sents) > 6:
        vs.append(Violation("R6.6", 0, f"{len(sents)} sentences in the paragraph"))
    limit, lrule = (20, "R5.1") if mode == "PROC" else (25, "R6.3")
    for i, sent in enumerate(sents):
        masked = PROTECT_RE.sub("\x00P\x00", sent)
        toks = WORD_RE.findall(masked)
        if len(toks) > limit:
            vs.append(Violation(lrule, i, f"{len(toks)} words", suggestion=f"maximum {limit}"))
        if ";" in sent:
            vs.append(Violation("R8.1", i, ";", suggestion="use a full stop"))
        low = [t.lower() for t in toks]
        flat = " " + " ".join(low) + " "
        for ph, alts in lex.phrases.items():
            if f" {ph} " in flat:
                vs.append(Violation("R1.1", i, ph, suggestion=" / ".join(sorted(set(alts))[:3])))
        for j, t in enumerate(low):
            if t == "\x00p\x00" or t.replace(".", "").isdigit(): continue
            if CONTRACTION_RE.search(t):
                vs.append(Violation("R4.2", i, t, suggestion="write the full words")); continue
            if t in BAD_BE:
                vs.append(Violation("R3.1", i, t, suggestion="use is, are, was, or were")); continue
            if t in lex.allowed or t in extra or (allow_technical and t in lex.technical): continue
            sug = lex.suggest(t)
            # A rejected verb right after a determiner is probably a technical noun (R1.5).
            noun_like = j > 0 and low[j - 1] in DET and lex.rejected_pos.get(t.rstrip("s")) == "v"
            vs.append(Violation("R1.1", i, t, severity="error" if sug and not noun_like else "info", suggestion=sug))
        for j in range(len(low) - 1):
            nxt = low[j + 1]
            if low[j] in HAVE and (nxt in lex.participles or nxt == "been"):
                vs.append(Violation("R3.4", i, f"{low[j]} {nxt}", suggestion="use simple past or present"))
            if mode == "PROC" and low[j] in BE and nxt in lex.participles and nxt.endswith(("ed", "en", "wn")):
                vs.append(Violation("R3.6", i, f"{low[j]} {nxt}", severity="warn", suggestion="use the active voice"))
    return vs


def errors(vs: list[Violation]) -> list[Violation]:
    return [v for v in vs if v.severity == "error"]


def describe(vs: list[Violation]) -> str:
    return "; ".join(f"{v.rule} '{v.text}'" + (f" -> {v.suggestion}" if v.suggestion else "") for v in vs)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("text"); ap.add_argument("--mode", default="PROC")
    a = ap.parse_args()
    out = check(a.text, Lexicon(), a.mode)
    print("clean" if not out else describe(out))
    print("errors:", len(errors(out)))
