"""Restore apostrophes and acronym casing that the old clean_title() stripped from stored titles (2026-10-16)."""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from common.validation import SMALL_WORDS, clean_title, keeps_case, validate_title

WORD_RE = re.compile(r"[^\W\d_]+")
TITLE_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")

# skipped on purpose: its, lets, ill, id, well, were, wed, shell, hell (all real words without the apostrophe)
CONTRACTIONS = {
    "dont": "don't", "doesnt": "doesn't", "didnt": "didn't", "cant": "can't", "wont": "won't",
    "isnt": "isn't", "arent": "aren't", "wasnt": "wasn't", "werent": "weren't", "aint": "ain't",
    "havent": "haven't", "hasnt": "hasn't", "hadnt": "hadn't", "wouldnt": "wouldn't",
    "couldnt": "couldn't", "shouldnt": "shouldn't", "mustnt": "mustn't", "neednt": "needn't",
    "im": "I'm", "ive": "I've", "youre": "you're", "theyre": "they're", "youve": "you've",
    "theyve": "they've", "weve": "we've", "youll": "you'll", "theyll": "they'll", "itll": "it'll",
    "whats": "what's", "thats": "that's", "theres": "there's", "heres": "here's", "wheres": "where's",
    "whos": "who's", "hows": "how's", "hes": "he's", "shes": "she's",
    "childrens": "children's", "womens": "women's", "mens": "men's",
}


POSSESSIVE_RE = re.compile(r"([^\W\d_]+)['’]s\b")


@dataclass
class Corpus:
    dominant: dict[str, str]  # sql -> SQL, mrna -> mRNA
    plural_n: Counter  # "toys" as a plain word, corpus-wide
    possessive_n: Counter  # "toy's", keyed by "toy"


def corpus_stats(rows: list[dict]) -> Corpus:
    title_forms: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        for w in WORD_RE.findall(r["title"]):
            title_forms[w.lower()][w] += 1
    forms: dict[str, Counter] = defaultdict(Counter)
    possessive_n: Counter = Counter()
    for r in rows:
        for w in WORD_RE.findall(r["prompt"]):
            if w.lower() in title_forms:
                forms[w.lower()][w] += 1
        possessive_n.update(b.lower() for b in POSSESSIVE_RE.findall(r["prompt"]))
    dominant = {}
    for lw, c in forms.items():
        form, n = c.most_common(1)[0]
        total = sum(c.values())
        # both majorities required: prompts rule out code names (userName), titles rule out emphasis caps (CONCISE)
        if (total >= 20 and n / total >= 0.6 and any(ch.isupper() for ch in form[1:])
                and title_forms[lw][form] * 2 >= sum(title_forms[lw].values())):
            dominant[lw] = form
    plural_n = Counter({lw: sum(c.values()) for lw, c in forms.items() if lw.endswith("s")})
    return Corpus(dominant, plural_n, possessive_n)


def match_case(src: str, repl: str) -> str:
    return repl[:1].upper() + repl[1:] if src[:1].isupper() else repl


def repair(title: str, prompt: str, corpus: Corpus) -> tuple[str, list[str]]:
    words = WORD_RE.findall(prompt)
    plain = {w.lower() for w in words}
    surface = set(words)
    possessed: dict[str, set[str]] = defaultdict(set)
    for b in POSSESSIVE_RE.findall(prompt):
        possessed[b.lower()].add(b)
    rules: list[str] = []

    def fix(m: re.Match) -> str:
        w = m.group(0)
        lw = w.lower()
        if "'" in w or "’" in w:
            return w
        if lw in CONTRACTIONS:
            rules.append("contraction")
            return match_case(w, CONTRACTIONS[lw])
        base = lw[:-1]
        if lw.endswith("s") and lw != "its" and base in possessed and lw not in plain:
            # names (Natsuki's) or words more often possessive than plural; otherwise it's a plural (Toys)
            proper = any(b[:1].isupper() for b in possessed[base])
            if proper or corpus.possessive_n[base] >= corpus.plural_n[lw]:
                rules.append("possessive")
                return w[:-1] + "'s"
        # consonant+"ys" is never a correct plural (Bodys, Citys), so it was a possessive
        if re.fullmatch(r"[a-z]{3,}[b-df-hj-np-tv-z]ys", lw) and lw not in plain:
            rules.append("possessive_ys")
            return w[:-1] + "'s"
        form = corpus.dominant.get(lw)
        # already-cased words (GLOBAL_HASHMAP) and domain names (Designs.ai) keep their spelling
        if (form and form != w and form in surface and lw not in SMALL_WORDS
                and not keeps_case(w) and title[m.start() - 1:m.start()] != "."):
            rules.append("casing")
            return form
        return w

    return TITLE_WORD_RE.sub(fix, title), rules


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data")
    ap.add_argument("--outdir", default="data/v2")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    splits = {s: [json.loads(l) for l in (Path(args.indir) / f"{s}.jsonl").open(encoding="utf-8")]
              for s in args.splits}
    corpus = corpus_stats([r for rows in splits.values() for r in rows])
    print(f"{len(corpus.dominant):,} words with a dominant inner-capital spelling")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    sample: list[tuple[str, str]] = []
    with (out / "repair_log.jsonl").open("w", encoding="utf-8") as log:
        for split, rows in splits.items():
            changed = 0
            for rec in rows:
                new, rules = repair(rec["title"], rec["prompt"], corpus)
                if new == rec["title"]:
                    continue
                # keep the old title if the fix fails validation or clean_title would change it
                if validate_title(new, rec["prompt"]) is not None or clean_title(new) != new:
                    stats["rejected"] += 1
                    continue
                changed += 1
                stats.update(rules)
                log.write(json.dumps({"id": rec["id"], "split": split, "old": rec["title"], "new": new,
                                      "rules": rules}, ensure_ascii=False) + "\n")
                sample.append((rec["title"], new))
                rec["title"] = new
            with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            print(f"{split}: {changed:,} / {len(rows):,} titles changed")

    print("by rule:", dict(stats))
    random.seed(0)
    for old, new in random.sample(sample, min(25, len(sample))):
        print(f"  {old}  ->  {new}")


if __name__ == "__main__":
    main()
