"""Build the STE lexicon from the two PyPI packages (AGPL, so outputs stay gitignored).

  python build_lexicon.py --fetch          # download the wheels, unpack, convert
  python build_lexicon.py --vocab X.jsonl --tech Y.jsonl --rules Z.json   # convert only

Writes next to this file:
  lexicon.jsonl    one row per dictionary entry (lemma, pos, status, forms, alternatives)
  pairs.jsonl      aligned non-STE / STE example sentences from rejected words
  technical.jsonl  the package's technical-name and technical-verb list
  rules.json       rule id, name, summary
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys, zipfile
from pathlib import Path

HERE = Path(__file__).parent


def fetch(dest: Path) -> tuple[Path, Path, Path]:
    dest.mkdir(parents=True, exist_ok=True)
    for pkg in ("biz-dfch-asdste100vocab", "biz-dfch-asdste100rules"):
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "-q", pkg, "-d", str(dest)], check=True)
    for whl in dest.glob("*.whl"):
        with zipfile.ZipFile(whl) as z:
            z.extractall(dest / whl.stem)
    vocab = next(dest.glob("*asdste100vocab*/biz/dfch/asdste100vocab/data/asdste100_issue9_base.jsonl"))
    tech = next(dest.glob("*asdste100vocab*/biz/dfch/asdste100vocab/data/asdste100_issue9_technical_words.jsonl"))
    rules = next(dest.glob("*asdste100rules*/biz/dfch/asdste100rules/data/asdste100_issue9_rules.json"))
    return vocab, tech, rules


def clean(w: str) -> str:
    return re.sub(r"[()]", "", w).strip().lower()


def read_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def convert(vocab: Path, tech: Path, rules: Path, out: Path) -> dict:
    rows = read_jsonl(vocab)
    lex, pairs, review = [], [], []
    for r in rows:
        lemma = clean(r["name"])
        forms = sorted({lemma, *(clean(s) for s in r.get("spellings") or [])})
        alts = [{"lemma": clean(a["name"]), "pos": a.get("type_")} for a in r.get("alternatives") or []]
        row = {"lemma": lemma, "pos": r.get("type_"), "status": r["status"], "forms": forms,
               "alternatives": alts, "source": r.get("source")}
        if r["status"] == "rejected" and not alts:
            review.append(lemma)
        if r["status"] == "approved" and not r.get("type_"):
            review.append(lemma)
        lex.append(row)
        for a in r.get("alternatives") or []:
            ste, non = a.get("ste_example") or [], a.get("nonste_example") or []
            for i in range(min(len(ste), len(non))):
                pairs.append({"non_ste": non[i], "ste": ste[i].capitalize() if ste[i].isupper() else ste[i],
                              "rejected": lemma, "approved": clean(a["name"])})
    with (out / "lexicon.jsonl").open("w", encoding="utf-8") as f:
        for row in lex: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for row in pairs: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "technical.jsonl").open("w", encoding="utf-8") as f:
        for r in read_jsonl(tech):
            f.write(json.dumps({"name": clean(r["name"]), "category": r.get("category")}, ensure_ascii=False) + "\n")
    rj = json.loads(rules.read_text(encoding="utf-8"))
    (out / "rules.json").write_text(json.dumps(
        [{"id": x["id_"], "name": x["name"], "summary": x["summary"], "section": x["section"]} for x in rj],
        ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "review.txt").write_text("\n".join(sorted(set(review))) + "\n", encoding="utf-8")
    return {"rows": len(lex), "approved": sum(r["status"] == "approved" for r in lex),
            "rejected": sum(r["status"] == "rejected" for r in lex), "pairs": len(pairs),
            "review": len(set(review)), "rules": len(rj)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--vocab"); ap.add_argument("--tech"); ap.add_argument("--rules")
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    if a.fetch:
        v, t, r = fetch(HERE / "_pypi")
    else:
        v, t, r = Path(a.vocab), Path(a.tech), Path(a.rules)
    print(convert(v, t, r, Path(a.out)))
