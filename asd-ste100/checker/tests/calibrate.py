"""Run the checker over the dictionary's own example pairs (needs lexicon built).

  python calibrate.py
Reports how often the STE side is error-free and how often the non-STE side is caught.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from checker import Lexicon, check, errors

pairs = [json.loads(l) for l in (Path(__file__).parent.parent.parent / "lexicon" / "pairs.jsonl").open(encoding="utf-8")]
n = len(pairs)
print(f"pairs: {n}")
# Strict STE measures the checker itself; with tech_words.txt the catch rate also drops by design.
for label, lex in (("strict STE", Lexicon(tech_words=False)), ("with tech_words.txt", Lexicon())):
    ste_ok = non_caught = 0; misses = []
    for p in pairs:
        se = errors(check(p["ste"], lex, "PROC"))
        ne = errors(check(p["non_ste"], lex, "PROC"))
        ste_ok += not se
        non_caught += bool(ne)
        if se and len(misses) < 8: misses.append((p["ste"], [f"{v.rule}:{v.text}" for v in se]))
    print(f"[{label}]")
    print(f"  STE side with zero errors:      {ste_ok}/{n} = {ste_ok/n:.1%}")
    print(f"  non-STE side with >=1 error:    {non_caught}/{n} = {non_caught/n:.1%}")
for m in misses: print("  STE side flagged:", m)
