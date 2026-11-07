"""Score model_lm/generate.py outputs on the 679 documentation dev pairs.

Same raw-decode numbers as the tagger probe (data/out/doc-v5b-pipeline/probes/doc_raw_eval2.py): exact,
unchanged, closer to / farther from the reference (word-level difflib on model_tokens), clean references
changed. Added: checker-clean outputs (no error-severity violation), protected values kept
(data/text_utils.protected_values equal to the source's), mean word similarity to the reference.
Run with data/out/.venv/bin/python:  score.py OUT.json RUN.jsonl [RUN.jsonl ...]
"""
import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tokenizer'), str(ROOT / 'checker'), str(ROOT / 'data')]
from ste_tokenizer import model_tokens  # noqa: E402
from checker import Lexicon, check, errors  # noqa: E402
from text_utils import protected_values  # noqa: E402

DEV = ROOT / 'data/out/lm-doc-v1/dev.jsonl'


def words(t):
    return [x.text for x in model_tokens(t)]


def sim(a, b):
    return difflib.SequenceMatcher(None, words(a), words(b), autojunk=False).ratio()


def score(path, rows, lex):
    out = {}
    for line in Path(path).open():
        j = json.loads(line)
        out[(j['id'], j['input'])] = j
    c = dict(n=0, exact=0, unchanged=0, closer=0, farther=0, checker_clean=0, protected_kept=0, not_stopped=0,
             fallbacks=0, clean_n=0, clean_changed=0, clean_checker_clean=0)
    simsum = 0.0
    for r in rows:
        j = out.get((r['id'], 'src'))
        if j:
            t = j['output']
            c['n'] += 1
            c['exact'] += t == r['ste']
            c['unchanged'] += t == r['src']
            d0, d1 = 1 - sim(r['src'], r['ste']), 1 - sim(t, r['ste'])
            c['closer'] += d1 < d0
            c['farther'] += d1 > d0
            c['checker_clean'] += not errors(check(t, lex, r['mode']))
            c['protected_kept'] += protected_values(t) == protected_values(r['src'])
            c['not_stopped'] += not j['stopped']
            c['fallbacks'] += j['fallbacks'] > 0
            simsum += 1 - d1
        k = out.get((r['id'], 'ste'))
        if k:
            c['clean_n'] += 1
            c['clean_changed'] += k['output'] != r['ste']
            c['clean_checker_clean'] += not errors(check(k['output'], lex, r['mode']))
    c['mean_similarity'] = round(simsum / max(c['n'], 1), 4)
    return c


def main():
    rows = [json.loads(l) for l in DEV.open()]
    lex = Lexicon()
    ref_clean = sum(not errors(check(r['ste'], lex, r['mode'])) for r in rows)
    src_clean = sum(not errors(check(r['src'], lex, r['mode'])) for r in rows)
    report = dict(references_checker_clean=ref_clean, sources_checker_clean=src_clean, n=len(rows), runs={})
    for p in sys.argv[2:]:
        report['runs'][Path(p).stem] = score(p, rows, lex)
        print(Path(p).stem, report['runs'][Path(p).stem], flush=True)
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
