"""Independent meaning review of a model's dev rewrites by Claude subagents (an error rate beside the gates).

prep writes chunks of the 679 doc dev rows (source and model output, the raw text: code spans are not masked, so
damage to them shows) for a checker subagent, which uses the v5 traps meaning-check prompt and adds a verdict.
Two checkers (a, b) review every row blind to each other; a third subagent rules on every row either flags
(flags writes its input). collect merges them and writes review.jsonl and summary.json.
The reference is not shown: the question is whether the rewrite keeps the source's meaning, not whether it
matches the teacher.

    python model_lm/meaning_review.py prep RUN.jsonl OUTDIR [SIZE]
    python model_lm/meaning_review.py flags OUTDIR NN | collect OUTDIR | gates OUTDIR [FREE_RUN.jsonl]
Run with data/out/.venv/bin/python and PYTHONPATH=checker.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'data/doc_pilot'), str(ROOT / 'data'), str(ROOT / 'checker')]
from checker import Lexicon, check, errors  # noqa: E402
from gen import proper_names  # noqa: E402
from gates_v3 import dropped_words  # noqa: E402
from ste_context_v3 import replaced_words  # noqa: E402
from ste_context_v5 import judge_prompt  # noqa: E402

DEV = ROOT / 'data/out/lm-doc-v1/dev.jsonl'
VERDICTS = ('ok', 'awkward', 'not_rewritten', 'meaning_change', 'garbled', 'literal_damage')


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()] if Path(p).exists() else []


def prep(run, outdir, size='20'):
    lex, size, out = Lexicon(), int(size), Path(outdir)
    (out / 'chunks').mkdir(parents=True, exist_ok=True)
    (out / 'judge-prompt.txt').write_text(judge_prompt(lex, 'traps') + '\n')
    dev = {r['id']: r for r in load(DEV)}
    rows = [dict(id=j['id'], mode=dev[j['id']]['mode'], src=dev[j['id']]['src'], out=j['output'])
            for j in load(run) if j['input'] == 'src']
    for n in range(0, len(rows), size):
        part, blocks = rows[n:n + size], []
        for k, r in enumerate(part):
            r['key'] = f'q{k}'
            names = proper_names(r['src'])
            blocks.append(f"### id: q{k}\nMEANING CHECK\nMode: {r['mode']}\nInput: {r['src']}\nRewrite: {r['out']}\n"
                          f"Unapproved input words that the rewrite had to replace: "
                          f"{', '.join(replaced_words(r['src'], r['mode'], lex, names)) or 'none'}\n"
                          f"Input words that are not in the rewrite: "
                          f"{', '.join(dropped_words(r['src'], r['out'], lex, names)) or 'none'}")
        c = n // size
        (out / 'chunks' / f'm{c:02d}.in.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in part))
        (out / 'chunks' / f'm{c:02d}.md').write_text('\n\n'.join(blocks) + '\n')
    (out / 'meta.json').write_text(json.dumps(dict(run=str(run), rows=len(rows), size=size), indent=2) + '\n')
    print(f'{len(rows)} rows in {(len(rows) + size - 1) // size} chunks')


def latest(p):
    return {str(e['id']): e for e in load(p) if isinstance(e, dict) and 'id' in e}


BAD = ('meaning_change', 'garbled', 'literal_damage')


def judges(f):
    return [latest(f.with_name(f.name.replace('.in.jsonl', f'.judge.{x}.jsonl'))) for x in 'ab']


def flags(outdir, n):
    """Write mNN.flags.md: the rows that either judge flagged, with both judges' findings, for the verifier."""
    f = Path(outdir) / 'chunks' / f'm{int(n):02d}.in.jsonl'
    a, b = judges(f)
    blocks = []
    for r in load(f):
        js = [j.get(r['key']) for j in (a, b)]
        if any(j and j.get('verdict') in BAD for j in js):
            found = '\n'.join(f"Checker {x}: verdict {j.get('verdict')}, severity {j.get('severity')}; differences: "
                              f"{json.dumps(j.get('differences'), ensure_ascii=False)}" for x, j in zip('AB', js) if j)
            blocks.append(f"### id: {r['key']}\nMode: {r['mode']}\nInput: {r['src']}\nRewrite: {r['out']}\n{found}")
    f.with_name(f.name.replace('.in.jsonl', '.flags.md')).write_text('\n\n'.join(blocks) + '\n')
    print(f'{len(blocks)} flagged rows')


def collect(outdir):
    out, lex = Path(outdir), Lexicon()
    rows, missing = [], 0
    for f in sorted((out / 'chunks').glob('m*.in.jsonl')):
        a, b = judges(f)
        verify = latest(f.with_name(f.name.replace('.in.jsonl', '.verify.jsonl')))
        for r in load(f):
            ja, jb, v = a.get(r['key']), b.get(r['key']), verify.get(r['key'])
            if ja is None and jb is None:
                missing += 1
                continue
            flagged = [x for x, j in (('a', ja), ('b', jb)) if j and j.get('verdict') in BAD]
            if not flagged:
                final = (ja or jb).get('verdict', 'ok')
            elif v is None:
                final = 'unverified'
            else:
                final = v.get('verdict') if v.get('upheld') else (v.get('verdict_if_refuted') or 'ok')
            rows.append(dict(id=r['id'], mode=r['mode'], src=r['src'], out=r['out'], judge_a=ja, judge_b=jb, verify=v,
                             flagged_by=flagged, verdict=final, unchanged=r['out'].strip() == r['src'].strip(),
                             checker_clean=not errors(check(r['out'], lex, r['mode']))))
    (out / 'review.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    bad = [r for r in rows if r['verdict'] in BAD]
    summary = dict(reviewed=len(rows), missing=missing, verdicts=dict(Counter(r['verdict'] for r in rows)),
                   flagged_by_a=sum('a' in r['flagged_by'] for r in rows), flagged_by_b=sum('b' in r['flagged_by'] for r in rows),
                   flagged_by_both=sum(len(r['flagged_by']) == 2 for r in rows),
                   flagged=sum(bool(r['flagged_by']) for r in rows), upheld=len(bad),
                   errors=len(bad), error_rate=round(len(bad) / max(1, len(rows)), 4),
                   errors_major=sum(1 for r in bad if (r['verify'] or {}).get('severity') == 'major'),
                   usable=sum(r['verdict'] in ('ok', 'awkward') and r['checker_clean'] and not r['unchanged'] for r in rows))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def gates(outdir, free=None):
    """Cross the verdicts with the deterministic gates (verify_v4, which includes the checker): the errors left if
    every gate failure fell back to the source, and (with the free-decoding run) if every row that the word
    constraint changed fell back too."""
    from gates_v4 import verify_v4
    lex, rows = Lexicon(), load(Path(outdir) / 'review.jsonl')
    same = {j['id']: j['output'].strip() for j in load(free) if j['input'] == 'src'} if free else {}

    def count(part):
        bad = [r for r in part if r['verdict'] in BAD]
        return dict(delivered=len(part), errors=len(bad),
                    major=sum((r['verify'] or {}).get('severity') == 'major' for r in bad))
    passed = [r for r in rows if not verify_v4(r['src'], r['out'], r['mode'], lex, False, proper_names(r['src']))]
    out = dict(all=count(rows), gates_pass=count(passed))
    if free:
        out['gates_pass_and_same_as_free'] = count([r for r in passed if same.get(r['id']) == r['out'].strip()])
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    {'prep': prep, 'flags': flags, 'collect': collect, 'gates': gates}[sys.argv[1]](*sys.argv[2:])
