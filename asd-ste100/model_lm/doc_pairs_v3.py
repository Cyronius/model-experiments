"""Training pairs for LM fine-tune round 4: data/out/lm-doc-v2 plus the Claude subagent teacher pairs on doc pool v6
(data/out/doc-teacher-claude/accepted.jsonl, data/doc_pilot/claude_teacher.py) and real pool-v6 text that already
passes the checker (data/out/doc-pool-v6/clean.jsonl) as identity rows (src = ste).

Claude pairs with a dropped-word review flag stay out (as the owner chose for flagged v5 pairs). Targets are
re-checked with the checker. Pool v6 documents (FreeBSD Handbook, Kubernetes tasks, Django, Cargo Book) are not in
dev or test. Dev and test are lm-doc-v1's, unchanged. Output: data/out/lm-doc-v3/{train,dev,test}.jsonl, summary.json.
Run with data/out/.venv/bin/python.
"""
import json, shutil, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'checker')]
from checker import Lexicon, check, errors  # noqa: E402

OUT = ROOT / 'data/out'
SRC, DEST = OUT / 'lm-doc-v2', OUT / 'lm-doc-v3'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    lex = Lexicon()
    train = [json.loads(l) for l in (SRC / 'train.jsonl').open()]
    have = {r['id'] for r in train}
    held = {json.loads(l)['group'] for s in ('dev', 'test') for l in (SRC / f'{s}.jsonl').open()}
    why = Counter()

    def add(r, ste, kind):
        if r['family'] == 'rust_book':
            raise ValueError('the Rust Book is the untouched holdout')
        if '\n' in r['text'] or '\n' in ste:
            why[kind + ': line break'] += 1
            return
        if errors(check(ste, lex, r['mode'])):
            why[kind + ': target not checker-clean'] += 1
            return
        rid, group = 'doc-v6-' + r['id'], 'doc-v6:' + r['family'] + ':' + r['doc_path']
        if rid in have:
            why[kind + ': duplicate'] += 1
            return
        assert group not in held
        have.add(rid)
        train.append(dict(id=rid, group=group, mode=r['mode'], src=r['text'] if kind == 'claude' else ste, ste=ste,
                          added=kind))
        why[kind + ': added'] += 1

    for r in map(json.loads, (OUT / 'doc-teacher-claude/accepted.jsonl').open()):
        if r['review_flags']['dropped_words']:
            why['claude: dropped-word flag'] += 1
            continue
        add(r, r['ste'], 'claude')
    for r in map(json.loads, (OUT / 'doc-pool-v6/clean.jsonl').open()):
        add(r, r['text'], 'clean')
    (DEST / 'train.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in train))
    for s in ('dev', 'test'):
        shutil.copy(SRC / f'{s}.jsonl', DEST / f'{s}.jsonl')
    summary = dict(train=len(train), from_v2=sum(1 for x in train if not x['id'].startswith('doc-v6-')),
                   edit_rows=sum(x['src'] != x['ste'] for x in train), counts=dict(why))
    (DEST / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == '__main__':
    main()
