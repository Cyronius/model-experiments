"""Training pairs for LM fine-tune round 3: data/out/lm-doc-v1 plus the unflagged teacher pairs that the
tagger left out only because its vocabulary could not encode them.

Added rows: second judge clean (as doc_rows), in a train-split document (doc_split), target checker-clean,
and failing the tagger encoding only for target_outside_vocabulary, target_unseen_slot, slot_overflow_* or
source_bucket_overflow. Pairs failing slot_fidelity stay out (closest to the owner's choice to leave out
pairs that drop a word the dictionary does not list). Dev and test are lm-doc-v1's, unchanged.
Output: data/out/lm-doc-v2/{train,dev,test}.jsonl, summary.json. Run with data/out/.venv/bin/python.
"""
import json, shutil, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'model_tagger'), str(ROOT / 'data'), str(ROOT / 'tokenizer'), str(ROOT / 'checker')]
from tagger import SteTokenizer, EncodingError  # noqa: E402
from data_revision_v5 import DOC, VOCAB, OUT, doc_split  # noqa: E402
from checker import Lexicon, check, errors  # noqa: E402

SRC, DEST = OUT / 'lm-doc-v1', OUT / 'lm-doc-v2'
ADD = ('target_outside_vocabulary', 'target_unseen_slot', 'slot_overflow', 'source_bucket_overflow')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    tok, lex = SteTokenizer(VOCAB), Lexicon()
    train = [json.loads(l) for l in (SRC / 'train.jsonl').open()]
    have = {r['id'] for r in train}
    held = {json.loads(l)['group'] for s in ('dev', 'test') for l in (SRC / f'{s}.jsonl').open()}
    why = Counter()
    for line in DOC.read_text().splitlines():
        r = json.loads(line)
        if (r['review_flags'].get('second_judge') or {}).get('differences') != []:
            continue
        try:
            tok.encode_pair(dict(src=r['text'], ste=r['ste'], mode=r['mode']), .7)
            continue
        except EncodingError as e:
            reason = str(e).split(':')[0]
        group = 'doc-v5:' + r['family'] + ':' + r['doc_path']
        if doc_split(group) != 'train':
            why['not train split'] += 1
            continue
        if not reason.startswith(ADD):
            why['left out: ' + reason] += 1
            continue
        if errors(check(r['ste'], lex, r['mode'])):
            why['target not checker-clean'] += 1
            continue
        if '\n' in r['text'] or '\n' in r['ste']:
            why['line break'] += 1
            continue
        rid = 'doc-v5-' + r['id']
        assert rid not in have and group not in held
        have.add(rid)
        train.append(dict(id=rid, group=group, mode=r['mode'], src=r['text'], ste=r['ste'], added=reason))
        why['added: ' + reason] += 1
    (DEST / 'train.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in train))
    for s in ('dev', 'test'):
        shutil.copy(SRC / f'{s}.jsonl', DEST / f'{s}.jsonl')
    summary = dict(train=len(train), from_v1=len(train) - sum(v for k, v in why.items() if k.startswith('added')),
                   counts=dict(why))
    (DEST / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == '__main__':
    main()
