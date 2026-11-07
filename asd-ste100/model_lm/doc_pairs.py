"""Export the 7,794 training-ready documentation pairs as surface text, split as the tagger split them.

Same filter and split as model_tagger/data_revision_v5.py (doc_rows): second judge clean, encodes with
data/out/fixes-v2/vocabulary, split by source document (5% dev, 5% test). The dev rows must equal the
679 of data/out/id-doc-v5/. Output: data/out/lm-doc-v1/{train,dev,test}.jsonl with id, group, mode, src, ste.
Run with data/out/.venv/bin/python.
"""
import gzip, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'model_tagger'), str(ROOT / 'data'), str(ROOT / 'tokenizer')]
from tagger import SteTokenizer  # noqa: E402
from data_revision_v5 import doc_rows, VOCAB, OUT  # noqa: E402

DEST = OUT / 'lm-doc-v1'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    tok = SteTokenizer(VOCAB)
    skipped = Counter()
    rows = {'train': [], 'dev': [], 'test': []}
    for r in doc_rows(tok, skipped):
        rows[r['split']].append(dict(id=r['id'], group=r['group'], mode=r['mode'], src=r['src'], ste=r['ste']))
    for split, items in rows.items():
        (DEST / f'{split}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in items))
    for split in ('dev', 'test'):
        ref = [json.loads(l)['id'] for l in gzip.open(OUT / f'id-doc-v5/shards/{split}-00000.jsonl.gz', 'rt')]
        if ref != [x['id'] for x in rows[split]]:
            raise ValueError(f'{split} rows differ from id-doc-v5')
    groups = {s: {x['group'] for x in v} for s, v in rows.items()}
    if groups['train'] & (groups['dev'] | groups['test']) or groups['dev'] & groups['test']:
        raise ValueError('a document is in two splits')
    summary = dict(splits={k: len(v) for k, v in rows.items()}, skipped=dict(skipped))
    (DEST / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == '__main__':
    main()
