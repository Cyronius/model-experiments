"""Build the confirmation-task subsamples exactly as qwen-space/RESERVED.md specifies and record their
hashes there. Computes no scores. Outputs under training-v2/work/shared-qwen8b/reserved/:
  {nq,touche}-subsample-ids.txt, {nq,touche}-docs.jsonl (id, text), {nq,touche}-queries.jsonl, {nq,touche}-qrels.tsv
"""
import csv, hashlib, io, json, re, sys, zipfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROJECT, SHARED_WORK

R = SHARED_WORK / 'reserved'
SEED = 20261112


def build(zipname, folder, tag, exclude_queries=()):
    with zipfile.ZipFile(R / zipname) as z:
        qrels = {}
        with z.open(f'{folder}/qrels/test.tsv') as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding='utf-8'), delimiter='\t'):
                if row['query-id'] in exclude_queries:
                    continue
                qrels.setdefault(row['query-id'], {})[row['corpus-id']] = int(row['score'])
        judged = {c for q in qrels.values() for c in q}
        ids = []
        with z.open(f'{folder}/corpus.jsonl') as f:
            for line in io.TextIOWrapper(f, encoding='utf-8'):
                ids.append(json.loads(line)['_id'])
        remaining = sorted(set(ids) - judged)
        chosen = np.random.default_rng(SEED).choice(np.array(remaining, dtype=object), 100000, replace=False)
        selected = sorted(judged | set(chosen.tolist()))
        sel = set(selected)
        blob = ('\n'.join(selected) + '\n').encode('utf-8')
        (R / f'{tag}-subsample-ids.txt').write_bytes(blob)
        with z.open(f'{folder}/corpus.jsonl') as f, open(R / f'{tag}-docs.jsonl', 'w') as out:
            for line in io.TextIOWrapper(f, encoding='utf-8'):
                r = json.loads(line)
                if r['_id'] in sel:
                    out.write(json.dumps({'_id': r['_id'], 'text': (r.get('title', '') + ' ' + r['text']).strip()}) + '\n')
        with z.open(f'{folder}/queries.jsonl') as f, open(R / f'{tag}-queries.jsonl', 'w') as out:
            for line in io.TextIOWrapper(f, encoding='utf-8'):
                r = json.loads(line)
                if r['_id'] in qrels:
                    out.write(json.dumps({'_id': r['_id'], 'text': r['text']}) + '\n')
        with open(R / f'{tag}-qrels.tsv', 'w') as out:
            for q, docs in sorted(qrels.items()):
                for c, g in sorted(docs.items()):
                    out.write(f'{q}\t{c}\t{g}\n')
    return {'corpus': len(ids), 'judged_in_corpus': len(judged & set(ids)), 'judged_total': len(judged),
            'selected': len(selected), 'queries': len(qrels), 'ids_sha256': hashlib.sha256(blob).hexdigest()}


def main():
    nq = build('nq.zip', 'nq', 'nq')
    to = build('webis-touche2020.zip', 'webis-touche2020', 'touche', exclude_queries=('1',))
    md5 = hashlib.md5((R / 'webis-touche2020.zip').read_bytes()).hexdigest()
    res = {'nq': nq, 'touche': to, 'touche_zip_md5': md5, 'seed': SEED}
    (R / 'subsample-build.json').write_text(json.dumps(res, indent=2) + '\n')
    p = PROJECT / 'qwen-space/RESERVED.md'
    s = p.read_text()
    s = s.replace('- NQ subsample ids SHA-256: _pending_',
                  f'- NQ subsample ids SHA-256: `{nq["ids_sha256"]}` ({nq["selected"]} documents, {nq["queries"]} queries; built 2026-11-12 by Plan A)')
    s = s.replace('- Touché subsample ids SHA-256: _pending_',
                  f'- Touché subsample ids SHA-256: `{to["ids_sha256"]}` ({to["selected"]} documents, {to["queries"]} queries; built 2026-11-12 by Plan A)')
    s = s.replace('- Touché zip md5: _pending_', f'- Touché zip md5: `{md5}`')
    p.write_text(s)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
