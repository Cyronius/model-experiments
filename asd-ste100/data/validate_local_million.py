"""Audit every local-corpus pair and its split, including shared document provenance."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sqlite3

from build_local_million import SAFE_OPERATIONS, target_quality
from build_pairs import BUCKETS, ROOT, digest, read_jsonl
from checker import Lexicon
from harvest_sources import sha256
from noise_rules import Noiser
from text_utils import normalized, protected_values, tokenize


def validate(output: Path):
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    config = json.loads((output / 'build_config.json').read_text(encoding='utf-8'))
    db = sqlite3.connect(output / 'pairs.sqlite3')
    if sha256(output / 'pairs.sqlite3') != summary['staging_db_sha256']:
        raise ValueError('staging DB differs from export manifest')
    lex,noiser = Lexicon(),Noiser()
    candidate_cache = {}
    reserved = {r[0] for r in db.execute('SELECT text_hash FROM reserved')}
    gold = {digest(normalized(r[field])) for r in read_jsonl(ROOT / 'lexicon/pairs.jsonl') for field in ('ste','non_ste')}
    reserved |= gold
    seen, target_cache, doc_splits, text_splits, group_splits, origin_splits = set(), set(), {}, {}, {}, {}
    labels, splits, buckets, noise, datasets = {}, Counter(), Counter(), Counter(), Counter()
    modes, origin_counts = Counter(), Counter()
    for shard in summary['shards']:
        path = output / shard['path']
        if sha256(path) != shard['sha256'] or path.stat().st_size != shard['bytes']:
            raise ValueError(f'shard differs from manifest: {path}')
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                pair_id, split, origin = row['id'], row['split'], row['origin_id']

                def require(condition, message):
                    if not condition:
                        raise ValueError(f'{pair_id}: {message}')

                require(pair_id not in seen, 'duplicate ID')
                seen.add(pair_id)
                require(split in {'train','dev','test'}, 'unknown split')
                require(row['mode'] in {'PROC','DESC'}, 'unknown mode')
                require(path.name.startswith(split + '-'), 'shard split differs')
                require(row['target_review'] == 'checker_only' and row['meaning_review'] == 'unjudged_rule', 'false review status')
                require('score' not in row and 'changed' not in row, 'fabricated judge fields')
                require(row['kind'] == 'local_rule' and row['noise'] in SAFE_OPERATIONS, 'unexpected noise operation')
                require(normalized(row['src']) != normalized(row['ste']), 'unchanged pair')
                parsed_tokens = {field: tokenize(row[field]) for field in ('src','ste')}
                require(Counter(t.text for t in parsed_tokens['src'] if t.protected) ==
                        Counter(t.text for t in parsed_tokens['ste'] if t.protected), 'protected content changed')
                require(pair_id == digest('\0'.join((row['mode'], normalized(row['src']), normalized(row['ste']))))[:24], 'content ID differs')
                require(origin == digest(normalized(row['ste']))[:24], 'origin differs from raw target')
                require(row['provenance']['document_id'] in row['document_ids'], 'primary document missing')
                require(bool(row['provenance'].get('license')) and bool(row['provenance'].get('revision')), 'missing source attribution')
                require(origin_splits.setdefault(origin, split) == split, 'target crosses splits')
                require(group_splits.setdefault(row['group'], split) == split, 'group crosses splits')
                for doc in row['document_ids']:
                    require(doc_splits.setdefault(doc, split) == split, 'document crosses splits')
                keys = []
                for field in ('src','ste'):
                    key = digest(normalized(row[field]))
                    keys.append(key)
                    require(key not in reserved, 'dictionary/legacy holdout overlap')
                    require(text_splits.setdefault(key, split) == split, 'text crosses splits')
                require(labels.setdefault((row['mode'],keys[0]),keys[1]) == keys[1], 'conflicting source labels')
                if origin not in target_cache:
                    require(target_quality(row['ste'],row['mode'],lex,config['max_unknown']) is None, 'target fails quality gates')
                    candidate_cache[origin] = {digest(op+'\0'+src) for op,src in noiser.candidates(row['ste'],row['mode']) if op in SAFE_OPERATIONS}
                    target_cache.add(origin)
                require(digest(row['noise']+'\0'+row['src']) in candidate_cache[origin], 'source is not the recorded rule corruption')
                for text_field,prefix in (('src','src'),('ste','tgt')):
                    tokens = parsed_tokens[text_field]
                    require(row[prefix+'_tokens'] == [t.text for t in tokens], 'surface tokens differ')
                    require(row[prefix+'_offsets'] == [[t.start,t.end] for t in tokens], 'surface offsets differ')
                    require(row['protected_'+prefix] == [i for i,t in enumerate(tokens) if t.protected], 'protection indices differ')
                source,target = row['src_tokens'],row['tgt_tokens']
                size = row['bucket']
                require(size in BUCKETS, 'unknown bucket')
                require(len(row['source_canvas']) == len(row['target_canvas']) == size, 'canvas length differs')
                require([t for t in row['source_canvas'] if t is not None] == source, 'source canvas cannot reconstruct')
                require(row['source_canvas'][::2][:len(source)] == source and all(t is None for t in row['source_canvas'][1::2]), 'source not interleaved')
                require([t for t in row['target_canvas'] if t is not None] == target, 'target canvas cannot reconstruct')
                positions = row['target_positions']
                require(len(positions) == len(target) and positions == sorted(set(positions)), 'target positions invalid')
                require(all(row['target_canvas'][p] == t for p,t in zip(positions,target)), 'target placement differs')
                i=j=0
                for op,si,tj in row['edits']:
                    require((si,tj)==(i,j), 'alignment has a gap')
                    require(op in {'keep','replace','delete','insert'}, 'unknown edit')
                    if op in {'keep','replace'}:
                        require(i<len(source) and j<len(target), 'alignment out of range')
                        require((source[i]==target[j]) == (op=='keep'), 'edit label differs')
                    i += op!='insert'; j += op!='delete'
                require((i,j)==(len(source),len(target)), 'alignment incomplete')
                splits[split]+=1; buckets[str(size)]+=1; noise[row['noise']]+=1; datasets[row['provenance']['dataset']]+=1
                modes[row['mode']]+=1; origin_counts[origin]+=1
                require(origin_counts[origin]<=config['variants'], 'too many variants for one target')
                if len(seen)%100000==0:
                    print(f'audited {len(seen):,}/{summary["pairs"]:,} pairs',flush=True)
    require(len(seen)==summary['pairs'], 'summary pair count differs')
    require(dict(splits)==summary['splits'] and dict(buckets)==summary['buckets'], 'summary distributions differ')
    require(len(seen)==db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0], 'staging count differs')
    import math
    require(max(modes.values(),default=0)<=math.ceil(summary['stage']['requested_pairs']*.7), 'mode cap exceeded')
    require(max(datasets.values(),default=0)<=math.ceil(summary['stage']['requested_pairs']*.6), 'dataset cap exceeded')
    result = {'status':'passed','pairs_checked':len(seen),'targets_checked':len(target_cache),
              'documents_checked':len(doc_splits),'splits':dict(splits),'buckets':dict(buckets),
              'noise':dict(noise),'datasets':dict(datasets),'modes':dict(modes),'teacher_calls':0,'judge_calls':0,
              'validator_sha256':sha256(Path(__file__))}
    (output/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    db.close()
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    print(json.dumps(validate(args.out),indent=2))
