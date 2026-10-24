"""Bounded, resumable four-core extraction/filtering/noising for the local corpus."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sqlite3
import time

from build_local_million import SAFE_OPERATIONS, ROOT, initialize_stage, target_quality
from build_pairs import PREAMBLE, digest
from checker import Lexicon
from harvest_sources import IMPERATIVE, chunks, sha256, texts
from noise_rules import Noiser
from resume_local_stage import starting_offsets
from text_utils import normalized, protected_values, tokenize

_LEX = _NOISER = _RESERVED = _CONFIG = None


def worker_init(reserved, variants, seed, max_unknown):
    global _LEX,_NOISER,_RESERVED,_CONFIG
    _LEX,_NOISER=Lexicon(),Noiser()
    _RESERVED=set(reserved)
    _CONFIG=(variants,seed,max_unknown)


def read_range(task):
    path=Path(task['path']); first,stop=task['first'],task['stop']
    if path.suffix=='.parquet':
        import pyarrow.parquet as pq
        parquet=pq.ParquetFile(path)
        offset=0; groups=[]; base=0
        for group in range(parquet.metadata.num_row_groups):
            count=parquet.metadata.row_group(group).num_rows
            if offset<stop and offset+count>first:
                if not groups:
                    base=offset
                groups.append(group)
            offset+=count
        index=base
        for batch in parquet.iter_batches(batch_size=128,row_groups=groups):
            for row in batch.to_pylist():
                if first<=index<stop:
                    yield index,row
                index+=1
                if index>=stop:
                    return
    else:
        with path.open('rb') as stream:
            stream.seek(task['byte_start'])
            index=first
            while index<stop:
                line=stream.readline()
                if not line:
                    break
                if line.strip():
                    yield index,json.loads(line)
                    index+=1


def process_batch(task):
    entry=task['entry']; variants,seed,max_unknown=_CONFIG
    results=[]
    for index,row in read_range(task):
        doc=entry['repo']+':'+digest(json.dumps(row,sort_keys=True,ensure_ascii=False))[:24]
        for role,text in texts(row,entry['format']):
            for chunk in chunks(text):
                if not 5<=len(chunk.split())<=60 or len(chunk)>4000 or len(tokenize(chunk))*2>256:
                    continue
                letters=[c for c in chunk if c.isalpha()]
                if not letters or sum(c.isascii() for c in letters)/len(letters)<.95:
                    continue
                mode='PROC' if role in {'user','system'} or IMPERATIVE.search(chunk) else 'DESC'
                origin=digest(normalized(chunk))[:24]
                reason='reserved' if digest(normalized(chunk)) in _RESERVED else target_quality(chunk,mode,_LEX,max_unknown)
                if reason:
                    results.append((origin,reason,[doc],None))
                    continue
                provenance={'dataset':entry['repo'],'revision':entry['revision'],'license':entry['license'],
                            'url':entry['card'],'file':task['name'],'record':index,'role':role,
                            'domain':entry['domain'],'document_id':doc}
                noise=_NOISER.variants(chunk,mode,variants,seed,SAFE_OPERATIONS)
                results.append((origin,None,[doc],(chunk,mode,entry['key'],provenance,noise)))
    return results


def tasks(source,manifest,starts,dataset_counts,max_dataset,batch_size):
    entries={e['key']:e for e in manifest['datasets']}
    per_dataset={}
    # Verify each file once before submitting it to workers.
    for key,entry in entries.items():
        def dataset_tasks(key=key,entry=entry):
            for item in manifest['files']:
                if item['dataset']!=key:
                    continue
                path=(source/item['path']).resolve()
                if sha256(path)!=item.get('sha256'):
                    raise ValueError(f'raw checksum differs: {path}')
                start=starts.get(str(path),0)
                offsets=None
                if path.suffix=='.parquet':
                    import pyarrow.parquet as pq
                    total=pq.ParquetFile(path).metadata.num_rows
                else:
                    offsets=[]
                    with path.open('rb') as stream:
                        while True:
                            position=stream.tell(); line=stream.readline()
                            if not line:
                                break
                            if line.strip():
                                offsets.append(position)
                    total=len(offsets)
                for first in range(start,total,batch_size):
                    yield {'path':str(path),'name':item['name'],'first':first,'stop':min(total,first+batch_size),
                           'byte_start':offsets[first] if offsets is not None else None,'entry':entry}
        per_dataset[key]=iter(dataset_tasks())
    schedule=[key for key in sorted(entries) for _ in range(entries[key]['export_weight'])]
    while schedule:
        for key in schedule[:]:
            if dataset_counts[key]>=max_dataset:
                schedule=[k for k in schedule if k!=key]
                continue
            try:
                yield next(per_dataset[key])
            except StopIteration:
                schedule=[k for k in schedule if k!=key]


def stage_parallel(source,output,n=1_000_000,workers=4,variants=6,seed=0,max_unknown=.2,exclude=None):
    db=initialize_stage(source,output,seed,variants,max_unknown,exclude)
    manifest=json.loads((source/'download_manifest.json').read_text(encoding='utf-8'))
    starts=starting_offsets(source,output)
    reserved={r[0] for r in db.execute('SELECT text_hash FROM reserved')}
    done={r[0] for r in db.execute('SELECT origin_id FROM processed')}
    accepted_origins={r[0] for r in db.execute('SELECT DISTINCT origin_id FROM pairs')}
    count=db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]
    initial=count; started=time.monotonic(); printed=count
    modes=Counter(dict(db.execute('SELECT mode,COUNT(*) FROM pairs GROUP BY mode')))
    datasets=Counter(dict(db.execute('SELECT dataset,COUNT(*) FROM pairs GROUP BY dataset')))
    max_mode,max_dataset=math.ceil(n*.7),math.ceil(n*.6)
    (output/'parallel_driver.json').write_text(json.dumps({'code_sha256':sha256(Path(__file__)),
        'workers':workers,'starting_records':starts,'initial_pairs':initial,'target_pairs':n},indent=2)+'\n',encoding='utf-8')
    iterator=iter(tasks(source,manifest,starts,datasets,max_dataset,256))
    with ProcessPoolExecutor(max_workers=workers,initializer=worker_init,initargs=(reserved,variants,seed,max_unknown)) as pool:
        while count<n:
            wave=[]
            for _ in range(workers*2):
                task=next(iterator,None)
                if task is None:
                    break
                wave.append(pool.submit(process_batch,task))
            if not wave:
                break
            for future in wave:
                for origin,reason,docs,payload in future.result():
                    if count>=n:
                        break
                    if origin in done:
                        if origin in accepted_origins:
                            db.executemany('INSERT OR IGNORE INTO target_documents VALUES (?,?)',[(origin,doc) for doc in docs])
                        continue
                    if payload:
                        text,mode,key,provenance,noise=payload
                        if modes[mode]>=max_mode or datasets[key]>=max_dataset:
                            continue
                    done.add(origin)
                    accepted=0
                    if payload:
                        for variant in noise:
                            noisy=variant['src']
                            if PREAMBLE.search(noisy) or digest(normalized(noisy)) in reserved or len(tokenize(noisy))*2>256:
                                continue
                            if protected_values(text)!=protected_values(noisy):
                                continue
                            source_hash,target_hash=digest(normalized(noisy)),digest(normalized(text))
                            old=db.execute('SELECT target_hash FROM source_labels WHERE mode=? AND source_hash=?',(mode,source_hash)).fetchone()
                            if old and old[0]!=target_hash:
                                continue
                            pair_id=digest('\0'.join((mode,normalized(noisy),normalized(text))))[:24]
                            cur=db.execute('INSERT OR IGNORE INTO pairs VALUES (?,?,?,?,?,?,?,?,?)',
                                (pair_id,origin,mode,noisy,text,variant['noise'],key,json.dumps(docs),json.dumps(provenance)))
                            db.execute('INSERT OR IGNORE INTO source_labels VALUES (?,?,?)',(mode,source_hash,target_hash))
                            accepted+=cur.rowcount; count+=cur.rowcount; modes[mode]+=cur.rowcount; datasets[key]+=cur.rowcount
                            if count>=n or modes[mode]>=max_mode or datasets[key]>=max_dataset:
                                break
                        reason='accepted' if accepted else 'no_safe_noise'
                    db.execute('INSERT INTO processed VALUES (?,?)',(origin,reason))
                    if accepted:
                        accepted_origins.add(origin)
                        db.executemany('INSERT OR IGNORE INTO target_documents VALUES (?,?)',[(origin,doc) for doc in docs])
                db.commit()
                if count-printed>=20000:
                    rate=(count-initial)/max(time.monotonic()-started,.01)
                    print(f'staged {count:,}/{n:,} pairs; {rate:.0f} pairs/s since parallel resume',flush=True)
                    printed=count
                if count>=n:
                    break
    summary={'requested_pairs':n,'staged_pairs':count,'modes':dict(modes),'datasets':dict(datasets),
             'target_outcomes':dict(db.execute('SELECT reason,COUNT(*) FROM processed GROUP BY reason')),
             'targets':db.execute('SELECT COUNT(DISTINCT origin_id) FROM pairs').fetchone()[0],
             'noise':dict(db.execute('SELECT noise,COUNT(*) FROM pairs GROUP BY noise')),
             'status':'ready' if count>=n else 'shortfall','teacher_calls':0,'judge_calls':0,
             'parallel_workers':workers,'latest_stage_seconds':round(time.monotonic()-started,1)}
    (output/'stage_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    db.close()
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sources',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--n',type=int,default=1_000_000)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--variants',type=int,default=6)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--max-unknown',type=float,default=.2)
    ap.add_argument('--exclude',type=Path,default=ROOT/'data/out/pairs-v1')
    args=ap.parse_args()
    if min(args.n,args.workers,args.variants)<=0 or not 0<=args.max_unknown<=1:
        ap.error('invalid build limits')
    print(json.dumps(stage_parallel(args.sources,args.out,args.n,args.workers,args.variants,args.seed,args.max_unknown,args.exclude),indent=2))
