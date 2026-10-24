"""Resume local staging near the last retained raw record, without re-tokenizing its prefix.

Earlier records are known to have been visited because each dataset is consumed in file
order. Starting one record before its last retained pair also covers interrupted documents.
Placeholder records retain exact record numbering but yield no text or training examples.
"""
import argparse
import itertools
import json
from pathlib import Path
import sqlite3

import build_local_million as builder
from harvest_sources import records, sha256


def starting_offsets(source, output):
    database=output/'pairs.sqlite3'
    if not database.exists():
        return {}
    db=sqlite3.connect(database)
    marks={(dataset,file):int(index) for dataset,file,index in db.execute(
        "SELECT dataset,json_extract(provenance,'$.file'),MAX(CAST(json_extract(provenance,'$.record') AS INTEGER)) FROM pairs GROUP BY dataset,json_extract(provenance,'$.file')")}
    db.close()
    manifest=json.loads((source/'download_manifest.json').read_text(encoding='utf-8'))
    starts={}
    for entry in manifest['datasets']:
        files=[item for item in manifest['files'] if item['dataset']==entry['key']]
        seen=[i for i,item in enumerate(files) if (entry['key'],item['name']) in marks]
        if not seen:
            continue
        last=max(seen)
        for i,item in enumerate(files[:last+1]):
            path=(source/item['path']).resolve()
            if i==last:
                starts[str(path)]=max(0,marks[(entry['key'],item['name'])]-1)
            elif path.suffix=='.parquet':
                import pyarrow.parquet as pq
                starts[str(path)]=pq.ParquetFile(path).metadata.num_rows
            else:
                with path.open(encoding='utf-8') as stream:
                    starts[str(path)]=sum(bool(line.strip()) for line in stream)
    return starts


def records_after(path, start):
    if not start:
        yield from records(path)
        return
    yield from itertools.repeat({},start)
    if path.suffix=='.parquet':
        import pyarrow.parquet as pq
        parquet=pq.ParquetFile(path)
        offset=0; groups=[]
        for group in range(parquet.metadata.num_row_groups):
            count=parquet.metadata.row_group(group).num_rows
            if offset+count>start:
                groups=list(range(group,parquet.metadata.num_row_groups))
                break
            offset+=count
        if not groups:
            return
        for batch in parquet.iter_batches(batch_size=128,row_groups=groups):
            for row in batch.to_pylist():
                if offset>=start:
                    yield row
                offset+=1
    else:
        with path.open(encoding='utf-8') as stream:
            offset=0
            for line in stream:
                if not line.strip():
                    continue
                if offset>=start:
                    yield json.loads(line)
                offset+=1


def resume(source,output,n=1_000_000,variants=6,seed=0,max_unknown=.2,exclude=None):
    starts=starting_offsets(source,output)
    original=builder.records
    builder.records=lambda path: records_after(path,starts.get(str(path.resolve()),0))
    output.mkdir(parents=True,exist_ok=True)
    (output/'resume_driver.json').write_text(json.dumps({'code_sha256':sha256(Path(__file__)),'starting_records':starts},indent=2)+'\n',encoding='utf-8')
    print(f'Resuming from {len(starts)} saved file positions',flush=True)
    try:
        return builder.stage(source,output,n,variants,seed,max_unknown,exclude)
    finally:
        builder.records=original


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sources',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--n',type=int,default=1_000_000)
    ap.add_argument('--variants',type=int,default=6)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--max-unknown',type=float,default=.2)
    ap.add_argument('--exclude',type=Path,default=builder.ROOT/'data/out/pairs-v1')
    args=ap.parse_args()
    print(json.dumps(resume(args.sources,args.out,args.n,args.variants,args.seed,args.max_unknown,args.exclude),indent=2))
