"""Bounded, resumable pinned-Parquet positive-pair preparation; no paid APIs."""
import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('RAYON_NUM_THREADS','2')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def make_lock():
    audit=json.loads((ROOT/'broad-data/source-audit.json').read_text())
    definitions=[
      dict(id='gooaq',repo='sentence-transformers/gooaq',config='pair',query='question',positive='answer',target=1200000,max_rows=3012496),
      dict(id='paq',repo='sentence-transformers/paq',config='pair',query='query',positive='answer',target=2000000,max_rows=8000000),
      dict(id='yahoo',repo='sentence-transformers/yahoo-answers',config='title-answer-pair',query='title',positive='answer',target=650000,max_rows=1198260),
      dict(id='hotpot',repo='sentence-transformers/hotpotqa',config='triplet-20',query='anchor',positive='positive',target=169032,max_rows=169032),
      dict(id='trivia',repo='sentence-transformers/trivia-qa',config='pair',query='query',positive='answer',target=73346,max_rows=73346),
      dict(id='squad',repo='rajpurkar/squad',config='plain_text',kind='squad',target=87599,max_rows=87599),
      dict(id='stack',repo='HuggingFaceH4/stack-exchange-preferences',kind='stack',target=800000,max_rows=8000000,domain_cap=40000),
      dict(id='horizon',repo='IFM/SFT-Reasoning',config='instruction-following',kind='horizon',target=25000,max_rows=500000),
    ]
    for s in definitions:
        a=audit[s['repo']];s['revision']=a['revision'];s['declared_license']=a['license']
        files=a.get('parquet_files',a.get('files',[]))
        if s['id']=='stack':
            s['files']=[p for p in files if p.endswith('.parquet') and p.startswith('data/') and '.meta.' not in p and '/meta.' not in p and '/android.' not in p and '/askubuntu.' not in p and '.meta/' not in p]
        else:
            s['files']=[p for p in files if p.startswith(s['config']+'/') and '/train-' in p and p.endswith('.parquet')]
            if s['id']=='horizon':s['files']=[p for p in files if p.startswith('instruction-following/') and p.endswith('.parquet')]
        assert s['files'],s
    lock=dict(seed=20261025,token_cap=128,teacher_targets=False,negative_policy='No mined or lower-vote answers labeled irrelevant',
              sampling='Deterministically shuffled files and row groups; round robin across every selected shard, bounded per source. Not a uniform sample of every underlying row.',
              maximum_selected_compressed_bytes=16*1024**3,sources=definitions,
              licenses='Record source terms and attribution; no claim that unlicensed derivative cards grant additional rights. Noncommercial flax Stack Exchange export excluded.')
    path=ROOT/'broad-data/sources.lock.json'
    if path.exists():assert json.loads(path.read_text())==lock
    else:atomic(path,lock)
    return lock


def domain_for(spec,path):
    return path.split('/')[1] if spec.get('kind')=='stack' else spec['id']


def fetch_source(spec,lock,out,resume_under_target=False,part_limit=None):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from transformers import AutoTokenizer
    from broad_pairs import candidates,prepare_batch
    destination=out/spec['id'];destination.mkdir(parents=True,exist_ok=True)
    request=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    final=destination/'manifest.json'
    if final.exists():
        previous=json.loads(final.read_text());assert previous['request_sha256']==request
        if previous.get('complete') and not (resume_under_target and not previous.get('target_met',False)):
            print('Verified completed source',spec['id'],previous['accepted'],flush=True);return previous
    tokenizer=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
    fs=HfFileSystem(token=False)
    paths=list(spec['files']);random.Random(str(lock['seed'])+spec['id']).shuffle(paths)
    units={p.stem:json.loads(p.read_text()) for p in destination.glob('*.meta.json')}
    stats=collections.Counter();domain_counts=collections.Counter();accepted=scanned=compressed=0;parts=[]
    for p in sorted(destination.glob('*.meta.json')):
        m=json.loads(p.read_text());assert m['request_sha256']==request
        artifact=destination/m['part'];assert digest(artifact)==m['sha256']
        parts.append(m);accepted+=m['accepted'];scanned+=m['rows'];compressed+=m['selected_compressed_bytes'];stats.update(m['rejections']);domain_counts[domain_for(spec,m['file'])]+=m['accepted']
    layout_path=destination/'layout.json'
    if layout_path.exists():
        layout=json.loads(layout_path.read_text());assert layout['request_sha256']==request
    else:
        files=[]
        for path in paths:
            url='datasets/'+spec['repo']+'@'+spec['revision']+'/'+path
            with fs.open(url,'rb',block_size=1024*1024) as f:
                pf=pq.ParquetFile(f)
                groups=list(range(pf.num_row_groups));random.Random(str(lock['seed'])+path).shuffle(groups)
                files.append(dict(path=path,groups=groups,rows=pf.metadata.num_rows))
        layout=dict(request_sha256=request,files=files);atomic(layout_path,layout)
    complete_units={m['unit'] for m in parts}
    opened=collections.OrderedDict()
    def parquet_file(path):
        if path in opened:
            opened.move_to_end(path);return opened[path][1]
        url='datasets/'+spec['repo']+'@'+spec['revision']+'/'+path
        f=fs.open(url,'rb',block_size=1024*1024);pf=pq.ParquetFile(f);opened[path]=(f,pf)
        if len(opened)>256:
            _,(old,_)=opened.popitem(last=False);old.close()
        return pf
    columns={'stack':['qid','question','answers','metadata','date'],'horizon':['text'],'squad':['id','title','context','question','answers']}.get(spec.get('kind'),[spec.get('query'),spec.get('positive')])
    for round_index in range(max(len(f['groups']) for f in layout['files'])):
        for info in layout['files']:
            if accepted>=spec['target'] or scanned>=spec['max_rows'] or (part_limit and len(parts)>=part_limit):
                break
            if round_index>=len(info['groups']):continue
            path=info['path'];group=info['groups'][round_index];unit=hashlib.sha256((path+':'+str(group)).encode()).hexdigest()[:24]
            if unit in complete_units:continue
            domain=domain_for(spec,path)
            if spec.get('domain_cap') and domain_counts[domain]>=spec['domain_cap']:continue
            pf=parquet_file(path);meta=pf.metadata.row_group(group)
            selected_bytes=sum(meta.column(i).total_compressed_size for i in range(meta.num_columns) if meta.column(i).path_in_schema.split('.')[0] in columns)
            if selected_bytes>256*1024**2 or meta.num_rows>100000:raise RuntimeError('Oversize row group')
            if compressed+selected_bytes>lock['maximum_selected_compressed_bytes']:raise RuntimeError('Source compressed-byte bound exceeded')
            rows=pf.read_row_group(group,columns=columns).to_pylist()
            all_items=[];parse_rejected=0
            for offset,row in enumerate(rows):
                origin=dict(repo=spec['repo'],revision=spec['revision'],file=path,row_group=group,row_in_group=offset)
                items=candidates(row,spec,origin)
                parse_rejected+=not bool(items);all_items.extend(items)
            prepared=[];rejected=collections.Counter(parser_or_quality_gate=int(parse_rejected))
            for i in range(0,len(all_items),256):
                aa,rr=prepare_batch(all_items[i:i+256],tokenizer,lock['token_cap']);prepared.extend(aa);rejected.update(rr)
            remaining=min(spec['target']-accepted,spec.get('domain_cap',spec['target'])-domain_counts[domain])
            prepared=prepared[:remaining]
            part=destination/(unit+'.jsonl');temporary=part.with_suffix('.partial')
            with temporary.open('w') as f:
                for item in prepared:
                    item.update(source=spec['repo'],relation='synthetic question answer' if spec.get('kind')=='horizon' else 'question answer relevance',source_group=item.pop('group'))
                    f.write(json.dumps(item,ensure_ascii=False)+'\n')
            temporary.replace(part)
            m=dict(unit=unit,request_sha256=request,part=part.name,sha256=digest(part),rows=len(rows),accepted=len(prepared),selected_compressed_bytes=selected_bytes,rejections=dict(rejected),domains={domain:len(prepared)},file=path,row_group=group)
            atomic(part.with_suffix('.meta.json'),m);parts.append(m);complete_units.add(unit)
            scanned+=len(rows);accepted+=len(prepared);compressed+=selected_bytes;stats.update(rejected);domain_counts[domain]+=len(prepared)
            if len(parts)%10==0 or accepted>=spec['target']:
                print(json.dumps(dict(source=spec['id'],parts=len(parts),scanned=scanned,accepted=accepted,target=spec['target'],selected_compressed_mb=round(compressed/1e6))),flush=True)
            atomic(final,dict(request_sha256=request,source=spec,complete=False,accepted=accepted,scanned=scanned,selected_compressed_bytes=compressed,rejections=dict(stats),domains=dict(domain_counts)))
        if accepted>=spec['target'] or scanned>=spec['max_rows'] or (part_limit and len(parts)>=part_limit):break
    result=dict(request_sha256=request,source=spec,complete=True,accepted=accepted,scanned=scanned,selected_compressed_bytes=compressed,rejections=dict(stats),domains=dict(domain_counts),parts=len(parts),target_met=accepted>=spec['target'],execution_part_limit=part_limit,status='Candidates only; global benchmark exclusions, deduplication and group split still required')
    for f,_ in opened.values():f.close()
    atomic(final,result);print('SOURCE DONE',spec['id'],accepted,'of',spec['target'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--only',nargs='*');p.add_argument('--workers',type=int,default=2);p.add_argument('--lock-only',action='store_true');p.add_argument('--structured-horizon',action='store_true');p.add_argument('--extra-lock',type=Path);p.add_argument('--resume-under-target',action='store_true');p.add_argument('--part-limit',type=int);args=p.parse_args()
    lock=make_lock()
    if args.lock_only:return
    out=ROOT/'work/broad-data/candidates';out.mkdir(parents=True,exist_ok=True)
    lookup={s['id']:s for s in lock['sources']}
    specs=[lookup[name] for name in args.only] if args.only else list(lookup.values())
    if args.structured_horizon:
        original=lookup['horizon']
        audit=json.loads((ROOT/'broad-data/source-audit.json').read_text())[original['repo']]
        spec=dict(original,id='horizon-structured',config='3efforts-pretrain',target=10000,max_rows=500000,
                  files=[f for f in audit['parquet_files'] if f.startswith('3efforts-pretrain/') and f.endswith('.parquet')])
        extra=ROOT/'broad-data/horizon-structured.lock.json'
        if extra.exists():assert json.loads(extra.read_text())==spec
        else:atomic(extra,spec)
        specs=[spec]
    if args.extra_lock:
        value=json.loads(args.extra_lock.read_text());specs=value if isinstance(value,list) else [value]
    # Preload lazy extension modules before starting independent file workers.
    import pyarrow.parquet
    from transformers import AutoTokenizer
    from huggingface_hub import HfFileSystem
    import broad_pairs
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(fetch_source,s,lock,out,args.resume_under_target,args.part_limit):s['id'] for s in specs}
        results=[];errors={}
        for future in concurrent.futures.as_completed(futures):
            try:results.append(future.result())
            except Exception as error:
                import traceback
                errors[futures[future]]=repr(error);traceback.print_exc()
                print('SOURCE FAILED',futures[future],repr(error),flush=True)
    summary=('horizon-structured-summary' if args.structured_horizon else args.extra_lock.stem+'-summary' if args.extra_lock else 'fetch-summary')
    atomic(ROOT/'broad-data'/(summary+'.json'),dict(sources=results,teacher_targets=False,paid_services_used=False))
    if errors:raise RuntimeError(errors)


if __name__=='__main__':main()
