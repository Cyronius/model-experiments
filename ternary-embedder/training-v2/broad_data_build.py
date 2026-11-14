"""Audit expanded positives and keep underlying groups and prior train exposures out of dev."""
import argparse
import collections
import hashlib
import heapq
import json
import os
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='true'
os.environ.setdefault('RAYON_NUM_THREADS','2')
from broad_data_fetch import atomic,digest
from broad_pairs import clean,identity,bert_prefix
from data_first_build import Groups,reference_texts,visible_key
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent


def lexical(text):
    import re
    return hashlib.blake2b(' '.join(re.findall(r'\w+',clean(text).casefold())).encode(),digest_size=16).digest()


def main():
    from transformers import AutoTokenizer
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'work/broad-data/ready');p.add_argument('--dev-fraction',type=float,default=.02);p.add_argument('--dev-queries-per-source',type=int,default=700);args=p.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'audit.json').exists():
        raise RuntimeError('Completed dataset is immutable; choose a new destination')
    tokenizer=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
    texts,reference_files=reference_texts()
    observed=ROOT/'work/expanded/observed-test-texts.jsonl'
    texts.extend(json.loads(line) for line in observed.open());reference_files[str(observed.relative_to(PROJECT))]=digest(observed)
    evaluation=json.loads((ROOT/'broad-data/evaluation.lock.json').read_text())
    for spec in evaluation['datasets'].values():
        for filename,record in spec['text_files'].items():
            path=Path(record['path']);assert digest(path)==record['sha256'];reference_files[str(path.relative_to(PROJECT))]=record['sha256']
            for line in path.open():
                r=json.loads(line);text=r.get('text','').strip();title=r.get('title','').strip()
                if text:texts.append(text)
                if title:texts.extend([title,(title+' '+text).strip()])
    import inspect,importlib.metadata
    from broad_reference_cache import load_cache,save_cache
    cache_identity=dict(format=1,token_cap=128,reference_files=reference_files,
                        tokenizer_files={p.name:digest(p) for p in (PROJECT/'tokenizer').iterdir() if p.is_file()},
                        transformations_sha256=digest(ROOT/'broad_pairs.py'),reference_loader_sha256=digest(ROOT/'data_first_build.py'),
                        lexical_sha256=hashlib.sha256(inspect.getsource(lexical).encode()).hexdigest(),
                        libraries={name:importlib.metadata.version(name) for name in ('transformers','tokenizers')})
    cache=ROOT/'work/broad-data/reference-identities.npz';cached=load_cache(cache,cache_identity)
    if cached:
        blocked,blocked_raw,blocked_lexical=cached;print('Verified cached reference identities:',len(blocked),flush=True)
    else:
        texts=list(dict.fromkeys(texts))
        blocked=set();blocked_raw={identity(t) for t in texts};blocked_lexical={lexical(t) for t in texts}
        print('Prepared full-text reference hashes:',len(texts),'unique records',flush=True)
        for i in range(0,len(texts),512):
            blocked.update(visible_key(ids) for ids in tokenizer([bert_prefix(t) for t in texts[i:i+512]],truncation=True,max_length=128)['input_ids'])
            if i%51200==0:print('Reference identities',i,'/',len(texts),flush=True)
        save_cache(cache,cache_identity,blocked,blocked_raw,blocked_lexical)
    del texts
    protected_paths=[ROOT/'work/science-mix-clean/dev.jsonl',ROOT/'work/nli-aux-global/dev.jsonl',ROOT/'work/human-triplets/dev.jsonl']
    protected_hashes={};legacy_dev=[]
    for path in protected_paths:
        if not path.exists():continue
        protected_hashes[str(path.relative_to(PROJECT))]=digest(path)
        for line in path.open():
            r=json.loads(line)
            if path==protected_paths[0]:legacy_dev.append(r)
            fields=[r[k] for k in ('query','positive','negative','sentence1','sentence2','premise','hypothesis') if isinstance(r.get(k),str)]
            for field in ('query_visible_key','positive_visible_key','negative_visible_key','visible1','visible2'):
                if r.get(field):blocked.add(bytes.fromhex(r[field]))
            for text in fields:blocked_raw.add(identity(text));blocked_lexical.add(lexical(text))
    base=ROOT/'work/science-mix-clean/train.jsonl'
    base_audit=json.loads((ROOT/'science-mix/clean-data-audit.json').read_text())
    assert digest(base)==base_audit['output_sha256']['train']
    sources=[dict(path=base,anchor=True,source='legacy')]
    candidate_root=ROOT/'work/broad-data/candidates'
    inputs={str(base.relative_to(PROJECT)):digest(base)}
    for folder in sorted(candidate_root.iterdir()):
        if not folder.is_dir():continue
        quarantine=ROOT/'broad-data/quarantine.json'
        excluded=json.loads(quarantine.read_text()) if quarantine.exists() else {}
        if folder.name in excluded:continue
        manifest=folder/'manifest.json'
        if not manifest.exists() or not json.loads(manifest.read_text()).get('complete'):
            raise RuntimeError('Source not complete: '+str(folder))
        for meta in sorted(folder.glob('*.meta.json')):
            info=json.loads(meta.read_text());part=folder/info['part'];assert digest(part)==info['sha256']
            sources.append(dict(path=part,anchor=False,source=folder.name));inputs[str(part.relative_to(PROJECT))]=info['sha256']
    local=ROOT/'work/broad-data/horizon-local/candidates.jsonl'
    if local.exists():sources.append(dict(path=local,anchor=False,source='horizon-local'));inputs[str(local.relative_to(PROJECT))]=digest(local)
    groups=Groups();seen=set();anchor_nodes=set();counts=collections.Counter();rejected=collections.Counter();source_stats=collections.defaultdict(collections.Counter)
    intermediate=out/'pairs.partial.jsonl';attribution=out/'attribution.jsonl'
    with intermediate.open('w') as output,attribution.open('w') as provenance:
        for spec in sources:
            batch=[]
            def accept(items):
                if not spec['anchor']:
                    from broad_quality import quality_pair
                    curated=[]
                    for row in items:
                        updated,changes=quality_pair(row);source=row.get('source','IFM/SFT-Reasoning')
                        if updated is None:
                            rejected[changes]+=1;source_stats[source][changes]+=1
                        else:
                            curated.append(updated)
                            if changes:
                                for change in changes:source_stats[source][change]+=1
                    items=curated
                if not items:return
                # Inputs are already curated; another HTML pass can erase literal
                # mathematical comparisons that were unescaped in the first pass.
                encoded=tokenizer([r[k] for r in items for k in ('query','positive')],truncation=False,verbose=False)['input_ids']
                for index,row in enumerate(items):
                    row=dict(row);q,p=row['query'],row['positive'];qi,pi=encoded[2*index:2*index+2]
                    source=row.get('source','IFM/SFT-Reasoning');source_stats[source]['candidate_pairs']+=1
                    if len(qi)>128 or len(pi)>128:
                        rejected['too_long']+=1;continue
                    qk,pk=visible_key(qi),visible_key(pi)
                    reason=None
                    if qk in blocked or pk in blocked or identity(q) in blocked_raw or identity(p) in blocked_raw or any(row.get(k) in blocked_raw for k in ('raw_positive_key','raw_query_key','original_positive_key')):
                        reason='benchmark_or_protected_dev_overlap'
                    elif lexical(q) in blocked_lexical or lexical(p) in blocked_lexical:
                        reason='punctuation_normalized_benchmark_or_dev_overlap'
                    elif qk==pk:reason='identical_visible_inputs'
                    elif qk+pk in seen:reason='duplicate_visible_pair'
                    if reason:rejected[reason]+=1;source_stats[source][reason]+=1;continue
                    seen.add(qk+pk);groups.union(qk,pk)
                    group=row.get('source_group',row.get('group'))
                    if group:groups.union(qk,b'G'+hashlib.blake2b((source+':'+group).encode(),digest_size=16).digest())
                    if spec['anchor']:anchor_nodes.add(qk)
                    origin=row.get('origin',{})
                    origin_id=hashlib.sha256(json.dumps(dict(source=source,origin=origin,q=qk.hex(),p=pk.hex()),sort_keys=True).encode()).hexdigest()
                    if origin:provenance.write(json.dumps(dict(id=origin_id,source=source,origin=origin,answer_evidence=row.get('answer_evidence')),ensure_ascii=False)+'\n')
                    item=dict(query=q,positive=p,source=source,relation='synthetic question answer' if source=='IFM/SFT-Reasoning' else row.get('relation','question answer relevance'),query_visible_key=qk.hex(),positive_visible_key=pk.hex(),source_group=group,origin_id=origin_id,_anchor=spec['anchor'])
                    output.write(json.dumps(item,ensure_ascii=False)+'\n');counts['kept']+=1;source_stats[source]['kept']+=1
                    source_stats[source]['query_tokens']+=len(qi);source_stats[source]['positive_tokens']+=len(pi)
                    if counts['kept']%100000==0:print('Audited eligible pairs',counts['kept'],flush=True)
            for line in spec['path'].open():
                batch.append(json.loads(line))
                if len(batch)>=256:accept(batch);batch=[]
            if batch:accept(batch)
            if len(sources)<20 or counts['kept']//100000!=counts.get('last_report',0):
                counts['last_report']=counts['kept']//100000;print('Audited',counts['kept'],'pairs; last',spec['source'],flush=True)
    del seen,blocked_raw,blocked_lexical,blocked
    prior_exposures={}
    for path in (ROOT/'work/nli-aux-global/train.jsonl',ROOT/'work/human-mix/train.jsonl'):
        prior_exposures[str(path.relative_to(PROJECT))]=digest(path)
        for line in path.open():
            row=json.loads(line)
            for field in ('visible1','visible2','query_visible_key','positive_visible_key','negative_visible_key'):
                key=row.get(field)
                if key and bytes.fromhex(key) in groups.index:anchor_nodes.add(bytes.fromhex(key))
    anchor_roots={groups.find(groups.index[node]) for node in anchor_nodes};del anchor_nodes
    node_split=bytearray(len(groups.parent));component_rows=collections.Counter();split_counts=collections.Counter();dev_choices=collections.defaultdict(dict)
    paths={s:out/(s+'.jsonl') for s in ('train','held-all')}
    anchor_path=out/'legacy-anchor.jsonl'
    with paths['train'].open('w') as train,paths['held-all'].open('w') as dev,anchor_path.open('w') as anchors:
        for line in intermediate.open():
            row=json.loads(line);qk=bytes.fromhex(row['query_visible_key']);pk=bytes.fromhex(row['positive_visible_key']);root=groups.find(groups.index[qk]);assert root==groups.find(groups.index[pk])
            split='train' if root in anchor_roots else groups.split(qk,20261025,args.dev_fraction)
            code=1 if split=='train' else 2
            for key in (qk,pk):
                index=groups.index[key];assert node_split[index] in (0,code);node_split[index]=code
            was_anchor=row.pop('_anchor');line=json.dumps(row,ensure_ascii=False)+'\n'
            (train if code==1 else dev).write(line)
            if was_anchor:
                assert code==1;anchors.write(line)
            split_counts[split]+=1;source_stats[row['source']][split]+=1;component_rows[root]+=1
            if code==2:
                key=row['query_visible_key'];score=hashlib.blake2b(b'dev-query-20261025'+qk,digest_size=8).hexdigest()
                dev_choices[row['source']][key]=score
    selected={source:set(sorted(values,key=lambda key:values[key])[:args.dev_queries_per_source]) for source,values in dev_choices.items()}
    dev_rows=0
    with (out/'dev.jsonl').open('w') as dev:
        for row in legacy_dev:
            dev.write(json.dumps(row,ensure_ascii=False)+'\n');dev_rows+=1
        for line in paths['held-all'].open():
            row=json.loads(line)
            if row['query_visible_key'] in selected[row['source']]:dev.write(line);dev_rows+=1
    audit=dict(status='Prepared and audited only; no paid allocation or training',teacher_targets=False,token_cap=128,
               train_rows=split_counts['train'],all_new_held_rows=split_counts['dev'],validation_rows=dev_rows,
               by_source={s:dict(v) for s,v in source_stats.items()},rejections=dict(rejected),
               source_parts_sha256=inputs,reference_text_files=reference_files,protected_legacy_dev_sha256=protected_hashes,
               reference_cache_sha256=digest(cache),
               original_training_components_forced_to_train=True,underlying_source_groups_kept_together=True,train_dev_visible_overlap=0,
               prior_finetuning_exposure_files_sha256=prior_exposures,
               largest_component_rows=max(component_rows.values()),components=len(component_rows),
               new_reserved_judgments_read=False,old_fiqa_scidocs_are_development=True,
               output_sha256={p.name:digest(p) for p in [paths['train'],paths['held-all'],out/'dev.jsonl',attribution,anchor_path]},
               split_seed=20261025,dev_fraction=args.dev_fraction,dev_queries_per_source=args.dev_queries_per_source,
               limits='Exact tokenizer-visible and punctuation-normalized exclusions; historical pretraining and semantic near-duplicate exposure remain unknown. Source QA and synthetic pairs are not universally human relevance judgments.')
    atomic(out/'audit.json',audit);atomic(ROOT/'broad-data/data-audit.json',audit);intermediate.unlink()
    print('BUILD COMPLETE',json.dumps({k:audit[k] for k in ('train_rows','all_new_held_rows','validation_rows','largest_component_rows')}),flush=True)


if __name__=='__main__':main()
