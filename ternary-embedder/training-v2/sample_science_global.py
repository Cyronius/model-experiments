"""Bounded, stratified science sampling across all pinned Parquet shards.

Read16 randomly chosen row groups per shard, not all42M abstracts. Select up to
250k eligible unique titles globally, then split connected text components.
"""
import collections,concurrent.futures,hashlib,json,os,re
from pathlib import Path
os.environ.setdefault('RAYON_NUM_THREADS','4')
import numpy as np,pyarrow.parquet as pq
from huggingface_hub import HfApi,HfFileSystem
from transformers import AutoTokenizer
from data_first_build import Groups,reference_texts,visible_key,sha
from data_first_fetch import clean,key,valid_pair
from pair_training_state import atomic_json
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
SOURCE={'id':'sentence-transformers/s2orc','revision':'8cfc394e83b2ebfcf38f90b508aea383df742439','config':'title-abstract-pair'}
SEED=67;GROUPS_PER_FILE=16;TARGET=250000

def lexical_key(text):return hashlib.blake2b(' '.join(re.findall(r'\w+',text.casefold())).encode(),digest_size=16).hexdigest()
def priority(text):return hashlib.blake2b(b'global-science67'+text.encode(),digest_size=16).digest()

def fetch(spec):
 index,path=spec;fs=HfFileSystem();url='datasets/'+SOURCE['id']+'@'+SOURCE['revision']+'/'+path
 with fs.open(url,'rb',block_size=1024*1024) as f:
  pf=pq.ParquetFile(f);rng=np.random.default_rng(SEED+index);groups=sorted(rng.choice(pf.num_row_groups,min(GROUPS_PER_FILE,pf.num_row_groups),replace=False).tolist());rows=[]
  for group in groups:
   for offset,row in enumerate(pf.read_row_group(group,columns=['title','abstract']).to_pylist()):rows.append(dict(row,parquet_file=path,row_group=group,row_in_group=offset))
  meta={'file':path,'total_rows':pf.metadata.num_rows,'total_row_groups':pf.num_row_groups,'selected_row_groups':groups,'sample_rows':len(rows),'selected_compressed_column_bytes':sum(pf.metadata.row_group(g).column(c).total_compressed_size for g in groups for c in range(2))}
 return rows,meta

def main():
 out=ROOT/'work/science-global';out.mkdir(exist_ok=True)
 if (out/'audit.json').exists():raise RuntimeError('An audited sample already exists')
 tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True);blocked=set();lexical=set();base_hashes={}
 base=ROOT/'work/hardneg/data'
 for split in ('train','dev'):
  p=base/(split+'.jsonl');base_hashes[str(p.relative_to(PROJECT))]=sha(p)
  for line in p.open():
   r=json.loads(line);blocked.update((r['query_visible_key'],r['positive_visible_key']));lexical.add(lexical_key(r['query']))
   if r.get('negative'):blocked.add(r['negative_visible_key'])
 aux=ROOT/'work/nli-aux-global'
 for split in ('train','dev'):
  p=aux/(split+'.jsonl');base_hashes[str(p.relative_to(PROJECT))]=sha(p)
  for line in p.open():
   r=json.loads(line);blocked.update((r['visible1'],r['visible2']))
 refs,reference_files=reference_texts();extra=ROOT/'work/expanded/observed-test-texts.jsonl';refs.extend(json.loads(l) for l in extra.open());reference_files[str(extra.relative_to(PROJECT))]=sha(extra)
 for off in range(0,len(refs),512):
  batch=refs[off:off+512];blocked.update(visible_key(ids).hex() for ids in tk(batch,max_length=128,truncation=True)['input_ids']);lexical.update(lexical_key(t) for t in batch)
 del refs
 info=HfApi().dataset_info(SOURCE['id'],revision=SOURCE['revision']);files=sorted(s.rfilename for s in info.siblings if s.rfilename.startswith(SOURCE['config']+'/') and s.rfilename.endswith('.parquet'))
 if len(files)!=62:raise RuntimeError('Pinned science file layout changed')
 candidates={};stats=collections.Counter();shards=[]
 def accept(rows):
  clean_rows=[]
  for row in rows:
   q,p=clean(row['title']),clean(row['abstract']);stats['raw_sample_rows']+=1;reason=valid_pair(q,p,'scientific title abstract; weak supervision')
   if not reason and (len(q.split())<3 or len(p.split())<12):reason='insufficient_scientific_text'
   if reason:stats[reason]+=1;continue
   clean_rows.append((q,p,row))
  for off in range(0,len(clean_rows),256):
   batch=clean_rows[off:off+256];enc=tk([s for q,p,_ in batch for s in (q,p)],truncation=True,max_length=128)['input_ids']
   for i,(q,p,row) in enumerate(batch):
    qi,pi=enc[2*i:2*i+2];qk,pk=visible_key(qi).hex(),visible_key(pi).hex();reason=None
    if len(qi)>=128 or len(pi)>=128:reason='at_token_cap'
    elif qk in blocked or pk in blocked:reason='visible_existing_or_benchmark_overlap'
    elif lexical_key(q) in lexical or lexical_key(p) in lexical:reason='punctuation_normalized_overlap'
    elif qk==pk:reason='identical_visible_inputs'
    if reason:stats[reason]+=1;continue
    item={'query':q,'positive':p,'source':SOURCE['id'],'relation':'scientific title abstract; weak supervision','query_key':key(q),'positive_key':key(p),'query_visible_key':qk,'positive_visible_key':pk,'source_row':{'parquet_file':row['parquet_file'],'row_group':row['row_group'],'row_in_group':row['row_in_group']}}
    old=candidates.get(qk)
    if old is not None:stats['duplicate_visible_title']+=1
    if old is None or (pk,str(item['source_row']))<(old['positive_visible_key'],str(old['source_row'])):candidates[qk]=item
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  iterator=iter(enumerate(files));pending={pool.submit(fetch,next(iterator)) for _ in range(4)}
  while pending:
   done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
   for future in done:
    rows,meta=future.result();accept(rows);shards.append(meta);print('Sampled shards',len(shards),'/',len(files),'eligible unique titles',len(candidates),flush=True)
    try:spec=next(iterator)
    except StopIteration:continue
    pending.add(pool.submit(fetch,spec))
 selected=sorted(candidates.values(),key=lambda r:priority(r['query_visible_key']))[:TARGET]
 if len(selected)<50000:raise RuntimeError('Too few eligible science pairs to justify a new dataset')
 groups=Groups()
 for r in selected:groups.union(bytes.fromhex(r['query_visible_key']),bytes.fromhex(r['positive_visible_key']))
 counts=collections.Counter();nodes={s:set() for s in ('train','dev')};handles={s:(out/(s+'.jsonl')).open('w') for s in nodes}
 try:
  for r in selected:
   split=groups.split(bytes.fromhex(r['query_visible_key']),23,.02);handles[split].write(json.dumps(r,ensure_ascii=False)+'\n');counts[split]+=1;nodes[split].update((r['query_visible_key'],r['positive_visible_key']))
 finally:
  for f in handles.values():f.close()
 assert not(nodes['train']&nodes['dev']) and not((nodes['train']|nodes['dev'])&blocked)
 audit={'source':SOURCE,'seed':SEED,'row_groups_per_shard':GROUPS_PER_FILE,'all_shards_sampled':len(shards),'shards':sorted(shards,key=lambda r:r['file']),'raw_sample_rows':stats['raw_sample_rows'],'eligible_unique_titles':len(candidates),'selected_pairs':len(selected),'splits':dict(counts),'rejections':dict(stats),'token_cap':128,'teacher_targets':False,'benchmark_qrels_read':False,'prior_pair_and_auxiliary_data_sha256':base_hashes,'reference_text_files':reference_files,'output_sha256':{s:sha(out/(s+'.jsonl')) for s in handles},'train_dev_shared_visible_texts':0,'shared_visible_texts_with_existing_data':0,'policy':'Fixed uniform row-group samples from every pinned shard; global query-hash selection; complete inputs below128; >=3-word titles and >=12-word abstracts; lexical punctuation-normalized exclusions in addition to model-visible exclusions','limits':'Title/abstract association is weak relevance supervision. Exact and punctuation-normalized exclusions do not establish absence of all semantic near duplicates. Compressed-column bytes do not include HTTP read-ahead overhead.'}
 atomic_json(out/'audit.json',audit);(ROOT/'science-global').mkdir(exist_ok=True);atomic_json(ROOT/'science-global/data-audit.json',audit);print('DONE',dict(counts),len(selected),flush=True)
if __name__=='__main__':main()
