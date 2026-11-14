"""Restore pinned MS MARCO BM25 candidates onto unchanged, audited train pairs.

Only training candidates are inspected. Benchmark relevance labels remain closed.
Heuristic filters reduce obvious false negatives; unjudged does not imply irrelevant.
"""
import collections,json,re
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from data_first_fetch import clean,key
from data_first_build import visible_key,reference_texts,sha
from pair_training_state import atomic_json
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent

def near_duplicate(a,b):
 a=set(re.findall(r'\w+',a.casefold()));b=set(re.findall(r'\w+',b.casefold()))
 return bool(a and b and len(a&b)/len(a|b)>.8)

def main():
 src=ROOT/'work/data-first/capfit';out=ROOT/'work/hardneg/data';out.mkdir(parents=True,exist_ok=True)
 if (out/'audit.json').exists():raise RuntimeError('Already built; do not overwrite an audited dataset')
 tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
 wanted={};known=collections.defaultdict(set);train_queries=set();dev_visible=set();stats=collections.Counter()
 for line in (src/'train.jsonl').open():
  r=json.loads(line);stats['train_rows']+=1;known[r['query_visible_key']].add(r['positive_visible_key']);train_queries.add(r['query_visible_key'])
  if r['source']=='sentence-transformers/msmarco-bm25':wanted[(r['query_key'],r['positive_key'])]=(r['query_visible_key'],r['positive_visible_key'])
 for line in (src/'dev.jsonl').open():
  r=json.loads(line);dev_visible.update((r['query_visible_key'],r['positive_visible_key']))
 refs,ref_files=reference_texts();blocked=set()
 for off in range(0,len(refs),512):blocked.update(visible_key(ids).hex() for ids in tk(refs[off:off+512],max_length=128,truncation=True)['input_ids'])
 del refs
 spec=json.loads((ROOT/'data-first/sources.lock.json').read_text())[0]
 stream=load_dataset(spec['id'],spec['config'],revision=spec['revision'],split='train',streaming=True)
 restored={};pending=[]
 def accept(batch):
  if not batch:return
  encoded=tk([r[1] for r in batch],truncation=True,max_length=128)['input_ids']
  for (identity,n,p),ids in zip(batch,encoded):
   qk,pk=wanted[identity];nk=visible_key(ids).hex();reason=None
   if len(ids)>=128:reason='negative_at_token_cap'
   elif len(n)<8 or len(n.split())<3:reason='short_negative'
   elif nk in blocked:reason='negative_benchmark_overlap'
   elif nk in dev_visible:reason='negative_dev_overlap'
   elif nk in train_queries:reason='negative_equals_train_query'
   elif nk in known[qk] or nk==pk:reason='negative_is_known_positive'
   elif near_duplicate(n,p):reason='near_duplicate_of_positive'
   if reason:stats[reason]+=1;continue
   restored[identity]=(n,nk);stats['restored_negatives']+=1
 for index,row in enumerate(stream):
  stats['source_rows_scanned']+=1
  identity=(key(row['query']),key(row['positive']))
  if identity in wanted and identity not in restored:
   stats['matching_train_candidates']+=1;pending.append((identity,clean(row['negative']),clean(row['positive'])))
   if len(pending)==256:accept(pending);pending=[]
  if (index+1)%100000==0:print(dict(stats),flush=True)
 accept(pending)
 with (src/'train.jsonl').open() as inp,(out/'train.jsonl').open('w') as f:
  for line in inp:
   r=json.loads(line);identity=(r['query_key'],r['positive_key'])
   if identity in restored:
    r['negative'],r['negative_visible_key']=restored[identity];r['negative_source']='pinned BM25 candidate; filtered, unjudged'
   f.write(json.dumps(r,ensure_ascii=False)+'\n')
 audit={'status':'audited; all original positive pairs preserved','teacher_targets':False,'source':spec,'input_positive_pairs_sha256':sha(src/'train.jsonl'),'dev_sha256':sha(src/'dev.jsonl'),'output_sha256':sha(out/'train.jsonl'),'stats':dict(stats),'reference_text_files':ref_files,'benchmark_qrels_read':False,'negative_policy':'MS MARCO original BM25 candidate; discard cap-truncated, benchmark/dev-visible, known-positive, query-identical and lexical Jaccard>0.8 positive near-duplicate negatives','limits':'These filters do not establish that all remaining candidates are irrelevant. No semantic teacher filtering or target vectors.'}
 atomic_json(out/'audit.json',audit);(ROOT/'hardneg').mkdir(exist_ok=True);atomic_json(ROOT/'hardneg/data-audit.json',audit);print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
