"""Add a disjoint hard-negative validation pool, without opening benchmark labels."""
import collections,json
from datasets import load_dataset
from transformers import AutoTokenizer
from restore_pair_negatives import ROOT,PROJECT,near_duplicate
from data_first_fetch import clean,key
from data_first_build import visible_key,reference_texts,sha
from pair_training_state import atomic_json

def main():
 src=ROOT/'work/data-first/capfit';out=ROOT/'work/hardneg/data'
 if (out/'dev-audit.json').exists():raise RuntimeError('Validation data already audited')
 tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True);blocked=set()
 for line in (out/'train.jsonl').open():
  r=json.loads(line);blocked.update((r['query_visible_key'],r['positive_visible_key']))
  if r.get('negative'):blocked.add(r['negative_visible_key'])
 refs,refs_files=reference_texts()
 for off in range(0,len(refs),512):blocked.update(visible_key(ids).hex() for ids in tk(refs[off:off+512],max_length=128,truncation=True)['input_ids'])
 rows=[json.loads(l) for l in (src/'dev.jsonl').open()];wanted={};known=collections.defaultdict(set);queries=set()
 for i,r in enumerate(rows):
  known[r['query_visible_key']].add(r['positive_visible_key']);queries.add(r['query_visible_key'])
  if r['source']=='sentence-transformers/msmarco-bm25':wanted[(r['query_key'],r['positive_key'])]=i
 spec=json.loads((ROOT/'data-first/sources.lock.json').read_text())[0];stats=collections.Counter();pending=[];seen=set()
 def accept(batch):
  if not batch:return
  enc=tk([x[1] for x in batch],truncation=True,max_length=128)['input_ids']
  for (identity,n,p),ids in zip(batch,enc):
   row=rows[wanted[identity]];nk=visible_key(ids).hex();reason=None
   if len(ids)>=128:reason='negative_at_cap'
   elif len(n)<8 or len(n.split())<3:reason='short_negative'
   elif nk in blocked:reason='negative_train_or_benchmark_overlap'
   elif nk in queries:reason='negative_equals_dev_query'
   elif nk in known[row['query_visible_key']]:reason='known_positive'
   elif near_duplicate(n,p):reason='near_duplicate_of_positive'
   if reason:stats[reason]+=1;continue
   row.update(negative=n,negative_visible_key=nk,negative_source='pinned BM25 candidate; disjoint validation pool');stats['retained_negatives']+=1
 for r in load_dataset(spec['id'],spec['config'],revision=spec['revision'],split='train',streaming=True):
  identity=(key(r['query']),key(r['positive']))
  if identity in wanted and identity not in seen:
   seen.add(identity);pending.append((identity,clean(r['negative']),clean(r['positive'])))
   if len(pending)==256:accept(pending);pending=[]
 accept(pending)
 with (out/'dev.jsonl').open('w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 audit={'positive_dev_rows':len(rows),'stats':dict(stats),'train_sha256':sha(out/'train.jsonl'),'dev_sha256':sha(out/'dev.jsonl'),'benchmark_qrels_read':False,'negative_training_text_overlap':0,'reference_text_files':refs_files,'limits':'Audited original BM25 candidates remain unjudged for relevance; exact model-visible exclusion only'}
 atomic_json(out/'dev-audit.json',audit);atomic_json(ROOT/'hardneg/dev-audit.json',audit);print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
