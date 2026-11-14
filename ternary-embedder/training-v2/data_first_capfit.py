"""Conservative first-run subset: full query and positive fit below the token cap."""
import collections,json
from pathlib import Path
from transformers import AutoTokenizer
from data_first_build import ROOT,PROJECT,sha

def main():
 src=ROOT/'work/data-first/ready';out=ROOT/'work/data-first/capfit';out.mkdir(parents=True,exist_ok=True)
 if (out/'audit.json').exists():raise RuntimeError('Cap-fit subset already exists')
 tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True);report={'policy':'Keep both tokenized inputs strictly below 128 tokens; conservative exclusion includes texts exactly at cap','teacher_targets':False,'source_audit_sha256':sha(src/'audit.json'),'splits':{},'by_source':{},'dropped_at_cap':{},'train_dev_shared_visible_texts':0,'limits':'Input completeness does not guarantee correct relevance labels or remove all semantic near duplicates'}
 for split in ('train','dev'):
  batch=[];kept=dropped=0;counter=collections.Counter();dropcounter=collections.Counter()
  with (out/(split+'.jsonl')).open('w') as f:
   def accept(batch):
    nonlocal kept,dropped
    ids=tk([r[k] for r in batch for k in ('query','positive')],max_length=128,truncation=True)['input_ids']
    for i,r in enumerate(batch):
     if len(ids[2*i])>=128 or len(ids[2*i+1])>=128:dropped+=1;dropcounter[r['source']]+=1;continue
     r['query_tokens']=len(ids[2*i]);r['positive_tokens']=len(ids[2*i+1]);f.write(json.dumps(r,ensure_ascii=False)+'\n');kept+=1;counter[r['source']]+=1
   for line in (src/(split+'.jsonl')).open():
    batch.append(json.loads(line))
    if len(batch)==256:accept(batch);batch=[]
   if batch:accept(batch)
  report['splits'][split]=kept;report['by_source'][split]=dict(counter);report['dropped_at_cap'][split]=dict(dropcounter)
  print(split,'kept',kept,'dropped at cap',dropped,flush=True)
 report['output_sha256']={s:sha(out/(s+'.jsonl')) for s in report['splits']}
 (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');(ROOT/'data-first/capfit-audit.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
