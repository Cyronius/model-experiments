"""Audit additional human NLI labels for a training-only semantic objective.

Use official train/dev splits; keep the auxiliary dev inputs disjoint from all
retrieval training text and block all held/benchmark text from auxiliary train.
"""
import collections,json,os
from pathlib import Path
os.environ.setdefault('RAYON_NUM_THREADS','4')
from datasets import load_dataset
from transformers import AutoTokenizer
from data_first_build import reference_texts,visible_key,sha
from data_first_fetch import clean,valid_pair
from pair_training_state import atomic_json
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
SOURCE={'id':'sentence-transformers/all-nli','config':'pair-class','revision':'d482672c8e74ce18da116f430137434ba2e52fab','label_names':['entailment','neutral','contradiction']}

def main():
 out=ROOT/'work/nli-aux-global';out.mkdir(exist_ok=True)
 if (out/'audit.json').exists():raise RuntimeError('Already audited; do not overwrite')
 tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True);base=ROOT/'work/hardneg/data';base_visible={s:set() for s in ('train','dev')};counts=collections.Counter()
 for split in base_visible:
  for line in (base/(split+'.jsonl')).open():
   r=json.loads(line);base_visible[split].update((r['query_visible_key'],r['positive_visible_key']))
   if r.get('negative'):base_visible[split].add(r['negative_visible_key'])
 assert not(base_visible['train']&base_visible['dev'])
 refs,reference_files=reference_texts();extra=ROOT/'work/expanded/observed-test-texts.jsonl';refs.extend(json.loads(l) for l in extra.open());reference_files[str(extra.relative_to(PROJECT))]=sha(extra);blocked=set()
 for off in range(0,len(refs),512):blocked.update(visible_key(ids).hex() for ids in tk(refs[off:off+512],max_length=128,truncation=True)['input_ids'])
 del refs
 dev_visible=set();train_visible=set();labels={s:collections.Counter() for s in ('train','dev')};seen={s:{} for s in labels}
 for split in ('dev','train'):
  stream=load_dataset(SOURCE['id'],SOURCE['config'],revision=SOURCE['revision'],split=split,streaming=split!='train')
  assert list(stream.features['label'].names)==SOURCE['label_names']
  # A bounded streaming buffer samples an early prefix. Globally shuffle the
  # complete memory-mapped training table so source ordering cannot bias quotas.
  if split=='train':stream=stream.shuffle(seed=41)
  pending=[]
  with (out/(split+'.jsonl')).open('w') as f:
   def accept(batch):
    enc=tk([r[k] for r in batch for k in ('sentence1','sentence2')],truncation=True,max_length=128)['input_ids']
    for i,row in enumerate(batch):
     label=row['label'];a,b=enc[2*i:2*i+2];ak,bk=visible_key(a).hex(),visible_key(b).hex();reason=None
     if split=='train' and labels[split][label]>=100000:continue
     if len(a)>=128 or len(b)>=128:reason='at_token_cap'
     elif ak in blocked or bk in blocked:reason='benchmark_input_overlap'
     elif split=='dev' and (ak in base_visible['train'] or bk in base_visible['train']):reason='dev_retrieval_train_overlap'
     elif split=='train' and any(k in base_visible['dev'] or k in dev_visible for k in (ak,bk)):reason='train_held_input_overlap'
     elif ak==bk:reason='identical_visible_inputs'
     elif (ak,bk) in seen[split]:reason='duplicate_pair' if seen[split][ak,bk]==label else 'conflicting_label'
     if reason:counts[split+'_'+reason]+=1;continue
     seen[split][ak,bk]=label;row.update(visible1=ak,visible2=bk);f.write(json.dumps(row,ensure_ascii=False)+'\n');labels[split][label]+=1
     (dev_visible if split=='dev' else train_visible).update((ak,bk))
   for index,row in enumerate(stream):
    counts[split+'_source_rows']+=1;label=int(row['label'])
    if label not in (0,1,2):counts[split+'_invalid_label']+=1;continue
    if split=='train' and labels[split][label]>=100000:continue
    a,b=clean(row['premise']),clean(row['hypothesis']);reason=valid_pair(a,b,'entailment auxiliary; cap training weight')
    if reason:counts[split+'_'+reason]+=1;continue
    pending.append({'sentence1':a,'sentence2':b,'label':label,'source_row':index})
    if len(pending)==256:accept(pending);pending=[]
    if split=='train' and all(labels[split][k]>=100000 for k in (0,1,2)):break
    if (index+1)%100000==0:print(split,index+1,dict(labels[split]),flush=True)
   if pending:accept(pending)
  print('Complete',split,dict(labels[split]),flush=True)
 assert not(train_visible&dev_visible) and not(train_visible&base_visible['dev']) and not(dev_visible&base_visible['train'])
 audit={'source':SOURCE,'teacher_targets':False,'kind':'Human NLI classification labels, not embedding targets or benchmark test labels','sampling':'Full memory-mapped upstream train split globally shuffled with seed41 before balanced100k/class quotas; no short streaming-buffer prefix sampling','output_directory':str(out.resolve()),'counts':dict(counts),'by_split_label':{k:dict(v) for k,v in labels.items()},'output_sha256':{s:sha(out/(s+'.jsonl')) for s in labels},'retrieval_base_sha256':{s:sha(base/(s+'.jsonl')) for s in labels},'reference_text_files':reference_files,'auxiliary_train_dev_shared_visible_texts':0,'auxiliary_train_retrieval_dev_shared_visible_texts':0,'auxiliary_dev_retrieval_train_shared_visible_texts':0,'reserved_fiqa_scidocs_qrels_read':False,'limits':'Exact model-visible identity exclusions, not exhaustive semantic near-duplicate detection. No embedding quality gain established yet.'}
 atomic_json(out/'audit.json',audit);(ROOT/'nli-aux').mkdir(exist_ok=True);atomic_json(ROOT/'nli-aux/data-audit.json',audit);print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
