"""Audit model-visible duplicates and build connected-component train/dev splits.

Uses benchmark TEXT only for exclusions; never reads relevance judgments.
Exact model-visible matching is not an exhaustive semantic near-duplicate audit.
"""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def visible_key(ids):return hashlib.blake2b(np.asarray(ids,dtype='<u2').tobytes(),digest_size=16).digest()
class Groups:
 def __init__(self):self.index={};self.parent=[];self.size=[];self.minimum=[]
 def node(self,key):
  if key not in self.index:
   self.index[key]=len(self.parent);self.parent.append(len(self.parent));self.size.append(1);self.minimum.append(key)
  return self.index[key]
 def find(self,i):
  while self.parent[i]!=i:self.parent[i]=self.parent[self.parent[i]];i=self.parent[i]
  return i
 def union(self,a,b):
  a,b=self.find(self.node(a)),self.find(self.node(b))
  if a==b:return
  if self.size[a]<self.size[b]:a,b=b,a
  self.parent[b]=a;self.size[a]+=self.size[b];self.minimum[a]=min(self.minimum[a],self.minimum[b])
 def split(self,key,seed,dev_fraction):
  root=self.find(self.index[key]);x=int.from_bytes(hashlib.blake2b(seed.to_bytes(8,'little')+self.minimum[root],digest_size=8).digest(),'little')
  return 'dev' if x/2**64<dev_fraction else 'train'

def reference_texts():
 base=PROJECT/'comparisons/qwen06b/data';texts=set();files={}
 for name in ('scifact','nfcorpus','arguana','fiqa','scidocs'):
  for filename in ('corpus.jsonl','queries.jsonl'):
   path=base/name/filename
   if not path.exists():raise RuntimeError('Missing exclusion text file: '+str(path))
   files[str(path.relative_to(PROJECT))]=sha(path)
   for line in path.open():
    r=json.loads(line);text=r['text'].strip();title=r.get('title','').strip()
    if text:texts.add(text)
    if title:texts.add(title);texts.add((title+' '+text).strip())
 path=base/'stsb-test.json';files[str(path.relative_to(PROJECT))]=sha(path)
 for r in json.loads(path.read_text()):texts.add(r['sentence1']);texts.add(r['sentence2'])
 return list(texts),files

def main():
 from transformers import AutoTokenizer
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=ROOT/'work/data-first/candidates');p.add_argument('--out',type=Path,default=ROOT/'work/data-first/ready');p.add_argument('--seed',type=int,default=23);p.add_argument('--dev-fraction',type=float,default=.02);a=p.parse_args()
 if not 0<a.dev_fraction<.5:p.error('Dev fraction must be between 0 and 0.5')
 a.out.mkdir(parents=True,exist_ok=True)
 if (a.out/'audit.json').exists():raise RuntimeError('Already built; choose a new destination to rebuild')
 tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True);refs,reference_files=reference_texts();blocked=set()
 for off in range(0,len(refs),512):
  blocked.update(visible_key(ids) for ids in tk(refs[off:off+512],max_length=128,truncation=True)['input_ids'])
 print('Reference text identities:',len(blocked),flush=True)
 del refs
 groups=Groups();seen=set();rejected=collections.Counter();stats={};source_files={};intermediate=a.out/'pairs.partial.jsonl';kept=0
 lock=json.loads((ROOT/'data-first/sources.lock.json').read_text())
 with intermediate.open('w') as output:
  for spec in lock:
   slug=spec['id'].split('/')[-1];path=a.input/(slug+'.jsonl');meta=json.loads(path.with_suffix('.manifest.json').read_text());digest=sha(path)
   if digest!=meta['sha256']:raise RuntimeError('Source checksum mismatch')
   if meta['request']['source']!=spec or meta['accepted']!=spec['quota']:raise RuntimeError('Full locked-source quota is required')
   source_files[slug]={'sha256':digest,'manifest':meta};counter=collections.Counter();stats[slug]=counter;batch=[]
   def accept(batch):
    nonlocal kept
    texts=[r[k] for r in batch for k in ('query','positive')]
    encoded=tk(texts,max_length=128,truncation=True,return_length=True)
    for i,row in enumerate(batch):
     qids,pids=encoded['input_ids'][2*i:2*i+2];qk,pk=visible_key(qids),visible_key(pids);counter['candidate_rows']+=1
     reason='benchmark_visible_overlap' if qk in blocked or pk in blocked else 'identical_after_truncation' if qk==pk else None
     pair=(min(qk,pk)+max(qk,pk)) if row['relation']=='duplicate question' else qk+pk
     if reason is None and pair in seen:reason='duplicate_visible_pair'
     if reason:rejected[reason]+=1;counter[reason]+=1;continue
     seen.add(pair);groups.union(qk,pk);row.update(query_visible_key=qk.hex(),positive_visible_key=pk.hex())
     counter['query_at_token_cap']+=len(qids)==128;counter['positive_at_token_cap']+=len(pids)==128
     counter['query_tokens']+=len(qids);counter['positive_tokens']+=len(pids);counter['kept_rows']+=1;kept+=1
     output.write(json.dumps(row,ensure_ascii=False)+'\n')
   for line in path.open():
    batch.append(json.loads(line))
    if len(batch)==256:accept(batch);batch=[]
   if batch:accept(batch)
   print('Visible audit:',slug,dict(counter),flush=True)
 del seen,blocked
 handles={s:(a.out/(s+'.jsonl')).open('w') for s in ('train','dev')};counts=collections.Counter();components=collections.Counter();split_nodes={'train':set(),'dev':set()}
 try:
  for line in intermediate.open():
   r=json.loads(line);qk=bytes.fromhex(r['query_visible_key']);pk=bytes.fromhex(r['positive_visible_key']);split=groups.split(qk,a.seed,a.dev_fraction)
   assert split==groups.split(pk,a.seed,a.dev_fraction)
   handles[split].write(line);counts[split]+=1;stats[r['source'].split('/')[-1]][split]+=1;components[groups.find(groups.index[qk])]+=1
   split_nodes[split].update((qk,pk))
 finally:
  for f in handles.values():f.close()
 overlap=len(split_nodes['train']&split_nodes['dev']);assert overlap==0
 audit={'status':'positive-pair dataset built; near-duplicate review and native training preflight remain','teacher_targets':False,'architecture_unchanged':True,'token_cap':128,'seed':a.seed,'dev_fraction_requested':a.dev_fraction,'kept_pairs':kept,'splits':dict(counts),'rejected':dict(rejected),'by_source':{k:dict(v) for k,v in stats.items()},'components':len(components),'largest_component_rows':max(components.values()),'train_dev_shared_visible_texts':overlap,'reference_text_files':reference_files,'reserved_unscored_tests':['FiQA2018','SCIDOCS'],'sources':source_files,'output_sha256':{s:sha(a.out/(s+'.jsonl')) for s in handles},'limits':'Exact model-visible deduplication; not exhaustive semantic near-duplicate detection. Test qrels were not read.'}
 (a.out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');(ROOT/'data-first/audit.json').write_text(json.dumps(audit,indent=2)+'\n');intermediate.unlink();print('BUILT',counts,'largest component',audit['largest_component_rows'],flush=True)
if __name__=='__main__':main()
