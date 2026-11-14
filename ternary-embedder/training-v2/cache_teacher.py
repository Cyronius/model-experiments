"""Cache Qwen targets on public training triplets only; never evaluation sets."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
REVISIONS={'Qwen/Qwen3-Embedding-0.6B':'97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3',
           'Qwen/Qwen3-Embedding-8B':'1d8ad4ca9b3dd8059ad90a75d4983776a23d44af'}
INSTRUCTION='Given a web search query, retrieve relevant passages that answer the query'
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--reuse-data',type=Path,help='Original data file for an earlier same-teacher text cache')
 p.add_argument('--reuse-targets',type=Path,help='Reuse exact input-text vectors from the earlier data')
 p.add_argument('--plain-queries',action='store_true',help='Symmetric teacher inputs without a query instruction')
 p.add_argument('--reuse-docs',type=Path,help='Reuse same-teacher passage targets from the exact same data')
 p.add_argument('--teacher',choices=list(REVISIONS),default='Qwen/Qwen3-Embedding-0.6B')
 p.add_argument('--download',action='store_true',help='Download selected public teacher if not cached')
 p.add_argument('--data',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True);p.add_argument('--batch',type=int,default=32)
 p.add_argument('--batch-tokens',type=int,default=4096,help='Maximum padded tokens in a teacher batch')
 p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--threads',type=int,default=8)
 a=p.parse_args();torch.set_num_threads(a.threads)
 source=json.loads(a.source.read_text())
 if source.get('split')!='train':raise ValueError('Only an explicit training split is accepted')
 if 'msmarco-bm25' not in json.dumps(source):raise ValueError('Expected the declared MS MARCO training source')
 if source.get('sha256') and source['sha256']!=hashlib.sha256(a.data.read_bytes()).hexdigest():
  raise ValueError('Training source manifest hash mismatch')
 rows=[json.loads(l) for l in a.data.read_text().splitlines()]
 def query_text(r):return r['query'] if a.plain_queries else f"Instruct: {INSTRUCTION}\nQuery:{r['query']}"
 reused={}
 if bool(a.reuse_data)!=bool(a.reuse_targets):p.error('Pass both reuse-data and reuse-targets')
 if a.reuse_data:
  previous=json.loads(a.reuse_targets.with_suffix('.json').read_text())
  if previous['data_sha256']!=hashlib.sha256(a.reuse_data.read_bytes()).hexdigest() or previous['teacher']!=a.teacher or previous['revision']!=REVISIONS[a.teacher]:
   raise ValueError('Text reuse requires verified source data and identical teacher')
  oldrows=[json.loads(l) for l in a.reuse_data.read_text().splitlines()]
  oldvalues=np.load(a.reuse_targets,mmap_mode='r')
  if oldvalues.shape[:2]!=(len(oldrows),3):raise ValueError('Invalid prior teacher cache')
  for i,r in enumerate(oldrows):
   q=r['query'] if previous['instruction'] is None else f"Instruct: {previous['instruction']}\nQuery:{r['query']}"
   reused[q]=oldvalues[i,0];reused[r['positive']]=oldvalues[i,1];reused[r['negative']]=oldvalues[i,2]
 if a.reuse_docs:
  previous=json.loads(a.reuse_docs.with_suffix('.json').read_text())
  if previous['data_sha256']!=hashlib.sha256(a.data.read_bytes()).hexdigest() or previous['teacher']!=a.teacher or previous['revision']!=REVISIONS[a.teacher]:
   raise ValueError('Passage target reuse requires identical data and teacher')
  cache=np.load(a.reuse_docs,mmap_mode='r')
  for i,r in enumerate(rows):
   reused[r['positive']]=cache[i,1];reused[r['negative']]=cache[i,2]
 required=list(dict.fromkeys(t for r in rows for t in (query_text(r),r['positive'],r['negative'])))
 reused={t:reused[t] for t in required if t in reused}
 texts=[t for t in required if t not in reused]
 revision=REVISIONS[a.teacher]
 path=PROJECT/'cache/hub'/('models--'+a.teacher.replace('/','--'))/'snapshots'/revision
 if a.download:
  from huggingface_hub import snapshot_download
  path=Path(snapshot_download(a.teacher,revision=revision,cache_dir=PROJECT/'cache/hub',
          allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json']))
 tk=AutoTokenizer.from_pretrained(path,padding_side='left',local_files_only=True)
 model=AutoModel.from_pretrained(path,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to(a.device).eval()
 if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
 started=time.perf_counter();tokens=0
 print(json.dumps({'tokenizing_unique_texts':len(texts)}),flush=True)
 ids=tk(texts,truncation=True,max_length=128)['input_ids'] if texts else []
 order=np.argsort([len(row) for row in ids],kind='stable')
 vectors=np.empty((len(texts),model.config.hidden_size),np.float32)
 batch=a.batch;i=0
 with torch.inference_mode():
  while i<len(texts):
   indexes=order[i:i+batch]
   while len(indexes)>1 and len(indexes)*len(ids[indexes[-1]])>a.batch_tokens:
    indexes=indexes[:max(1,a.batch_tokens//len(ids[indexes[-1]]))]
   count=len(indexes)
   try:
    enc=tk.pad({'input_ids':[ids[j] for j in indexes]},padding=True,return_tensors='pt').to(a.device)
    output=model(**enc,use_cache=False).last_hidden_state[:,-1].float()
    vector=F.normalize(output,dim=-1).cpu().numpy()
   except torch.OutOfMemoryError:
    if batch==1:raise
    enc=None;output=None;vector=None
    import gc
    gc.collect();torch.cuda.empty_cache();batch=max(1,batch//2)
    print(json.dumps({'memory_backoff_batch':batch}),flush=True)
    continue
   tokens+=int(enc['attention_mask'].sum());vectors[indexes]=vector;i+=count
   print(json.dumps({'encoded':i,'total':len(texts),'batch':batch}),flush=True)
 lookup=dict(reused);lookup.update({t:v for t,v in zip(texts,vectors)})
 values=np.stack([[lookup[query_text(r)],lookup[r['positive']],lookup[r['negative']]] for r in rows])
 a.out.parent.mkdir(parents=True,exist_ok=True);np.save(a.out,values)
 a.out.with_suffix('.json').write_text(json.dumps({'data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),
  'source':source,'teacher':a.teacher,'revision':revision,'instruction':None if a.plain_queries else INSTRUCTION,
  'max_length':128,'dtype':'bfloat16','vectors':list(values.shape),
  'encode_seconds':time.perf_counter()-started,'unique_texts':len(texts),'reused_texts':len(reused),'reuse_data':str(a.reuse_data) if a.reuse_data else None,'real_tokens':tokens,
  'initial_batch':a.batch,'final_batch':batch,'length_bucketing':True,'max_padded_batch_tokens':a.batch_tokens,
  'peak_cuda_bytes':torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None},indent=2)+'\n')
if __name__=='__main__':main()
