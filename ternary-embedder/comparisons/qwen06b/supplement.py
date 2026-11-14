"""Dynamic-padding CPU timings and matched STSBenchmark test-set quality."""
import json,time,sys
from pathlib import Path
import numpy as np,torch
from scipy.stats import spearmanr
from transformers import AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
sys.path.insert(0,str(PROJECT/'native-stage'))
from fused_encoder import FusedEncoder,digest
MODEL=PROJECT/'cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'

@torch.inference_mode()
def main():
 m=AutoModel.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval()
 tk=AutoTokenizer.from_pretrained(MODEL,padding_side='left',local_files_only=True)
 ntk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
 def encode(texts):
  b=tk(texts,padding=True,truncation=True,max_length=128,return_tensors='pt')
  return torch.nn.functional.normalize(m(**b).last_hidden_state[:,-1].float(),dim=-1).numpy()
 prior=ROOT/'supplement.json'
 rows=json.loads(prior.read_text())['dynamic_speed'] if prior.exists() else [];text=['A ternary embedding model runs on the NPU.']
 for threads in (() if rows else (2,8)):
  torch.set_num_threads(threads)
  for _ in range(5):encode(text)
  samples=[]
  for _ in range(30):
   t=time.perf_counter();encode(text);samples.append((time.perf_counter()-t)*1000)
  rows.append({'threads':threads,'tokens':len(tk(text[0])['input_ids']),'median_ms':float(np.median(samples)),'samples_ms':samples})
 print('DYNAMIC',json.dumps(rows),flush=True)
 report={'qwen_revision':MODEL.name,'qwen_dtype':'bfloat16','dynamic_speed':rows,'sts':{},'sts_instruction':'Retrieve semantically similar text.'}
 (ROOT/'supplement.json').write_text(json.dumps(report,indent=2)+'\n')
 torch.set_num_threads(8)
 pairs=json.loads((ROOT/'data/stsb-test.json').read_text());texts=[r['sentence1'] for r in pairs]+[r['sentence2'] for r in pairs]
 for name in ['qwen_cpu','ternary_npu','qwen_cpu_prompted']:
  path=ROOT/f'STSBenchmark_{name}.npy'
  if path.exists():v=np.load(path)
  else:
   chunks=[];start=time.perf_counter()
   if name=='ternary_npu':engine=FusedEncoder(128)
   try:
    for i in range(0,len(texts),32):
     block=texts[i:i+32]
     if name.startswith('qwen_cpu'):
      chunks.append(encode([f'Instruct: Retrieve semantically similar text.\nQuery:{t}' for t in block] if name.endswith('prompted') else block))
     else:
      b=ntk(block,padding='max_length',truncation=True,max_length=128,return_tensors='pt')
      chunks.append(np.concatenate([engine(b['input_ids'][j:j+1],b['attention_mask'][j:j+1].bool()).numpy() for j in range(len(block))]))
     if i%320==0:print('STS',name,i,len(texts),round(time.perf_counter()-start,1),flush=True)
    v=np.concatenate(chunks);np.save(path,v)
   finally:
    if name=='ternary_npu':engine.close()
  n=len(pairs);sims=np.sum(v[:n]*v[n:],axis=1)
  report['sts'][name]={'spearman':float(spearmanr([float(r['score']) for r in pairs],sims).statistic)*100,'pairs':n,'max_length':128,'query_prompts':name.endswith('prompted')}
  (ROOT/'supplement.json').write_text(json.dumps(report,indent=2)+'\n')
  print('STS_SCORE',name,json.dumps(report['sts'][name]),flush=True)
 report['sts_data_sha256']=digest(ROOT/'data/stsb-test.json')
 (ROOT/'supplement.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
