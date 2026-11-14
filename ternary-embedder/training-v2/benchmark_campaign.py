"""Matched CPU latency of frozen candidates and local references, including tokenization."""
import argparse,json,os,platform,sys,time
from pathlib import Path
import numpy as np,torch
from transformers import AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'retrieval-diagnostics'))
from diagnose import student
from pair_training_state import digest
from pair_reference_validation import MeanEncoder

class QwenEncoder(torch.nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,ids,mask):
  # Left padding keeps each nonempty sequence's last real token at index -1.
  h=self.model(input_ids=ids,attention_mask=mask.long()).last_hidden_state[:,-1].float()
  return None,None,torch.nn.functional.normalize(h,dim=-1)

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--qwen',action='store_true');a=p.parse_args();torch.set_num_threads(2)
 frozen=json.loads(a.freeze.read_text());tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
 result={'device':'CPU','threads':2,'platform':platform.platform(),'torch':torch.__version__,'scope':'Cached effective ternary weights with int8 activation simulation, executed through dense floating-point CPU kernels; not native packed ternary CPU speed. Batch1, loaded weights, tokenization included; fixed padded32/128 tokens.','timings':{},'model_metadata':{}}
 result['cpu_affinity']=sorted(os.sched_getaffinity(0));result['minimum_warmup_seconds']=.75
 def measure(model,tokenizer,baseline,prefix=''):
  out={}
  for cap,text in [(32,'The cat sits beside the window while the rain falls.'),(128,'Scientific evidence supports careful testing of retrieval models. '*20)]:
   def call():
    e=tokenizer([prefix+text],padding='max_length',truncation=True,max_length=cap,return_tensors='pt')
    return model(e['input_ids'],e['attention_mask'].bool())[2]
   warm_start=time.perf_counter();warm_calls=0
   while warm_calls<8 or time.perf_counter()-warm_start<.75:call();warm_calls+=1
   samples=[]
   for _ in range(60):
    tick=time.perf_counter();v=call();samples.append((time.perf_counter()-tick)*1000)
   assert torch.isfinite(v).all()
   assert float((v.norm(dim=-1)-1).abs().max())<1e-4
   out[str(cap)]={'batch':1,'padded_tokens':cap,'real_tokens':int(tokenizer([prefix+text],truncation=True,max_length=cap,return_tensors='pt')['attention_mask'].sum()),'median_ms':float(np.median(samples)),'p90_ms':float(np.percentile(samples,90)),'samples_ms':samples}
   out[str(cap)]['warmup_calls']=warm_calls
  return out
 for name,selection in frozen['models'].items():
  path=Path(selection['path']);assert digest(path)==selection['sha256'];model=student(path,'quant' if selection['precision']=='ternary' else 'float')
  result['timings'][name]=measure(model,tk,False)
  result['model_metadata'][name]={'checkpoint_sha256':selection['sha256'],'precision':selection['precision'],'layers':selection['layers'],'dtype':'float32','dimensions':384}
 for name,repo,revision in [('bge_micro','TaylorAI/bge-micro-v2','3edf6d7de0faa426b09780416fe61009f26ae589'),('minilm','sentence-transformers/all-MiniLM-L6-v2','1110a243fdf4706b3f48f1d95db1a4f5529b4d41')]:
  path=PROJECT/'cache/hub'/('models--'+repo.replace('/','--'))/'snapshots'/revision;model=MeanEncoder(AutoModel.from_pretrained(path,local_files_only=True)).eval();tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
  result['timings'][name]=measure(model,tokenizer,True)
  result['model_metadata'][name]={'repo':repo,'revision':revision,'dtype':'float32','dimensions':384}
 if a.qwen:
  revision='97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3';path=PROJECT/'cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots'/revision
  tokenizer=AutoTokenizer.from_pretrained(path,padding_side='left',local_files_only=True)
  model=QwenEncoder(AutoModel.from_pretrained(path,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)).eval()
  for name,prefix in [('qwen_0_6b_document',''),('qwen_0_6b_query','Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ')]:
   result['timings'][name]=measure(model,tokenizer,True,prefix)
   result['model_metadata'][name]={'repo':'Qwen/Qwen3-Embedding-0.6B','revision':revision,'dtype':'bfloat16','dimensions':1024,'padding_side':'left','query_instruction_prefix':prefix}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print({k:{c:round(v['median_ms'],3) for c,v in row.items()} for k,row in result['timings'].items()})
if __name__=='__main__':main()
