"""Validate the frozen candidate on the unchanged NPU graph and time its CLI path."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'native-stage'))
from fused_encoder import FusedEncoder,digest
from query_head import query_role

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,default=ROOT/'frozen-candidate.json');p.add_argument('--output',type=Path,default=ROOT/'candidate-npu.json');p.add_argument('--branch',choices=['ternary']);p.add_argument('--verify-only',action='store_true',help='Check vectors without reporting latency; timing runs require dibs --measure');p.add_argument('--single',action='store_true',help='Verify and time one text per dispatch at both sequence caps');p.add_argument('--query',action='store_true');a=p.parse_args()
 torch.set_num_threads(2);frozen=json.loads(a.freeze.read_text());frozen=frozen['models'][a.branch] if a.branch else frozen;checkpoint=Path(frozen['path'])
 assert digest(checkpoint)==frozen['sha256']
 tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
 texts=['A ternary embedding model runs on the NPU.','A small encoder produces text vectors.',
        'Neural networks compare the meaning of sentences.','The weather is sunny today.']
 report={'checkpoint_sha256':digest(checkpoint),'timings_requested':not a.verify_only,'single_text':a.single,'cpu_threads':2,'cpu_affinity':sorted(os.sched_getaffinity(0)),'runs':[]}
 for cap in (32,128):
  e=FusedEncoder(cap,checkpoint=checkpoint)
  try:
   fixture=np.load(PROJECT/f'native-stage/fixtures/s{cap}.trace.npz')
   ids=torch.from_numpy(fixture['ids']);mask=torch.from_numpy(fixture['mask']);batch=1 if a.single else 128//cap
   got=[]
   for i in range(0,4,batch):got.append(e(ids[i:i+batch],mask[i:i+batch],query=a.query))
   got=torch.cat(got)
   with query_role(e.model,a.query):ref=e.model(ids,mask)[2]
   cos=F.cosine_similarity(got,ref)
   assert float(cos.min())>.995
   def run():
    enc=tk(texts[:batch],padding='max_length',truncation=True,max_length=cap,return_tensors='pt')
    return e(enc['input_ids'],enc['attention_mask'].bool(),query=a.query)
   row={'cap':cap,'batch':batch,'min_cosine_to_cpu':float(cos.min()),'max_vector_abs_error':float((got-ref).abs().max()),
        'verification_only':a.verify_only,'query_role':a.query}
   if not a.verify_only:
    warm_start=time.perf_counter();warm_calls=0
    while warm_calls<5 or time.perf_counter()-warm_start<.75:run();warm_calls+=1
    samples=[]
    for _ in range(30):
     tick=time.perf_counter();run();samples.append((time.perf_counter()-tick)*1000)
    row.update(median_text_to_vector_ms=float(np.median(samples)),samples_ms=samples,warmup_calls=warm_calls,minimum_warmup_seconds=.75)
   report['runs'].append(row);print(row,flush=True)
  finally:e.close()
 a.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
