"""Full-corpus retrieval using the frozen round-2 model on the real NPU."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'round2'
sys.path.insert(0,str(PROJECT/'native-stage'))
sys.path.insert(0,str(PROJECT/'comparisons/qwen06b'))
from fused_encoder import FusedEncoder,digest
from compare import data
from round2_metrics import evaluate
from transformers import AutoTokenizer
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,default=OUT/'frozen-candidate.json');p.add_argument('--output',type=Path,default=OUT);p.add_argument('--branch',choices=['ternary']);p.add_argument('--quality-only',action='store_true',help='Omit operational timings; latency benchmarks require dibs --measure');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(2);freeze=json.loads(a.freeze.read_text());freeze=freeze['models'][a.branch] if a.branch else freeze;checkpoint=Path(freeze['path'])
 assert digest(checkpoint)==freeze['sha256']
 tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
 e=FusedEncoder(128,checkpoint=checkpoint);report={'checkpoint_sha256':digest(checkpoint),'tasks':{}}
 try:
  for task in ('SciFact','NFCorpus','ArguAna'):
   d=data(task);texts=d['queries']+d['texts'];v=np.empty((len(texts),384),np.float32);start=time.perf_counter()
   for off in range(0,len(texts),100):
    enc=tk(texts[off:off+100],padding='max_length',truncation=True,max_length=128,return_tensors='pt')
    for j in range(len(enc['input_ids'])):v[off+j]=e(enc['input_ids'][j:j+1],enc['attention_mask'][j:j+1].bool(),query=off+j<len(d['queries'])).numpy()[0]
   nq=len(d['queries']);metrics,rows=evaluate(v[:nq]@v[nq:].T,d)
   report['tasks'][task]={'metrics':metrics}
   if not a.quality_only:report['tasks'][task]['seconds']=time.perf_counter()-start
   (a.output/f'npu-{task}-rows.json').write_text(json.dumps(rows)+'\n')
   print(task,report['tasks'][task],flush=True)
  report['mean_core_retrieval']=float(np.mean([report['tasks'][t]['metrics']['ndcg10'] for t in ('SciFact','NFCorpus')]))
  (a.output/'npu-quality.json').write_text(json.dumps(report,indent=2)+'\n')
 finally:e.close()
if __name__=='__main__':main()
