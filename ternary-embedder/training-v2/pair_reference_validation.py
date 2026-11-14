"""Put internal source-validation numbers in context using local reference models."""
import argparse,json
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModel
from pair_validation import PairValidation
from pair_training_state import digest
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
class MeanEncoder(torch.nn.Module):
 def __init__(self,model):super().__init__();self.model=model;self.dims={'out_dim':384}
 def forward(self,ids,mask):
  h=self.model(input_ids=ids,attention_mask=mask.long()).last_hidden_state;v=(h*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
  return None,None,torch.nn.functional.normalize(v,dim=-1)
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['bge_micro','minilm'],required=True);a=p.parse_args();torch.set_num_threads(2)
 repo={'bge_micro':'TaylorAI/bge-micro-v2','minilm':'sentence-transformers/all-MiniLM-L6-v2'}[a.model];snapshots=PROJECT/'cache/hub'/('models--'+repo.replace('/','--'))/'snapshots';path=next(snapshots.iterdir())
 tk=AutoTokenizer.from_pretrained(path,local_files_only=True);model=MeanEncoder(AutoModel.from_pretrained(path,local_files_only=True)).eval();data=ROOT/'work/data-first/capfit/dev.jsonl';rows=[json.loads(l) for l in data.read_text().splitlines()]
 result=PairValidation(rows,tk)(model,'cpu',64);result.update(model=repo,revision=path.name,data_sha256=digest(data),token_cap=128,scope='Same internal validation protocol; not independent test quality')
 out=ROOT/'data-first/results';(out/('internal-reference-'+a.model+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(a.model,result['macro_ndcg10'],flush=True)
if __name__=='__main__':main()
