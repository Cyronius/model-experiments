"""Paired CPU text-to-vector benchmark with precomputed ternary weights."""
import datetime,json,sys,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent.parent
sys.path.insert(0,str(PROJECT/'native-stage'))
from encoder import CHECKPOINT,digest
sys.path.insert(0,str(PROJECT/'retrieval-diagnostics'))
from diagnose import student
MODEL=PROJECT/'cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'

@torch.inference_mode()
def main():
    ours=student(CHECKPOINT,'quant')
    qwen=AutoModel.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval()
    ot=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
    qt=AutoTokenizer.from_pretrained(MODEL,padding_side='left',local_files_only=True)
    texts=['A ternary embedding model runs on the NPU.','A small encoder produces text vectors.',
           'Neural networks compare the meaning of sentences.','The weather is sunny today.']
    report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'device':'CPU',
            'torch':torch.__version__,'qwen_revision':MODEL.name,'checkpoint_sha256':digest(CHECKPOINT),
            'ours_arithmetic':'Precomputed ternary effective float32 weights, int8 activation simulation',
            'qwen_arithmetic':'bfloat16 SDPA','warmups':5,'samples':30,'runs':[]}
    for threads in (2,8):
        torch.set_num_threads(threads)
        for padding,cap,batch in [('dynamic',128,1),('fixed',128,1),('fixed',32,4)]:
            def encode(model,tokenizer,name):
                x=tokenizer(texts[:batch],padding=True if padding=='dynamic' else 'max_length',truncation=True,max_length=cap,return_tensors='pt')
                if name=='ours':v=model(x['input_ids'],x['attention_mask'].bool())[2]
                else:v=F.normalize(model(**x).last_hidden_state[:,-1].float(),dim=-1)
                assert torch.isfinite(v).all()
                return v
            for _ in range(5):encode(ours,ot,'ours');encode(qwen,qt,'qwen')
            samples={'ours':[],'qwen':[]}
            for i in range(30):
                for name in (('ours','qwen') if i%2==0 else ('qwen','ours')):
                    model,tk=(ours,ot) if name=='ours' else (qwen,qt)
                    start=time.perf_counter();encode(model,tk,name);samples[name].append((time.perf_counter()-start)*1000)
            medians={name:float(np.median(v)) for name,v in samples.items()}
            row={'threads':threads,'padding':padding,'cap':cap,'batch':batch,'median_ms':medians,
                 'embeddings_per_second':{name:batch*1000/ms for name,ms in medians.items()},
                 'qwen_over_ours_latency':medians['qwen']/medians['ours'],'samples_ms':samples}
            report['runs'].append(row)
            (ROOT/'cpu-speed.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
if __name__=='__main__':main()
