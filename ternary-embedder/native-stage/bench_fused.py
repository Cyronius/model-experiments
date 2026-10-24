"""Verify and benchmark the one-dispatch full encoder body."""
import argparse
import datetime
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from encoder import ROOT,CHECKPOINT,digest,DLL
from fused_encoder import FusedEncoder
from bench_encoder import pack,timing

sys.path.insert(0,str(ROOT.parent/'retrieval-diagnostics'))
from diagnose import student,dataset,TOKENIZER,MINILM
from transformers import AutoTokenizer,AutoModel
from transformers.utils import logging
logging.disable_progress_bar()


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--caps',type=int,nargs='+',default=[32,128])
    p.add_argument('--samples',type=int,default=15)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    ref=student(CHECKPOINT,'quant')
    tk=AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True)
    data=dataset('SciFact',Path('C:/Users/josha/.cache/huggingface/datasets'))
    report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'checkpoint_sha256':digest(CHECKPOINT),'runtime_dll_sha256':digest(DLL),
            'scope':'Both complete encoder layers on NPU in one dispatch; CPU tokenizer, input embeddings/norm, pooling, float output projection',
            'arithmetic':'int8 activations, packed ternary weights, bf16 Q/K/V/attention probabilities and GELU outputs, split-bf16 fp32 math',
            'threads':args.threads,'runs':[]}
    mini=None
    for cap in args.caps:
        engine=FusedEncoder(cap)
        row={'cap':cap,'batch':128//cap,'build':engine.manifest}
        try:
            trace=np.load(ROOT/f'fixtures/s{cap}.trace.npz')
            ids,mask=torch.from_numpy(trace['ids']),torch.from_numpy(trace['mask'])
            got,refs,states=[],[],[]
            for start in range(0,4,128//cap):
                a,b=ids[start:start+128//cap],mask[start:start+128//cap]
                v,s=engine(a,b,trace=True)
                got.append(v);states.append(s);refs.append(ref(a,b))
            vector=torch.cat(got);reference=torch.cat([r[2] for r in refs])
            cosine=F.cosine_similarity(vector,reference)
            row['verification']={'max_vector_abs_error':float((vector-reference).abs().max()),
                                 'min_cosine':float(cosine.min()),'mean_cosine':float(cosine.mean()),
                                 'max_state_abs_errors':[float((torch.cat([s[i] for s in states])-torch.cat([r[0][i] for r in refs])).abs().max()) for i in range(3)],
                                 'max_valid_state_abs_errors':[float((torch.cat([s[i] for s in states])-torch.cat([r[0][i] for r in refs]))[mask].abs().max()) for i in range(3)],
                                 'masked_query_states_may_differ':True}
            print(json.dumps({'cap':cap,'verification':row['verification'],'profile':engine.profile()}),flush=True)
            # Approximate bf16 nonlinear arithmetic is judged by final vectors,
            # then by an independent full-corpus retrieval evaluation.
            if cosine.min()<.995:
                raise AssertionError(f'Full encoder numerical failure: {row["verification"]}')
            if not args.verify_only:
                if mini is None:
                    mt=AutoTokenizer.from_pretrained(MINILM,local_files_only=True)
                    mini=AutoModel.from_pretrained(MINILM,local_files_only=True).eval()
                texts=data['texts'][:128//cap]
                ids,mask=pack(tk,texts,cap)
                def e2e():
                    a,b=pack(tk,texts,cap)
                    return engine(a,b)
                def baseline():
                    enc=mt(texts,padding='max_length',truncation=True,max_length=cap,return_tensors='pt')
                    h=mini(**enc).last_hidden_state;m=enc['attention_mask']
                    return F.normalize((h*m[...,None]).sum(1)/m.sum(1,keepdim=True),dim=-1)
                engine.profile(reset=True)
                row['forward']=timing(lambda:engine(ids,mask),args.samples,len(texts))
                row['forward_profile']=engine.profile(reset=True)
                row['text_to_vector']=timing(e2e,args.samples,len(texts))
                row['e2e_profile']=engine.profile(reset=True)
                row['cpu_minilm_text_to_vector']=timing(baseline,args.samples,len(texts))
                print(json.dumps({'cap':cap,'timing':{k:v for k,v in row.items() if k not in ('build','verification')}}),flush=True)
            report['runs'].append(row)
            filename='fused-verification.json' if args.verify_only else 'fused-result.json'
            (ROOT/filename).write_text(json.dumps(report,indent=2)+'\n')
        finally:
            engine.close()


if __name__=='__main__':
    main()
