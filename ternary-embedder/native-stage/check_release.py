"""Verify, time and retrieval-score a released ternary embedder folder on the NPU against CPU/GPU reference vectors (2026-10-28)."""
import argparse
import datetime
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from fused_encoder import FusedEncoder,ROOT
from bench_encoder import timing

sys.path.insert(0,str(ROOT.parent/'retrieval-diagnostics'))
from diagnose import dataset,evaluate,paired
from model import Student
from safetensors.torch import load_file
from transformers import AutoTokenizer

CACHE=Path('C:/Users/josha/.cache/huggingface/datasets')
CORPORA=('SciFact','NFCorpus')


def texts_file(out):
    data={n:dataset(n,CACHE) for n in CORPORA}
    sci=data['SciFact']
    payload={'query':sci['queries'][:1],'micro':sci['texts'][:4],
             'corpora':{n:d['queries']+d['texts'] for n,d in data.items()}}
    out.write_text(json.dumps(payload),encoding='utf-8')
    return data,payload


def float_student(model_dir):
    m=Student(**json.loads((model_dir/'config.json').read_text())['dims'])
    m.load_state_dict(load_file(model_dir/'model.safetensors'));m.set_quant(True)
    return m.eval()


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path)
    p.add_argument('--out',type=Path,default=ROOT/'results/release-check')
    p.add_argument('--dump-texts',action='store_true',help='Write texts.json for bench_ref.py and stop')
    p.add_argument('--samples',type=int,default=15)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    data,payload=texts_file(a.out/'texts.json')
    if a.dump_texts:
        return
    torch.set_num_threads(2)
    tk=AutoTokenizer.from_pretrained(a.model)
    ref=float_student(a.model)
    report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':str(a.model),
            'weights':'model.ternary.safetensors' if (a.model/'model.ternary.safetensors').is_file() else 'model.safetensors',
            'runs':{}}

    def pack(texts,cap):
        enc=tk(texts,padding='max_length',truncation=True,max_length=cap,return_tensors='pt')
        return enc['input_ids'],enc['attention_mask'].bool()

    for cap in (32,128):
        engine=FusedEncoder(cap,checkpoint=a.model)
        try:
            per=128//cap
            texts=data['SciFact']['texts'][:8]+data['SciFact']['queries'][:8]
            got,want,errors=[],[],[]
            for s in range(0,len(texts),per):
                ids,mask=pack(texts[s:s+per],cap)
                v,states=engine(ids,mask,trace=True)
                rs,_,rv=ref(ids.masked_fill(~mask,0),mask)
                got.append(v);want.append(rv)
                errors.append([float((x-r)[mask].abs().max()) for x,r in zip(states,rs)])
            got,want=torch.cat(got),torch.cat(want)
            cos=F.cosine_similarity(got,want)
            row={'build':engine.manifest['artifact_sha256'],'program_text_bytes':engine.manifest['program_text_bytes'],
                 'verification':{'texts':len(texts),'min_cosine':float(cos.min()),'mean_cosine':float(cos.mean()),
                                 'max_vector_abs_error':float((got-want).abs().max()),
                                 'max_valid_state_abs_error_per_layer':np.max(errors,axis=0).tolist()}}
            print(json.dumps({'cap':cap,**row['verification']}),flush=True)
            if cos.min()<.995:
                raise AssertionError(f'NPU vectors disagree with the CPU reference: {row["verification"]}')
            q=pack(payload['query'],cap)
            row['one_query_forward']=timing(lambda:engine(*q),a.samples,1)
            row['one_query_text_to_vector']=timing(lambda:engine(*pack(payload['query'],cap)),a.samples,1)
            micro=payload['micro'][:per]
            row['full_batch_text_to_vector']=timing(lambda:engine(*pack(micro,cap)),a.samples,per)
            engine.profile(reset=True)
            print(json.dumps({'cap':cap,'timing':{k:{x:v[x] for x in ('median_ms','embeddings_per_second')}
                              for k,v in row.items() if k.endswith(('forward','vector'))}}),flush=True)
            if cap==128:
                row['bulk']={}
                for name in CORPORA:
                    texts=payload['corpora'][name]
                    vectors=np.empty((len(texts),384),np.float32)
                    start=time.perf_counter()
                    for off in range(0,len(texts),100):
                        ids,mask=pack(texts[off:off+100],128)
                        for i in range(len(ids)):
                            vectors[off+i]=engine(ids[i:i+1],mask[i:i+1]).numpy()[0]
                    elapsed=time.perf_counter()-start
                    np.save(a.out/f'npu_{name}.npy',vectors)
                    row['bulk'][name]={'texts':len(texts),'seconds':elapsed,'embeddings_per_second':len(texts)/elapsed,
                                       'dispatch_median_ms':float(np.median(engine.profile(reset=True)['dispatch_samples_ms']))}
                    print(json.dumps({name:row['bulk'][name]}),flush=True)
            report['runs'][cap]=row
        finally:
            engine.close()

    report['retrieval']={}
    for name in CORPORA:
        d=data[name];nq=len(d['queries'])
        scored={}
        for source,path in (('npu',a.out/f'npu_{name}.npy'),('cpu',a.out/'cpu'/f'{name}.npy'),('gpu',a.out/'gpu'/f'{name}.npy')):
            if path.is_file():
                v=np.load(path);scored[source]=(v,)+evaluate(v[:nq]@v[nq:].T,d)
        item={k:s[1]['ndcg10'] for k,s in scored.items()}
        if 'cpu' in scored:
            cpu=scored['cpu'][0]
            for k in scored:
                if k!='cpu':
                    v=scored[k][0]
                    item[f'{k}_vs_cpu']={'paired':paired(scored['cpu'][2],scored[k][2]),
                                         'cosine_min':float(np.min(np.sum(v*cpu,1))),
                                         'max_abs_error':float(np.abs(v-cpu).max())}
        report['retrieval'][name]=item
        print(json.dumps({name:item}),flush=True)
    (a.out/'npu-report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
