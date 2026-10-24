"""Verify final NPU-backed vectors, then time real text-to-vector requests."""
import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from encoder import Encoder, ROOT, DLL, CHECKPOINT, digest

DIAG = ROOT.parent/'retrieval-diagnostics'
sys.path.insert(0,str(DIAG))
from diagnose import dataset, student, TOKENIZER, MINILM
from transformers import AutoTokenizer, AutoModel


def pack(tokenizer,texts,cap):
    enc = tokenizer(texts,padding='max_length',truncation=True,max_length=cap,return_tensors='pt')
    return enc['input_ids'],enc['attention_mask'].bool()


def timing(fn, samples, count):
    fn()
    times = []
    for _ in range(samples):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter()-t)*1000)
    return {'samples_ms':times, 'median_ms':statistics.median(times),
            'p95_ms':float(np.percentile(times,95)),
            'embeddings_per_second':count*1000/statistics.median(times)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples',type=int,default=9)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    engine = Encoder()
    report = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope':'Complete hybrid encoder: 12 NPU ternary projections, CPU attention/nonlinear ops/pooling',
              'checkpoint_sha256':digest(CHECKPOINT), 'runtime_dll':str(DLL),
              'runtime_sha256':digest(DLL),'threads':args.threads,
              'builds':{str(s):p.manifest for s,p in engine.projections.items()},'verification':{},'benchmarks':[]}
    out = ROOT/('encoder-verification.json' if args.verify_only else 'encoder-result.json')
    try:
        reference = student(CHECKPOINT,'quant')
        for cap in (8,32,128):
            trace = np.load(ROOT/f'fixtures/s{cap}.trace.npz')
            ids,mask = torch.from_numpy(trace['ids']),torch.from_numpy(trace['mask'])
            vectors, states, refs = [],[],[]
            for start in range(0,4,128//cap):
                a,b = ids[start:start+128//cap],mask[start:start+128//cap]
                v,s = engine(a,b,trace=True)
                vectors.append(v)
                states.append(s)
                refs.append(reference(a,b))
            got = torch.cat(vectors)
            ref = torch.cat([r[2] for r in refs])
            delta = (got-ref).abs()
            torch.testing.assert_close(got,ref,atol=2e-5,rtol=2e-4)
            layer_errors = []
            for li in range(3):
                a = torch.cat([s[li] for s in states])
                b = torch.cat([r[0][li] for r in refs])
                layer_errors.append(float((a-b).abs().max()))
                torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-4)
            report['verification'][f'cap{cap}'] = {'texts':4,'token_rows':4*cap,
                'max_vector_abs_error':float(delta.max()),'min_cosine':float(F.cosine_similarity(got,ref).min()),
                'max_state_abs_errors':layer_errors}
            print(json.dumps({'verification':report['verification'][f'cap{cap}'],'cap':cap}),flush=True)
        if not args.verify_only:
            tk = AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True)
            mt = AutoTokenizer.from_pretrained(MINILM,local_files_only=True)
            mini = AutoModel.from_pretrained(MINILM,local_files_only=True).eval()
            data = dataset('SciFact',Path('C:/Users/josha/.cache/huggingface/datasets'))
            for cap in (32,128):
                batch = 128//cap
                texts = data['texts'][:batch]
                ids,mask = pack(tk,texts,cap)
                mi,mm = pack(mt,texts,cap)
                def npu_e2e():
                    a,b = pack(tk,texts,cap)
                    return engine(a,b)
                def mini_forward():
                    h = mini(input_ids=mi,attention_mask=mm.long()).last_hidden_state
                    return F.normalize((h*mm[...,None]).sum(1)/mm.sum(1,keepdim=True),dim=-1)
                def mini_e2e():
                    enc = mt(texts,padding='max_length',truncation=True,max_length=cap,return_tensors='pt')
                    h = mini(**enc).last_hidden_state
                    m = enc['attention_mask']
                    return F.normalize((h*m[...,None]).sum(1)/m.sum(1,keepdim=True),dim=-1)
                engine.profile(reset=True)
                row = {'cap':cap,'batch':batch,'texts':len(texts),'padding':'fixed max_length',
                       'npu_forward':timing(lambda:engine(ids,mask),args.samples,batch),
                       'npu_profile':engine.profile(reset=True),
                       'npu_text_to_vector':timing(npu_e2e,args.samples,batch),
                       'cpu_student_forward':timing(lambda:reference(ids,mask),args.samples,batch),
                       'cpu_minilm_forward':timing(mini_forward,args.samples,batch),
                       'cpu_minilm_text_to_vector':timing(mini_e2e,args.samples,batch)}
                report['benchmarks'].append(row)
                print(json.dumps(row),flush=True)
        report['verification_profile'] = engine.profile()
        out.write_text(json.dumps(report,indent=2)+'\n')
    finally:
        engine.close()


if __name__ == '__main__':
    main()
