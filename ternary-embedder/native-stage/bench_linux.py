"""Offline Linux verification and paired output-transfer benchmark."""
import argparse
import datetime
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from fused_encoder import FusedEncoder, ROOT, CHECKPOINT, DLL, digest
from transformers import AutoTokenizer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=int,default=50)
    p.add_argument('--output',type=Path,default=ROOT/'linux-result.json')
    a=p.parse_args()
    torch.set_num_threads(2)
    tk=AutoTokenizer.from_pretrained(ROOT.parent/'tokenizer',local_files_only=True)
    texts=['A ternary embedding model runs on the NPU.',
           'A small encoder produces text vectors.',
           'Neural networks compare the meaning of sentences.',
           'The weather is sunny today.']
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'checkpoint_sha256':digest(CHECKPOINT),'runtime_sha256':digest(DLL),
            'host_source_sha256':{name:digest(ROOT/name) for name in
                                  ('runtime.py','encoder.py','fused_encoder.py','host/td_npu.cpp')},
            'threads':2,'samples':a.samples,'warmups':5,'runs':[]}
    with torch.inference_mode():
        for cap in (32,128):
            e=FusedEncoder(cap)
            try:
                fixture=np.load(ROOT/f'fixtures/s{cap}.trace.npz')
                ids=torch.from_numpy(fixture['ids']);mask=torch.from_numpy(fixture['mask'])
                actual=[];traces=[]
                batch=128//cap
                for i in range(0,len(ids),batch):
                    v,states=e(ids[i:i+batch],mask[i:i+batch],trace=True)
                    actual.append(v);traces.append(states)
                got=torch.cat(actual);ref=e.model(ids,mask)[2]
                cosine=F.cosine_similarity(got,ref)
                assert float(cosine.min())>.995
                fixture_ref=torch.from_numpy(fixture['vectors'])
                assert float(F.cosine_similarity(got,fixture_ref).min())>.995
                for i in range(0,len(ids),batch):
                    fast=e(ids[i:i+batch],mask[i:i+batch])
                    assert torch.equal(fast,actual[i//batch])
                enc=tk(texts[:batch],padding='max_length',truncation=True,max_length=cap,return_tensors='pt')
                ids,mask=enc['input_ids'],enc['attention_mask'].bool()
                cpu=e.model(ids,mask)[2];npu=e(ids,mask)
                assert float(F.cosine_similarity(cpu,npu).min())>.995
                samples={'full_output':[],'final_output':[],'text_to_vector':[]}
                for iteration in range(a.samples+5):
                    # Alternate order to reduce systematic thermal/load bias.
                    for full in ((True,False) if iteration%2==0 else (False,True)):
                        e.full_output=full
                        tick=time.perf_counter(); e(ids,mask); elapsed=(time.perf_counter()-tick)*1000
                        if iteration>=5:samples['full_output' if full else 'final_output'].append(elapsed)
                    e.full_output=False
                    tick=time.perf_counter()
                    enc=tk(texts[:batch],padding='max_length',truncation=True,max_length=cap,return_tensors='pt')
                    e(enc['input_ids'],enc['attention_mask'].bool())
                    if iteration>=5:samples['text_to_vector'].append((time.perf_counter()-tick)*1000)
                medians={k:float(np.median(v)) for k,v in samples.items()}
                row={'cap':cap,'batch':batch,'min_fixture_cosine':float(cosine.min()),
                     'max_fixture_vector_error':float((got-ref).abs().max()),
                     'fixture_sha256':digest(ROOT/f'fixtures/s{cap}.trace.npz'),
                     'min_text_cosine':float(F.cosine_similarity(cpu,npu).min()),
                     'median_ms':medians,'embeddings_per_second':{k:batch*1000/v for k,v in medians.items()},
                     'final_output_speedup':medians['full_output']/medians['final_output'],
                     'samples_ms':samples,'build':e.manifest,'profile':e.profile()}
                result['runs'].append(row)
                print(json.dumps({k:v for k,v in row.items() if k not in ('samples_ms','build','profile')}),flush=True)
                a.output.write_text(json.dumps(result,indent=2)+'\n')
            finally:e.close()


if __name__=='__main__':main()
