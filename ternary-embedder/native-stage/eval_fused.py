"""Full-corpus paired retrieval check of the actual NPU-produced vectors."""
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from encoder import ROOT,CHECKPOINT,digest,DLL
from fused_encoder import FusedEncoder
from bench_encoder import pack

DIAG=ROOT.parent/'retrieval-diagnostics'
sys.path.insert(0,str(DIAG))
from diagnose import dataset,evaluate,paired,TOKENIZER
from transformers import AutoTokenizer


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    tk=AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True)
    engine=FusedEncoder(128)
    report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'checkpoint_sha256':digest(CHECKPOINT),'max_length':128,'query_prompts':False,
            'build':engine.manifest,'runtime_dll_sha256':digest(DLL),
            'encoder_source_sha256':digest(ROOT/'fused_encoder.py'),
            'scope':'Exact full-corpus ranking of actual one-dispatch NPU vectors',
            'datasets':{}}
    out=ROOT/'results/fused-retrieval'
    out.mkdir(parents=True,exist_ok=True)
    deltas=[]
    try:
        for name in ('SciFact','NFCorpus'):
            d=dataset(name,Path('C:/Users/josha/.cache/huggingface/datasets'))
            texts=d['queries']+d['texts']
            key=hashlib.sha256(json.dumps({'texts':texts,'checkpoint':report['checkpoint_sha256'],
                'build':engine.manifest,'encoder_source':digest(ROOT/'fused_encoder.py'),
                'tokenizer':digest(TOKENIZER/'tokenizer.json')},sort_keys=True).encode()).hexdigest()
            meta=out/f'{name}.json';vpath=out/f'{name}.npy'
            if meta.is_file() and vpath.is_file() and json.loads(meta.read_text())['key']==key:
                vectors=np.load(vpath);timings=json.loads(meta.read_text())['timing']
            else:
                vectors=np.empty((len(texts),384),np.float32)
                engine.profile(reset=True)
                start=time.perf_counter()
                # Batch tokenization includes every original text, even if truncated.
                for offset in range(0,len(texts),100):
                    ids,mask=pack(tk,texts[offset:offset+100],128)
                    for i in range(len(ids)):
                        vectors[offset+i]=engine(ids[i:i+1],mask[i:i+1]).numpy()[0]
                    print(json.dumps({'dataset':name,'encoded':min(offset+100,len(texts)),
                                      'total':len(texts),'seconds':round(time.perf_counter()-start,2)}),flush=True)
                elapsed=time.perf_counter()-start
                timings={'text_to_vector_seconds':elapsed,'embeddings_per_second':len(texts)/elapsed,
                         'profile':engine.profile(reset=True)}
                np.save(vpath,vectors)
                meta.write_text(json.dumps({'key':key,'timing':timings},indent=2)+'\n')
            nq=len(d['queries'])
            metrics,rows=evaluate(vectors[:nq]@vectors[nq:].T,d)
            cpu_path=DIAG/f'cache/embeddings/{name}_qat150_quant.npy'
            cpu_meta=json.loads(cpu_path.with_suffix('.json').read_text())
            expected_key=hashlib.sha256(json.dumps({'identity':cpu_meta['identity'],'texts':texts,
                'max_length':128,'encoder_version':1},sort_keys=True).encode()).hexdigest()
            if cpu_meta['key']!=expected_key or cpu_meta['identity']['sha256']!=report['checkpoint_sha256']:
                raise ValueError('CPU embedding cache does not match checkpoint and texts')
            cpu=np.load(cpu_path)
            cpu_metrics,cpu_rows=evaluate(cpu[:nq]@cpu[nq:].T,d)
            deltas.append(np.array([b['ndcg10']-a['ndcg10'] for a,b in zip(cpu_rows,rows)]))
            cos=np.sum(vectors*cpu,axis=1)/(np.linalg.norm(vectors,axis=1)*np.linalg.norm(cpu,axis=1))
            item={'npu':metrics,'cpu_reference':cpu_metrics,
                  'paired_cpu_to_npu':paired(cpu_rows,rows),
                  'vectors':{'total':len(vectors),'finite':bool(np.isfinite(vectors).all()),
                             'max_abs_error':float(np.max(np.abs(vectors-cpu))),
                             'cosine_min':float(cos.min()),'cosine_mean':float(cos.mean()),
                             'cosine_p01':float(np.percentile(cos,1))},
                  'source_sha256':d['source_sha256'],'timing':timings,
                  'npu_vectors_sha256':digest(vpath),'cpu_vectors_sha256':digest(cpu_path)}
            (out/f'{name}.rows.json').write_text(json.dumps(rows,indent=2)+'\n')
            report['datasets'][name]=item
            print(json.dumps({'dataset':name,'result':{k:v for k,v in item.items() if k!='timing'}}),flush=True)
            report['mean_ndcg10']=float(np.mean([r['npu']['ndcg10'] for r in report['datasets'].values()]))
            (ROOT/'fused-retrieval.json').write_text(json.dumps(report,indent=2)+'\n')
        rng=np.random.default_rng(0)
        boot=np.mean([v[rng.integers(0,len(v),size=(4000,len(v)))].mean(1) for v in deltas],axis=0)*100
        report['mean_cpu_to_npu']={'delta_ndcg10':float(np.mean([v.mean() for v in deltas])*100),
            'paired_query_bootstrap_95_ci':np.percentile(boot,[2.5,97.5]).tolist(),
            'scope':'Equal task average; paired query sampling, not training-seed uncertainty'}
        (ROOT/'fused-retrieval.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        engine.close()


if __name__=='__main__':
    main()
