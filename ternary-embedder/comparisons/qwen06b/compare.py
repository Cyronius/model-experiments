"""Local, no-charge retrieval and speed comparison: Qwen CPU versus ternary NPU."""
import csv,datetime,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
sys.path.insert(0,str(PROJECT/'native-stage'))
from fused_encoder import FusedEncoder,CHECKPOINT,digest
sys.path.insert(0,str(PROJECT/'retrieval-diagnostics'))
from diagnose import evaluate,paired
MODEL=PROJECT/'cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
PROMPTS={'SciFact':'Given a scientific claim, retrieve documents that support or refute the claim',
         'NFCorpus':'Given a question, retrieve relevant documents that answer the question'}

def data(name):
    p=ROOT/'data'/name.lower()
    corpus=sorted([json.loads(l) for l in (p/'corpus.jsonl').read_text().splitlines()],key=lambda r:r['_id'])
    qrels={}
    for r in csv.DictReader((p/'qrels/test.tsv').open(),delimiter='\t'):
        qrels.setdefault(r['query-id'],{})[r['corpus-id']]=int(r['score'])
    queries=sorted([json.loads(l) for l in (p/'queries.jsonl').read_text().splitlines() if json.loads(l)['_id'] in qrels],key=lambda r:r['_id'])
    return {'name':name,'cids':[r['_id'] for r in corpus], 'texts':[(r.get('title','')+' '+r['text']).strip() for r in corpus],
            'qids':[r['_id'] for r in queries],'queries':[r['text'] for r in queries],'qrels':qrels,
            'source_sha256':{str(f.relative_to(p)):digest(f) for f in [p/'corpus.jsonl',p/'queries.jsonl',p/'qrels/test.tsv']}}

@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    tk=AutoTokenizer.from_pretrained(MODEL,padding_side='left',local_files_only=True)
    model=AutoModel.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval()
    ntk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
    report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'qwen_revision':MODEL.name,
            'qwen_model_sha256':digest(MODEL/'model.safetensors'),'checkpoint_sha256':digest(CHECKPOINT),
            'qwen_device':'CPU','qwen_dtype':'bfloat16','ternary_device':'NPU','threads':2,
            'quality_cap':128,'qwen_dimensions':1024,'query_instructions':PROMPTS,'speed':[],'quality':{}}
    def qencode(texts,cap=128,pad=True):
        batch=tk(texts,padding='max_length' if pad else True,truncation=True,max_length=cap,return_tensors='pt')
        h=model(**batch).last_hidden_state[:,-1].float()
        return F.normalize(h,dim=-1).numpy()
    def save(): (ROOT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    texts=['A ternary embedding model runs on the NPU.','A small encoder produces text vectors.',
           'Neural networks compare the meaning of sentences.','The weather is sunny today.']
    if (ROOT/'result.json').exists():
        report['speed']=json.loads((ROOT/'result.json').read_text())['speed']
    for cap,batch in ([] if report['speed'] else [(32,4),(128,1)]):
        engine=FusedEncoder(cap)
        def nencode():
            inp=ntk(texts[:batch],padding='max_length',truncation=True,max_length=cap,return_tensors='pt')
            return engine(inp['input_ids'],inp['attention_mask'].bool())
        row={'cap':cap,'batch':batch,'timings':{}}
        try:
            for name,fn in [('ternary_npu',nencode),('qwen_cpu',lambda:qencode(texts[:batch],cap))]:
                for _ in range(5):fn()
                samples=[]
                for _ in range(30):
                    t=time.perf_counter();fn();samples.append((time.perf_counter()-t)*1000)
                median=float(np.median(samples));row['timings'][name]={'median_ms':median,'embeddings_per_second':batch*1000/median,'samples_ms':samples}
            report['speed'].append(row);save();print('SPEED',json.dumps(row),flush=True)
        finally:engine.close()
    torch.set_num_threads(8)
    report['quality_threads']=8
    for name in ['SciFact','NFCorpus']:
        d=data(name);nq=len(d['queries']);rows={};summaries={};vectors={}
        engine=FusedEncoder(128)
        try:
            for kind,textlist in [('ternary_npu',d['queries']+d['texts']),('qwen_cpu_docs',d['texts']),
                                  ('qwen_cpu_queries_plain',d['queries']),
                                  ('qwen_cpu_queries_prompted',[f'Instruct: {PROMPTS[name]}\nQuery:{q}' for q in d['queries']])]:
                path=ROOT/f'{name}_{kind}.npy'
                if path.exists():vectors[kind]=np.load(path);continue
                start=time.perf_counter();chunks=[];batch=64 if kind=='ternary_npu' else 16
                for i in range(0,len(textlist),batch):
                    chunk=textlist[i:i+batch]
                    if kind=='ternary_npu':
                        inp=ntk(chunk,padding='max_length',truncation=True,max_length=128,return_tensors='pt')
                        chunks.append(np.concatenate([engine(inp['input_ids'][j:j+1],inp['attention_mask'][j:j+1].bool()).numpy() for j in range(len(chunk))]))
                    else:chunks.append(qencode(chunk,pad=False))
                    if i%(batch*10)==0:print('ENCODE',name,kind,i,len(textlist),round(time.perf_counter()-start,1),flush=True)
                vectors[kind]=np.concatenate(chunks);np.save(path,vectors[kind])
                print('DONE',name,kind,time.perf_counter()-start,flush=True)
        finally:engine.close()
        nv=vectors['ternary_npu']; docs=vectors['qwen_cpu_docs']
        for label,scores in [('ternary_npu',nv[:nq]@nv[nq:].T),
                             ('qwen_plain',vectors['qwen_cpu_queries_plain']@docs.T),
                             ('qwen_prompted',vectors['qwen_cpu_queries_prompted']@docs.T)]:
            summaries[label],rows[label]=evaluate(scores,d)
        report['quality'][name]={'metrics':summaries,'paired_plain':paired(rows['ternary_npu'],rows['qwen_plain']),
                                'paired_prompted':paired(rows['ternary_npu'],rows['qwen_prompted']),
                                'data_sha256':d['source_sha256']}
        (ROOT/f'{name}_rows.json').write_text(json.dumps(rows)+'\n');save()
        print('QUALITY',name,json.dumps(report['quality'][name]),flush=True)
    report['mean_ndcg10']={k:float(np.mean([v['metrics'][k]['ndcg10'] for v in report['quality'].values()])) for k in ['ternary_npu','qwen_plain','qwen_prompted']};save()
if __name__=='__main__':main()
