"""Conditional, single-candidate confirmation after the fixed development gate."""
import argparse
import concurrent.futures
import csv
import datetime
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import urllib.request

from pair_training_state import atomic_json, digest

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
PROFILE=ROOT/'broad-data'
DEVELOPMENT=PROFILE/'results'
OUT=DEVELOPMENT/'reserved'
TASKS=('webis-touche2020','trec-covid')
REFS={
    'bge_micro':PROJECT/'cache/hub/models--TaylorAI--bge-micro-v2/snapshots/3edf6d7de0faa426b09780416fe61009f26ae589',
    'minilm':PROJECT/'cache/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41'}


def read(path):return json.loads(path.read_text())


def activate():
    gate=read(DEVELOPMENT/'development-gate.json')
    assert gate['passed'] and gate['candidate']
    assert digest(DEVELOPMENT/'development-summary.json')==gate['summary_sha256']
    report=read(DEVELOPMENT/'development-summary.json')
    candidate=report['models'][gate['candidate']]['selection']
    assert report['models'][gate['candidate']]['passes_all_gates']
    assert digest(Path(candidate['path']))==candidate['sha256']
    resource=read(ROOT/'work/broad-data-run/pod.json')
    assert resource['deleted'] and resource['deletion_verified_http_status']==404
    lock=read(PROFILE/'evaluation.lock.json')
    for spec in lock['datasets'].values():
        for record in spec['text_files'].values():assert digest(Path(record['path']))==record['sha256']
    protocol=read(PROFILE/'evaluation-protocol.json')
    OUT.mkdir(parents=True,exist_ok=True)
    receipt=dict(candidate_name=gate['candidate'],candidate=candidate,
        activated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        development_summary_sha256=gate['summary_sha256'],
        protocol_sha256=digest(PROFILE/'evaluation-protocol.json'),
        qrels_lock_sha256=digest(PROFILE/'reserved-qrels.lock.json'),
        excluded_query_ids=protocol['reserved_exclude_query_ids'],
        preview_exposure=protocol['reserved_preview_exposure'],
        no_further_training_or_selection=True,token_cap=128,query_instructions=False)
    path=OUT/'activation.json'
    if path.exists():
        prior=read(path)
        for key in ('candidate','development_summary_sha256','protocol_sha256','qrels_lock_sha256','excluded_query_ids'):
            assert prior[key]==receipt[key]
        receipt=prior
    else:
        assert not list(OUT.glob('*-test.tsv'))
        atomic_json(path,receipt)
    # Full judgments are accessed only AFTER the durable candidate receipt.
    qrels_lock=read(PROFILE/'reserved-qrels.lock.json')
    for task in TASKS:
        spec=qrels_lock['datasets'][task]
        files=[f for f in spec['files'] if f['path']=='test.tsv']
        assert len(files)==1
        target=OUT/(task+'-test.tsv')
        if not target.exists():
            url='https://huggingface.co/datasets/'+spec['repo']+'/resolve/'+spec['revision']+'/test.tsv'
            with urllib.request.urlopen(url,timeout=60) as response:payload=response.read()
            assert len(payload)==files[0]['size']
            temporary=target.with_suffix('.tmp');temporary.write_bytes(payload);temporary.replace(target)
    return receipt


def data(task):
    record=read(PROFILE/'evaluation.lock.json')['datasets'][task]['text_files']
    with Path(record['corpus.jsonl']['path']).open() as source:
        corpus=sorted((json.loads(line) for line in source),key=lambda r:str(r['_id']))
    excluded=set(read(OUT/'activation.json')['excluded_query_ids'].get(task,[]))
    qrels={}
    target=OUT/(task+'-test.tsv')
    with target.open() as source:
        for row in csv.DictReader(source,delimiter='\t'):
            qid=str(row['query-id'])
            if qid not in excluded:qrels.setdefault(qid,{})[str(row['corpus-id'])]=int(row['score'])
    with Path(record['queries.jsonl']['path']).open() as source:
        queries=sorted((json.loads(line) for line in source),key=lambda r:str(r['_id']))
    queries=[r for r in queries if str(r['_id']) in qrels]
    return dict(name=task,cids=[str(r['_id']) for r in corpus],texts=[(r.get('title','')+' '+r['text']).strip() for r in corpus],
        qids=[str(r['_id']) for r in queries],queries=[r['text'] for r in queries],qrels=qrels,
        source_sha256=dict(corpus=record['corpus.jsonl']['sha256'],queries=record['queries.jsonl']['sha256'],qrels=digest(target)),excluded_query_ids=sorted(excluded))


def worker(name):
    import numpy as np
    import torch
    from transformers import AutoModel,AutoTokenizer
    sys.path.insert(0,str(PROJECT/'retrieval-diagnostics'))
    from diagnose import cached_encode,student
    from round2_metrics import evaluate
    from broad_pairs import bert_prefix
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    receipt=read(OUT/'activation.json')
    if name=='candidate':
        selected=receipt['candidate'];path=Path(selected['path']);assert digest(path)==selected['sha256']
        if selected['precision'] in ('q4','q8'):
            from export_pair_precision import load_reference
            model=load_reference(path)
        else:model=student(path,'quant' if selected['precision']=='ternary' else 'float')
        tokenizer=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
        identity=dict(sha256=selected['sha256'],precision=selected['precision'],cap=128)
    else:
        path=REFS[name];model=AutoModel.from_pretrained(path,local_files_only=True).eval()
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        weights=sorted(path.glob('*.safetensors')) or sorted(path.glob('pytorch_model*.bin'));assert weights
        identity=dict(revision=path.name,weights={p.name:digest(p) for p in weights},pooling='masked-mean-unit',cap=128)
    identity['prefix_sha256']=digest(ROOT/'broad_pairs.py')
    result=dict(identity=identity,activation_sha256=digest(OUT/'activation.json'),tasks={})
    rows={}
    with torch.inference_mode():
        for task in TASKS:
            dataset=data(task)
            bounded=[bert_prefix(text) for text in dataset['queries']+dataset['texts']]
            vectors=cached_encode(model,tokenizer,bounded,name!='candidate',64,identity,
                ROOT/'work/broad-data-evaluation/reserved-vectors'/(name+'-'+task+'.npy'))
            n=len(dataset['queries']);metrics,per_query=evaluate(vectors[:n]@vectors[n:].T,dataset)
            result['tasks'][task]=dict(metrics,source_sha256=dataset['source_sha256'],excluded_query_ids=dataset['excluded_query_ids'])
            rows[task]=per_query
            atomic_json(OUT/(name+'-quality.json'),result);atomic_json(OUT/(name+'-rows.json'),rows)
            print(name,task,metrics['ndcg10'],flush=True)


def contrast(left,right,tasks=TASKS):
    import numpy as np
    rng=np.random.default_rng(20261025);per_task={};samples=[];means=[]
    for task in tasks:
        a,b=left[task],right[task]
        assert [r['query_id'] for r in a]==[r['query_id'] for r in b]
        delta=np.array([y['ndcg10']-x['ndcg10'] for x,y in zip(a,b)])
        boot=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1)*100
        means.append(float(delta.mean()*100));samples.append(boot)
        per_task[task]=dict(delta_ndcg10=means[-1],paired_query_bootstrap_95_ci=np.percentile(boot,[2.5,97.5]).tolist())
    return dict(tasks=per_task,mean_delta_ndcg10=statistics.mean(means),
        paired_query_bootstrap_95_ci=np.percentile(np.mean(samples,axis=0),[2.5,97.5]).tolist(),seed=20261025,resamples=5000)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=('candidate',*REFS));args=parser.parse_args()
    if args.worker:worker(args.worker);return
    receipt=activate()
    def launch(name):
        target=OUT/(name+'-quality.json')
        if target.exists() and set(read(target)['tasks'])==set(TASKS):
            assert read(target)['activation_sha256']==digest(OUT/'activation.json');return
        with (OUT/(name+'.log')).open('a') as output:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',name],check=True,stdout=output,stderr=subprocess.STDOUT)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:list(executor.map(launch,('candidate',*REFS)))
    models={name:read(OUT/(name+'-quality.json')) for name in ('candidate',*REFS)}
    for model in models.values():
        assert model['activation_sha256']==digest(OUT/'activation.json')
        model['mean_retrieval']=statistics.mean(v['ndcg10'] for v in model['tasks'].values())
    for task in TASKS:assert all(m['tasks'][task]['source_sha256']==models['candidate']['tasks'][task]['source_sha256'] for m in models.values())
    candidate_rows=read(OUT/'candidate-rows.json')
    result=dict(scope='One frozen candidate on reserved query subsets; Touche query1 excluded due recorded pre-score preview. Not full MTEB.',
        activation=receipt,models=models,candidate_minus_references={name:contrast(read(OUT/(name+'-rows.json')),candidate_rows) for name in REFS},
        beats_each_task={name:all(models['candidate']['tasks'][t]['ndcg10']>models[name]['tasks'][t]['ndcg10'] for t in TASKS) for name in REFS},
        no_further_training_or_selection=True,default_deployment_changed=False)
    atomic_json(DEVELOPMENT/'reserved-confirmation.json',result)


if __name__=='__main__':main()
