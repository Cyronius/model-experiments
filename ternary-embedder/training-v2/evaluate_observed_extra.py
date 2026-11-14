"""Score SHA-frozen candidates on already observed FiQA/SCIDOCS development data."""
import argparse
import json
from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer
from pair_training_state import atomic_json,digest

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'comparisons/qwen06b'))
sys.path.insert(0,str(PROJECT/'retrieval-diagnostics'))
from compare import data
from diagnose import cached_encode,student
from round2_metrics import evaluate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--model',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2)
    entry=json.loads(args.freeze.read_text())['models'][args.model]
    target=args.output/(args.model+'-quality.json')
    if target.exists():
        old=json.loads(target.read_text())
        if set(old['tasks'])=={'FiQA','SCIDOCS'} and old['selection']['sha256']==entry['sha256']:return
    path=Path(entry['path']);assert digest(path)==entry['sha256']
    model=student(path,'quant' if entry['precision']=='ternary' else 'float')
    tokenizer=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
    identity=dict(sha256=entry['sha256'],precision=entry['precision'],cap=128,
                  encoder_sha256=digest(PROJECT/'retrieval-diagnostics/diagnose.py'))
    args.output.mkdir(parents=True,exist_ok=True)
    result=dict(selection=entry,token_cap=128,query_instructions=False,tasks={},
                scope='Previously opened development judgments; no new reserved labels')
    rows={}
    with torch.inference_mode():
        for task in ('FiQA','SCIDOCS'):
            dataset=data(task);n=len(dataset['queries'])
            cache=ROOT/'work/broad-data-evaluation/vectors'/(entry['sha256']+'-'+task+'.npy')
            vectors=cached_encode(model,tokenizer,dataset['queries']+dataset['texts'],False,64,identity,cache)
            metrics,per_query=evaluate(vectors[:n]@vectors[n:].T,dataset)
            result['tasks'][task]=dict(metrics,source_sha256=dataset['source_sha256']);rows[task]=per_query
            atomic_json(target,result);atomic_json(args.output/(args.model+'-rows.json'),rows)
            print(args.model,task,metrics['ndcg10'],flush=True)


if __name__=='__main__':main()
