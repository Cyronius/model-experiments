"""Original-limit control on the same pod after the 4k inference diagnostic."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
from correctness_heal import (load_source,project,install_lora,load_adapter,
    recipe_hashes,verify_source,load_tokenizer,evaluate,save)
from retry_healing_loops import compare


def run(args):
    data=json.loads(args.data.read_text());request=json.loads(args.request.read_text())
    meta=json.loads((args.adapter/'adapter.json').read_text())
    if meta['step']!=32 or meta['data_sha256']!=hashlib.sha256(args.data.read_bytes()).hexdigest():raise ValueError('Adapter identity differs')
    if meta['source']!=data['source'] or meta['recipe_sha256']!=recipe_hashes(args.recipe):raise ValueError('Recipe differs')
    verify_source(args.source,data['source'])
    if args.out.exists():raise ValueError('Use fresh control directory')
    args.out.mkdir(parents=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tok=load_tokenizer(args.source);model,_=load_source(args.source,args.device)
    project(model,args.recipe,data['width']);info=install_lora(model,data['settings']['rank'],data['settings']['alpha'])
    if info['modules']!=meta['modules']:raise ValueError('Adapter module layout differs')
    load_adapter(model,args.adapter);model.eval()
    generations=[];report={'status':'running','purpose':'Same-pod original-limit control after historical prefixes diverged','batches':[]}
    save(report,args.out/'result.json')
    for index,batch in enumerate(request['batches']):
        if time.time()>=args.deadline:raise TimeoutError('Control deadline reached')
        cases=copy.deepcopy(batch['cases'])
        for case in cases:case['max_new_tokens']=batch['original_limit']
        path=args.out/f'batch-{index:02d}.json'
        summary=evaluate(model,tok,cases,path,batch_size=4)
        generations.extend(json.loads(path.read_text())['cases'])
        report['batches'].append({'batch':index,'limit':batch['original_limit'],'summary':summary})
        save({'cases':generations},args.out/'generations.json');save(report,args.out/'result.json')
    extended=json.loads((args.extended/'generations.json').read_text())['cases']
    report['paired_4096_comparisons']=compare(extended,{c['id']:c for c in generations})
    report['historical_comparisons']=compare(generations,request['selected32_baseline'])
    report['status']='complete';save(report,args.out/'result.json')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['source','recipe','data','request','adapter','extended','out']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--deadline',type=float,required=True)
    run(ap.parse_args())
