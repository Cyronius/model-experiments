"""Rerun previously flagged prompts on the selected adapter with a 4096-token cap.

This is an inference-only diagnostic, separate from the frozen healing scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from correctness_heal import (load_source,project,install_lora,load_adapter,
    recipe_hashes,verify_source,load_tokenizer,evaluate,save)


def compare(cases,baseline):
    """Preserve completion and repetition as separate outcomes."""
    results=[]
    for row in cases:
        old=baseline[row['id']]
        before=old['answer'];after=row['answer']
        common=0
        for a,b in zip(before,after):
            if a!=b:break
            common+=1
        results.append({'id':row['id'],'previous_generated_tokens':old['generated_tokens'],
            'new_generated_tokens':row['generated_tokens'],'previous_ended':old['ended'],
            'new_ended':row['ended'],'previous_loop_flag':bool(old.get('combined_loop_flag',old.get('loop_flag',False))),
            'new_loop_flag':row['combined_loop_flag'],
            'previous_answer_is_exact_prefix':after.startswith(before),
            'common_prefix_characters':common,'previous_answer_characters':len(before),
            'finished_after_old_length':bool(not old['ended'] and row['ended'] and row['generated_tokens']>old['generated_tokens'])})
    return results


def run(args):
    data=json.loads(args.data.read_text());request=json.loads(args.request.read_text())
    meta=json.loads((args.adapter/'adapter.json').read_text())
    if request['max_new_tokens']!=4096:raise ValueError('This experiment uses exactly a 4096-token cap')
    if any(c['max_new_tokens']!=4096 for c in request['cases']):raise ValueError('Case token limit differs')
    if meta['data_sha256']!=hashlib.sha256(args.data.read_bytes()).hexdigest():raise ValueError('Adapter data identity differs')
    if meta['source']!=data['source'] or meta['recipe_sha256']!=recipe_hashes(args.recipe):raise ValueError('Source recipe differs')
    if meta['step']!=32:raise ValueError('This diagnostic uses the selected update-32 adapter')
    verify_source(args.source,data['source'])
    if args.out.exists():raise ValueError('Use a fresh diagnostic directory')
    args.out.mkdir(parents=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report={'status':'running','source':data['source'],'selected_step':32,
        'request_sha256':hashlib.sha256(args.request.read_bytes()).hexdigest(),
        'max_new_tokens':4096,'training_performed':False,'changed_setting':'max_new_tokens only'}
    save(report,args.out/'result.json')
    tok=load_tokenizer(args.source);model,_=load_source(args.source,args.device)
    project(model,args.recipe,data['width'])
    info=install_lora(model,data['settings']['rank'],data['settings']['alpha'])
    if info['modules']!=meta['modules']:raise ValueError('Adapter module layout differs')
    load_adapter(model,args.adapter);model.eval()
    if time.time()>=args.deadline:raise TimeoutError('Inference deadline reached during reconstruction')
    generations=[];report['batches']=[]
    for index,batch in enumerate(request['batches']):
        if time.time()>=args.deadline:raise TimeoutError('Inference deadline reached between batches')
        path=args.out/f'batch-{index:02d}.json'
        summary=evaluate(model,tok,batch['cases'],path,batch_size=4)
        generations.extend(json.loads(path.read_text())['cases'])
        report['batches'].append({'original_split':batch['split'],'original_limit':batch['original_limit'],
            'original_batch_index':batch['original_batch_index'],'summary':summary})
        save({'cases':generations},args.out/'generations.json')
        save(report,args.out/'result.json')
    report['comparisons']=compare(generations,request['selected32_baseline'])
    targets=set(request['target_ids'])
    report['target_summary']={'cases':len(targets),
        'ended':sum(r['new_ended'] for r in report['comparisons'] if r['id'] in targets),
        'loop_flags':sum(r['new_loop_flag'] for r in report['comparisons'] if r['id'] in targets),
        'prefix_matches':sum(r['previous_answer_is_exact_prefix'] for r in report['comparisons'] if r['id'] in targets)}
    report['status']='complete';save(report,args.out/'result.json')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['source','recipe','data','request','adapter','out']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--deadline',type=float,required=True)
    run(ap.parse_args())
