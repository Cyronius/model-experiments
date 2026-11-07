"""Frozen-checkpoint test evaluation: actual decoding, clean-input controls and baselines."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
from tagger import (Config,Tagger,Tags,SteTokenizer,Lexicon,EncodingError,collate,source_units,
                    rewrite,check,errors,split_sentences)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tokenizer'))
from prepare import input_paths,records
from ste_tokenizer import model_tokens,sha256
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data'))
from quality_gates import unsafe_changes
from audit_splits import key as canonical_key
from packed import unpack
from copy_fidelity import preserved as copy_preserved


def words(text):return [t.text for t in model_tokens(text)]


def distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]


@torch.no_grad()
def rewrite_batch(model,tok,lex,inputs,max_passes=5):
    model.eval();tags=Tags(tok);result=[None]*len(inputs);active=[]
    for i,item in enumerate(inputs):
        try:
            encoded=tok.encode_source(item['text'],item['mode'])
            original=Counter({s['surface']:s['count'] for s in encoded['slots']})
            active.append({'i':i,'text':item['text'],'mode':item['mode'],'original':original,'seen':{item['text']}})
        except EncodingError as e:result[i]={'text':None,'passes':0,'resolved':False,'reason':str(e)}
    for step in range(max_passes):
        ready=[];encoded=[]
        for state in active:
            try:
                row=dict(tok.encode_source(state['text'],state['mode']),src=state['text'],mode=state['mode'])
                ready.append(state);encoded.append(row)
            except EncodingError as e:result[state['i']]={'text':None,'passes':step,'resolved':False,'reason':str(e)}
        if not ready:break
        batch=collate(encoded,tok,lex);scores,case_scores=model(batch)
        chosen=scores.argmax(-1).tolist();cases=case_scores.argmax(-1).tolist();active=[]
        for j,(state,row) in enumerate(zip(ready,encoded)):
            count=len(row['source_spans'])+1;actions=chosen[j][:count];casing=cases[j][:count]
            try:
                output=tags.apply(source_units(row),actions,casing,row['slots'])
                values,case=map(list,zip(*output));tokens=tok.restore_tokens(values,row['slots'],case)
                if Counter(w for w in tokens if w in state['original'])!=state['original']:raise EncodingError('original_slot_fidelity')
                updated=tok.detokenize(values,row['slots'],case)
            except EncodingError as e:
                result[state['i']]={'text':None,'passes':step+1,'resolved':False,'reason':str(e)};continue
            clean=not errors(check(updated,lex,state['mode']))
            if clean:result[state['i']]={'text':updated,'passes':step+1,'resolved':True,'reason':'checker_clean'}
            elif all(t==tags.KEEP for t in actions) or updated in state['seen']:
                result[state['i']]={'text':updated,'passes':step+1,'resolved':False,'reason':'stalled'}
            elif step+1==max_passes:result[state['i']]={'text':updated,'passes':max_passes,'resolved':False,'reason':'pass_cap'}
            else:state['seen'].add(updated);state['text']=updated;active.append(state)
    assert all(r is not None for r in result)
    return result


def heldout_rows(pairs):
    paths=list(input_paths(pairs));result=[]
    for path,_ in paths:
        if not path.name.startswith('test-'):continue
        for row in records(path):
            if row['split']!='test':raise ValueError('test shard contains another split')
            result.append({k:row[k] for k in ('id','group','src','ste','mode','kind','noise','meaning_review')})
    return result


def verify_holdout(rows,rollouts,tok):
    keys=defaultdict(set)
    for row in rows:
        for text in (row['src'],row['ste']):keys[canonical_key(text)].add(row['group'])
    bad=set();accepted_test=set();scanned=0
    tags=Tags(tok)
    data=np.memmap(rollouts/'train.data.i32',dtype='<i4',mode='r')
    index=np.load(rollouts/'train.index.npy',mmap_mode='r')
    for row in records(rollouts/'pairs.jsonl.gz'):
        if row['split']=='test':accepted_test.add(row['id'])
        if row['split']!='train':continue
        texts=[row['src'],row['ste']]
        if row['edit_passes']>1:
            current=row['src']
            for i in range(row['edit_passes']-1):
                encoded=tok.encode_pair({'src':current,'ste':row['ste'],'mode':row['mode']},.7)
                packed=unpack(data,index[row['first_example']+i])
                out=tags.apply(source_units(encoded),packed['labels'].tolist(),packed['label_cases'].tolist(),encoded['slots'])
                values,cases=map(list,zip(*out));current=tok.detokenize(values,encoded['slots'],cases);texts.append(current)
        for text in texts:bad.update(keys.get(canonical_key(text),()))
        scanned+=1
        if scanned%100000==0:print(f'checked heldout isolation against {scanned:,} training pairs',flush=True)
    return bad,accepted_test


def score(row,source,prediction,tok,lex,clean):
    target=words(row['ste']);before=words(source);text=prediction['text'];after=words(text) if text is not None else None
    before_errors=errors(check(source,lex,row['mode']));after_errors=errors(check(text,lex,row['mode'])) if text is not None else []
    detected=True;literal=[]
    if text is not None:
        encoded=tok.encode_source(source,row['mode']);counts=Counter({s['surface']:s['count'] for s in encoded['slots']})
        detected=copy_preserved(text,encoded['slots'])
        literal=unsafe_changes(source,text)
    base=distance(before,target);remaining=distance(after,target) if after is not None else None
    return {'id':row['id'],'kind':row['kind'],'noise':row['noise'],'mode':row['mode'],'group':row['group'],
            'cohort':'clean_control' if clean else 'noisy_source','source':source,'target':row['ste'],'prediction':prediction,
            'identity_exact_target':before==target,'exact_target_tokens':after==target if after is not None else False,
            'output_present':after is not None,'changed_tokens':after!=before if after is not None else False,
            'input_target_distance':base,'output_target_distance':remaining,
            'distance_improved':remaining is not None and remaining<base,
            'distance_worsened':remaining is not None and remaining>base,
            'input_token_count':len(before),'output_token_count':len(after) if after is not None else None,
            'output_edit_distance':distance(before,after) if after is not None else None,
            'input_checker_errors':len(before_errors),'output_checker_errors':len(after_errors) if after is not None else None,
            'output_checker_clean':not after_errors if after is not None else False,
            'input_sentences':len(split_sentences(source)),'output_sentences':len(split_sentences(text)) if text is not None else 0,
            'detected_slots_preserved':detected if after is not None else None,'literal_changes':literal}


def aggregate(rows):
    n=len(rows);provided=[r for r in rows if r['output_present']]
    reasons=Counter(r['prediction']['reason'] for r in rows)
    result={'cases':n,'outputs':len(provided),'no_output':n-len(provided),'exact_target':sum(r['exact_target_tokens'] for r in rows),
            'identity_exact_target':sum(r['identity_exact_target'] for r in rows),'changed_outputs':sum(r['changed_tokens'] for r in rows),
            'distance_improved':sum(r['distance_improved'] for r in rows),'distance_worsened':sum(r['distance_worsened'] for r in rows),
            'checker_clean_outputs':sum(r['output_checker_clean'] for r in rows),'resolved':sum(r['prediction']['resolved'] for r in rows),
            'detected_slot_failures_in_returned_outputs':sum(r['detected_slots_preserved'] is False for r in rows),
            'literal_changed_outputs':sum(bool(r['literal_changes']) for r in rows),'reasons':dict(reasons),
            'exact_target_rate':sum(r['exact_target_tokens'] for r in rows)/max(1,n),
            'clean_input_unchanged_rate':sum(not r['changed_tokens'] and r['output_present'] for r in rows)/max(1,n),
            'surface_token_edit_rate_on_returned_outputs':sum(r['output_edit_distance'] for r in provided)/max(1,sum(r['input_token_count'] for r in provided)),
            'input_errors_per_100_sentences':100*sum(r['input_checker_errors'] for r in rows)/max(1,sum(r['input_sentences'] for r in rows)),
            'output_errors_per_100_sentences':100*sum(r['output_checker_errors'] for r in provided)/max(1,sum(r['output_sentences'] for r in provided)),
            'passes_mean':sum(r['prediction']['passes'] for r in rows)/max(1,n),
            'passes_p95':float(np.percentile([r['prediction']['passes'] for r in rows],95)) if rows else None}
    return result


def sample_add(heaps,group,row,limit):
    priority=int(hashlib.sha256((group+'\0'+row['id']).encode()).hexdigest(),16)
    heap=heaps.setdefault(group,[]);entry=(-priority,row['id'],row)
    if len(heap)<limit:heapq.heappush(heap,entry)
    elif entry[:2]>heap[0][:2]:heapq.heapreplace(heap,entry)


def run(args):
    if args.out.exists() and any(args.out.iterdir()):raise ValueError('use new/empty evaluation output')
    args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(args.threads)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    tok=SteTokenizer(args.vocab);lex=Lexicon()
    if sha256(args.vocab/'vocab.json')!=state['vocab_sha256']:raise ValueError('checkpoint vocabulary differs')
    if state['code_sha256']['tagger.py']!=sha256(Path(__file__).parent/'tagger.py'):raise ValueError('runtime differs from checkpoint')
    model=Tagger(tok,Config(**state['model_config']));model.load_state_dict(state['model']);model.eval()
    rows=heldout_rows(args.pairs)
    bad,accepted=verify_holdout(rows,args.rollouts,tok)
    kept=[r for r in rows if r['group'] not in bad]
    isolation={'parent_test_pairs':len(rows),'excluded_overlap_groups':sorted(bad),'eligible_test_pairs':len(kept),
               'training_representable_test_pairs':sum(r['id'] in accepted for r in kept),'fully_frozen_checkpoint':True}
    (args.out/'isolation.json').write_text(json.dumps(isolation,indent=2)+'\n')
    # Reference agreement checks the batch decoder against the frozen production
    # loop before using batching for the whole holdout.
    probes=sorted(kept,key=lambda r:hashlib.sha256(r['id'].encode()).digest())[:24]
    expected=[]
    for row in probes:
        try:expected.append(rewrite(model,tok,lex,row['src'],row['mode']))
        except EncodingError as e:expected.append({'text':None,'passes':0,'resolved':False,'reason':str(e)})
    actual=rewrite_batch(model,tok,lex,[{'text':r['src'],'mode':r['mode']} for r in probes])
    agreement=sum(a==b for a,b in zip(expected,actual))
    (args.out/'batch-agreement.json').write_text(json.dumps({'cases':len(probes),'exact_agreement':agreement,'differences':[{'id':r['id'],'single':a,'batch':b} for r,a,b in zip(probes,expected,actual) if a!=b]},indent=2)+'\n')
    if agreement!=len(probes):raise RuntimeError('batch decoder differs from single-request decoder; inspect before scoring')
    results=[];heaps={};start=time.monotonic()
    with gzip.open(args.out/'predictions.jsonl.gz','wt',encoding='utf-8') as output:
        for cohort in ('noisy_source','clean_control'):
            for offset in range(0,len(kept),args.batch_size):
                chunk=kept[offset:offset+args.batch_size]
                texts=[r['src'] if cohort=='noisy_source' else r['ste'] for r in chunk]
                predictions=rewrite_batch(model,tok,lex,[{'text':t,'mode':r['mode']} for t,r in zip(texts,chunk)])
                for row,source,prediction in zip(chunk,texts,predictions):
                    item=score(row,source,prediction,tok,lex,cohort=='clean_control')
                    item['training_representable']=row['id'] in accepted
                    output.write(json.dumps(item,ensure_ascii=False)+'\n');results.append(item)
                    if cohort=='noisy_source':
                        sample_add(heaps,'uniform_noisy',item,20);sample_add(heaps,'kind:'+row['kind'],item,12)
                        if item['literal_changes']:sample_add(heaps,'literal_failures',item,12)
                        if item['exact_target_tokens']:sample_add(heaps,'exact_repairs',item,12)
                    else:sample_add(heaps,'clean_controls',item,24)
                done=min(offset+args.batch_size,len(kept))
                if done%1024==0 or done==len(kept):print(f'{cohort}: {done:,}/{len(kept):,}; elapsed {time.monotonic()-start:.1f}s',flush=True)
    summary={'scope':'Frozen FP32 PyTorch checkpoint; all eligible clean-corpus test pairs, plus their clean targets as controls.',
             'isolation':isolation,'checkpoint_step':state['step'],'checkpoint_bytes':args.checkpoint.stat().st_size,
             'threads':args.threads,'batch_size':args.batch_size,'batch_agreement_cases':agreement,'decode_seconds':time.monotonic()-start,
             'groups':{},'hashes':{'checkpoint':sha256(args.checkpoint),'vocab':sha256(args.vocab/'vocab.json'),
                                  'pairs_summary':sha256(args.pairs/'summary.json'),'rollouts_summary':sha256(args.rollouts/'summary.json'),
                                  'runtime':sha256(Path(__file__).parent/'tagger.py'),'evaluation':sha256(Path(__file__))},
             'limits':['Generated test targets are not owner-verified gold; exact target matching does not prove meaning preservation.',
                       'Checker-clean is the existing partial checker only; it can miss bad grammar/meaning.',
                       'Detected-slot fidelity and additional literal checks are reported separately.',
                       'Token counts use model surface tokens, not a Qwen or tiktoken tokenizer.',
                       'Batched quality throughput is not single-request latency; timings below measure the latter separately.']}
    for cohort in ('noisy_source','clean_control'):
        subset=[r for r in results if r['cohort']==cohort]
        summary['groups'][cohort]=aggregate(subset)
        for kind in sorted({r['kind'] for r in subset}):summary['groups'][cohort+'/'+kind]=aggregate([r for r in subset if r['kind']==kind])
        summary['groups'][cohort+'/training_representable']=aggregate([r for r in subset if r['training_representable']])
    # Four-thread, individual FP32 requests: warm up separately, then time both
    # sentences and short paragraphs. No inference threshold or checkpoint tuning.
    latencies=defaultdict(list);latency_cases=[]
    for row in sorted(kept,key=lambda r:hashlib.sha256(('latency'+r['id']).encode()).digest()):
        sentences=len(split_sentences(row['src']));category='sentence' if sentences==1 else 'paragraph_2_to_6' if 2<=sentences<=6 else None
        if category is not None and len(latencies[category])<100:
            if len(latency_cases)<5:rewrite_batch(model,tok,lex,[{'text':row['src'],'mode':row['mode']}])
            start=time.perf_counter()
            try:prediction=rewrite(model,tok,lex,row['src'],row['mode'])
            except EncodingError:prediction={'text':None,'reason':'encoding_failure'}
            milliseconds=1000*(time.perf_counter()-start);latencies[category].append(milliseconds)
            latency_cases.append({'id':row['id'],'category':category,'milliseconds':milliseconds,'reason':prediction['reason']})
        if len(latencies['sentence'])>=100 and len(latencies['paragraph_2_to_6'])>=100:break
    summary['single_request_latency_ms']={k:{'cases':len(v),'p50':float(np.percentile(v,50)),'p95':float(np.percentile(v,95)),'max':max(v)} for k,v in latencies.items()}
    (args.out/'latency-cases.json').write_text(json.dumps(latency_cases,indent=2)+'\n')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    examples=[dict(row,sample_group=g) for g,heap in sorted(heaps.items()) for _,_,row in sorted(heap,key=lambda x:-x[0])]
    (args.out/'examples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in examples))
    (args.out/'examples.txt').write_text('\n\n'.join(f"{r['sample_group']} | {r['kind']} | {r['id']}\nINPUT: {r['source']}\nTARGET: {r['target']}\nOUTPUT: {r['prediction']['text']}\nSTATUS: {r['prediction']['reason']} | exact={r['exact_target_tokens']} | literal_changes={r['literal_changes']}" for r in examples))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','vocab','pairs','rollouts','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();print(json.dumps(run(a),indent=2))
