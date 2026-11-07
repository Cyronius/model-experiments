"""Export and audit five-pass teacher-forced tag trajectories and compact training data."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import torch
from tagger import (Tags, SteTokenizer, Lexicon, EncodingError, source_units, target_units,
                    supervision, check, errors)
from packed import pack, unpack, KINDS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tokenizer'))
from prepare import input_paths, records
from ste_tokenizer import sha256, model_tokens
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data'))
from quality_gates import unsafe_changes

TOK=LEX=TAGS=None


def initialize(vocab):
    global TOK,LEX,TAGS
    torch.set_num_threads(1)
    TOK,LEX=SteTokenizer(Path(vocab)),Lexicon()
    TAGS=Tags(TOK)


def original_fidelity(words, slots):
    expected=Counter({s['surface']:s['count'] for s in slots})
    if Counter(w for w in words if w in expected)!=expected:
        raise EncodingError('original_slot_fidelity')


def trajectory(row,max_passes=5):
    current=dict(row);states=[]
    original_slots=row['slots']
    for step in range(max_passes+1):
        units,target=source_units(current),target_units(current,TOK)
        if source_units(TOK.encode_source(current['src'],row['mode'],1.0))!=units:
            raise EncodingError('slotted_unit_mismatch')
        if units==target:
            labels,casing=[TAGS.KEEP]*(len(units)+1),[0]*(len(units)+1)
            TAGS.apply(units,labels,casing,current['slots'])
            states.append((current,labels,casing,True))
            return states
        if step==max_passes:raise EncodingError('pass_cap')
        labels,casing=supervision(units,target,TAGS)
        output=TAGS.apply(units,labels,casing,current['slots'])
        values,cases=map(list,zip(*output))
        words=TOK.restore_tokens(values,current['slots'],cases)
        original_fidelity(words,original_slots)
        text=TOK.detokenize(values,current['slots'],cases)
        if [t.text for t in model_tokens(text)]!=words:
            raise EncodingError('detokenized_span_change')
        if unsafe_changes(current['src'],text):raise EncodingError('intermediate_literal_change')
        if output==units:raise EncodingError('stalled')
        states.append((current,labels,casing,False))
        if output!=target and not errors(check(text,LEX,row['mode'])):
            raise EncodingError('checker_exit_before_target')
        encoded=TOK.encode_pair({'src':text,'ste':row['ste'],'mode':row['mode']},.7)
        current=dict(encoded,src=text,ste=row['ste'],mode=row['mode'],kind=row['kind'])
    raise AssertionError('unreachable')


def process(row):
    try:
        states=trajectory(row)
        packed=[pack(state,labels,cases,terminal,TAGS,LEX) for state,labels,cases,terminal in states]
        return row,packed,None
    except (EncodingError,ValueError) as error:
        return row,None,str(error)


def fingerprints():
    directory=Path(__file__).parent
    return {name:sha256(directory/name) for name in ('tagger.py','packed.py','rollouts.py','../tokenizer/ste_tokenizer.py','../tokenizer/protected_regions.py','../data/quality_gates.py','../checker/checker.py','../data/text_utils.py','../data/build_pairs.py','../lexicon/lexicon.jsonl','../lexicon/technical.jsonl','../lexicon/tech_words.txt')}


def export(pairs,vocab,output,workers=8):
    if output.exists() and any(output.iterdir()):raise ValueError('use an empty output directory')
    paths=list(input_paths(pairs));source=json.loads((pairs/'summary.json').read_text())
    if source.get('vocab_sha256')!=sha256(vocab/'vocab.json'):raise ValueError('parent vocabulary differs')
    codes=fingerprints();output.mkdir(parents=True,exist_ok=True)
    streams={s:(output/f'{s}.data.i32').open('wb') for s in ('train','dev','test')}
    indices={s:[] for s in streams};offsets=Counter();scanned=accepted=0
    counts=Counter();passes=Counter();reasons=Counter();slices=Counter()
    with gzip.open(output/'pairs.jsonl.gz','wt',encoding='utf-8') as meta,(output/'rejected.jsonl').open('w') as rejected:
        def inputs():
            for path,_ in paths:yield from records(path)
        iterator=iter(inputs())
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(str(vocab),)) as pool:
            while True:
                wave=[]
                for _ in range(workers*64):
                    row=next(iterator,None)
                    if row is None:break
                    wave.append(row)
                if not wave:break
                for row,packed,reason in pool.map(process,wave,chunksize=16):
                    scanned+=1
                    if reason:
                        rejected.write(json.dumps({'id':row['id'],'kind':row['kind'],'split':row['split'],'reason':reason})+'\n')
                        reasons[reason]+=1;slices[(row['kind'],'rejected')]+=1
                    else:
                        accepted+=1;split=row['split'];start=len(indices[split])
                        for array,header in packed:
                            indices[split].append((offsets[split],*header));array.tofile(streams[split]);offsets[split]+=len(array)
                        entry={k:row[k] for k in ('id','origin_id','group','split','kind','noise','meaning_review','src','ste','mode')}
                        for k in ('target_review','provenance','document_ids','joint_members'):
                            if k in row:entry[k]=row[k]
                        entry.update(first_example=start,examples=len(packed),edit_passes=len(packed)-1)
                        meta.write(json.dumps(entry,ensure_ascii=False)+'\n')
                        counts[split]+=1;passes[str(len(packed)-1)]+=1;slices[(row['kind'],'accepted')]+=1
                    if scanned%25000==0:print(f'audited {scanned:,}; accepted {accepted:,}',flush=True)
    for s,stream in streams.items():
        stream.close();np.save(output/f'{s}.index.npy',np.asarray(indices[s],dtype='<i8').reshape(-1,6))
    if scanned!=source['pairs'] or fingerprints()!=codes:raise RuntimeError('input accounting or code changed during export')
    result={'schema':'ste-tagger-rollouts-v1','scanned':scanned,'accepted_pairs':accepted,'rejected_pairs':scanned-accepted,
            'pair_splits':dict(counts),'example_splits':{s:len(v) for s,v in indices.items()},
            'edit_passes':dict(passes),'reasons':dict(reasons),'slices':{f'{k}/{v}':n for (k,v),n in slices.items()},
            'max_edit_passes':5,'slot_probability_all_states':.7,'terminal_keep_examples':True,'checker_stop_after_first_edit':True,
            'parent_summary_sha256':sha256(pairs/'summary.json'),'vocab_sha256':sha256(vocab/'vocab.json'),
            'code_sha256':codes,'workers':workers,
            'files':{p.name:sha256(p) for p in output.iterdir() if p.is_file()},
            'limits':['Teacher-forced representability and exact protected-slot checks are not meaning verification.',
                      'Failed trajectories are excluded as whole pairs; no partial prefix enters training.',
                      'Original pair splits/groups and review labels are inherited unchanged.']}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def verify_one(args):
    row,entry,states=args
    if row['id']!=entry['id'] or any(entry[k]!=row[k] for k in ('origin_id','group','split','kind','noise','meaning_review','src','ste','mode')):
        raise ValueError('pair lineage mismatch')
    for key in ('target_review','provenance','document_ids','joint_members'):
        if entry.get(key)!=row.get(key):raise ValueError('attribution/review changed')
    current=dict(row)
    for index,stored in enumerate(states):
        labels=stored['labels'].tolist();cases=stored['label_cases'].tolist()
        terminal=index==len(states)-1
        expected,header=pack(current,labels,cases,terminal,TAGS,LEX)
        # Recomputed source, span, case, slot-reference, checker and transform features
        # must equal every stored field, not just decode to similar prose.
        unpacked=unpack(expected,(0,*header))
        for name in ('ids','cases','refs','starts','ends','unit_ids','unit_cases','feature_bits','transform_bits'):
            if not np.array_equal(stored[name],unpacked[name]):raise ValueError('packed feature mismatch: '+name)
        if stored['mode']!=row['mode_id'] or stored['kind']!=KINDS.index(row['kind']) or bool(stored['terminal'])!=terminal:
            raise ValueError('packed conditioning mismatch')
        units=source_units(current)
        if source_units(TOK.encode_source(current['src'],row['mode'],1.0))!=units:raise ValueError('inference/training units differ')
        output=TAGS.apply(units,labels,cases,current['slots'])
        target=target_units(current,TOK)
        if terminal:
            if not all(t==TAGS.KEEP for t in labels) or output!=target:raise ValueError('terminal target mismatch')
        else:
            if units==target or len(states)-1>5:raise ValueError('invalid rollout length')
            values,casing=map(list,zip(*output));words=TOK.restore_tokens(values,current['slots'],casing)
            original_fidelity(words,row['slots'])
            text=TOK.detokenize(values,current['slots'],casing)
            if [t.text for t in model_tokens(text)]!=words:raise ValueError('roundtrip mismatch')
            if unsafe_changes(current['src'],text):raise ValueError('intermediate literal change')
            if output!=target and not errors(check(text,LEX,row['mode'])):raise ValueError('early checker exit')
            current=dict(TOK.encode_pair({'src':text,'ste':row['ste'],'mode':row['mode']},.7),src=text,ste=row['ste'],mode=row['mode'],kind=row['kind'])
    return row['split'],row['kind'],len(states)-1


def validate(pairs,vocab,output,workers=8):
    summary=json.loads((output/'summary.json').read_text())
    if fingerprints()!=summary['code_sha256'] or sha256(pairs/'summary.json')!=summary['parent_summary_sha256'] or sha256(vocab/'vocab.json')!=summary['vocab_sha256']:
        raise ValueError('audit fingerprint mismatch')
    for name,expected in summary['files'].items():
        if sha256(output/name)!=expected:raise ValueError('packed file hash mismatch: '+name)
    quarantined=set(summary.get('quarantine',{}).get('groups',[]))
    data={s:np.memmap(output/f'{s}.data.i32',mode='r',dtype='<i4') for s in ('train','dev','test')}
    index={s:np.load(output/f'{s}.index.npy',mmap_mode='r') for s in data}
    meta=iter(records(output/'pairs.jsonl.gz'));rejections=iter(records(output/'rejected.jsonl'))
    entry=next(meta,None);rejected=next(rejections,None);counts=Counter();reasons=Counter();passes=Counter();expected_index=Counter();scanned=0
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(str(vocab),)) as pool:
        wave=[]
        def flush():
            for split,kind,n in pool.map(verify_one,wave,chunksize=16):counts[split]+=1;passes[str(n)]+=1
            wave.clear()
        for path,_ in input_paths(pairs):
            for row in records(path):
                scanned+=1
                if entry and row['id']==entry['id']:
                    split=row['split'];first=entry['first_example'];number=entry['examples']
                    if first!=expected_index[split] or number!=entry['edit_passes']+1:raise ValueError('example index coverage mismatch')
                    states=[{k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in unpack(data[split],index[split][i]).items()} for i in range(first,first+number)]
                    expected_index[split]+=number;wave.append((row,entry,states));entry=next(meta,None)
                    if len(wave)>=workers*64:flush()
                elif rejected and row['id']==rejected['id']:
                    # Rejections are replayed too, so a bad export cannot silently omit pairs.
                    if rejected['reason']=='canonical_split_overlap':
                        if row['group'] not in quarantined:raise ValueError('quarantine group differs')
                        reason='canonical_split_overlap'
                    else:
                        reason=pool.submit(process,row).result()[2]
                    if reason!=rejected['reason']:raise ValueError('rejection reason differs')
                    reasons[reason]+=1;rejected=next(rejections,None)
                else:raise ValueError('missing pair or changed ordering')
                if scanned%50000==0:print(f'verified {scanned:,} parent pairs',flush=True)
        if wave:flush()
    if entry or rejected or dict(counts)!=summary['pair_splits'] or dict(reasons)!=summary['reasons'] or dict(passes)!=summary['edit_passes']:
        raise ValueError('rollout audit accounting mismatch')
    for split in data:
        if expected_index[split]!=len(index[split]):raise ValueError('unaccounted examples')
    result={'passed':True,'parent_pairs':scanned,'accepted_pairs':sum(counts.values()),'rejected_pairs':sum(reasons.values()),
            'examples':sum(expected_index.values()),'edit_passes':dict(passes),'all_intermediate_slots_exact':True,
            'all_terminal_targets_exact':True,'parent_splits_and_metadata_preserved':True,
            'summary_sha256':sha256(output/'summary.json'),'validation_code_sha256':sha256(Path(__file__))}
    (output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('export','validate'))
    parser.add_argument('--pairs',type=Path,required=True);parser.add_argument('--vocab',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    print(json.dumps((export if args.command=='export' else validate)(args.pairs,args.vocab,args.out,args.workers),indent=2))
