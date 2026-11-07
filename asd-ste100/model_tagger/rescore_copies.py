"""Correct copy-fidelity scoring without changing any frozen model prediction."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from evaluate_model import aggregate,SteTokenizer,sha256
from copy_fidelity import preserved


def rescore(source,vocab,output):
    if output.exists() and any(output.iterdir()):raise ValueError('use a new/empty directory')
    output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((source/'summary.json').read_text());tok=SteTokenizer(vocab)
    rows=[];changes=[];lookup={}
    with gzip.open(output/'predictions.jsonl.gz','wt',encoding='utf-8') as stream:
        for row in (json.loads(line) for line in gzip.open(source/'predictions.jsonl.gz','rt')):
            prior=row['detected_slots_preserved']
            if row['output_present']:
                slots=tok.encode_source(row['source'],row['mode'])['slots']
                row['detected_slots_preserved']=preserved(row['prediction']['text'],slots)
            if prior!=row['detected_slots_preserved']:
                row['retokenization_fidelity_flag']=prior
                changes.append({'id':row['id'],'cohort':row['cohort'],'old':prior,'new':row['detected_slots_preserved']})
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');rows.append(row);lookup[(row['cohort'],row['id'])]=row
    summary['groups']={}
    for cohort in ('noisy_source','clean_control'):
        subset=[r for r in rows if r['cohort']==cohort]
        summary['groups'][cohort]=aggregate(subset)
        for kind in sorted({r['kind'] for r in subset}):summary['groups'][cohort+'/'+kind]=aggregate([r for r in subset if r['kind']==kind])
        summary['groups'][cohort+'/training_representable']=aggregate([r for r in subset if r['training_representable']])
    summary['copy_fidelity_method']='Longest exact source-slot surfaces with word boundaries where appropriate; case-sensitive, non-overlapping counts. Raw literal-guard failures are separate.'
    summary['copy_score_corrections']=changes
    summary['hashes']['inference_evaluation']=summary['hashes']['evaluation']
    summary['hashes']['score_evaluation']=sha256(Path(__file__).parent/'evaluate_model.py')
    summary['hashes']['copy_fidelity']=sha256(Path(__file__).parent/'copy_fidelity.py')
    summary['hashes']['copy_rescore']=sha256(Path(__file__))
    summary['parent_evaluation_summary_sha256']=sha256(source/'summary.json')
    summary['postprocess']='Only detected-slot fidelity scores corrected. Predictions/checkpoint/latencies/all other per-case scores are unchanged.'
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for filename in ('isolation.json','batch-agreement.json','latency-cases.json'):(output/filename).write_bytes((source/filename).read_bytes())
    examples=[]
    for line in (source/'examples.jsonl').read_text().splitlines():
        old=json.loads(line);row=dict(lookup[(old['cohort'],old['id'])],sample_group=old['sample_group']);examples.append(row)
    (output/'examples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in examples))
    (output/'examples.txt').write_text('\n\n'.join(f"{r['sample_group']} | {r['kind']} | {r['id']}\nINPUT: {r['source']}\nTARGET: {r['target']}\nOUTPUT: {r['prediction']['text']}\nSTATUS: {r['prediction']['reason']} | exact={r['exact_target_tokens']} | literal_changes={r['literal_changes']}" for r in examples))
    return {'cases':len(rows),'fidelity_corrections':len(changes),'slot_failures':sum(r['detected_slots_preserved'] is False for r in rows)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','vocab','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(rescore(a.source,a.vocab,a.out),indent=2))
