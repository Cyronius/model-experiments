"""Prepare a 4k-token loop diagnostic; a new paid job needs explicit approval."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tarfile
import time

import run_scale_pod as lifecycle

ROOT=Path(__file__).resolve().parent
HEALING=ROOT/'results/correctness-healing-20261025'
PRIOR=HEALING/'mc-heal-4d3614e4c060'
SELECTED=HEALING/'mc-heal-resume-12789a78a60b'
RESULTS=ROOT/'results/loop-retry-20261025'
JOB_ROOT=ROOT.parent/'target/manifold-loop-retry'
CAP=3.50
MAX_MINUTES=35


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def prepare_request():
    data=json.loads((HEALING/'data.json').read_text())
    legacy=json.loads((PRIOR/'reproduction/legacy-data.json').read_text())['generation_eval']
    splits={'dev':data['dev'],'test':data['test'],'legacy72':legacy}
    evidence={}
    for job,sub in [(PRIOR,'healing'),(SELECTED,'healing-extended')]:
        for path in sorted((job/sub).rglob('*.json')):
            obj=json.loads(path.read_text())
            if not isinstance(obj,dict) or 'cases' not in obj:continue
            for row in obj['cases']:
                if row.get('combined_loop_flag') or row.get('loop_flag') or row.get('repeated_line_flag'):
                    evidence.setdefault(row['id'],[]).append({'file':str(path.relative_to(ROOT)),
                        'file_sha256':sha256(path),'ended':row['ended'],'generated_tokens':row['generated_tokens']})
    if not evidence:raise ValueError('No previously flagged prompts found')
    batches=[];baseline={};included=set()
    for split,cases in splits.items():
        old_path=SELECTED/'healing-extended'/('step-0032/dev.json' if split=='dev' else 'healed-selected-'+split+'.json')
        old={r['id']:r for r in json.loads(old_path.read_text())['cases']}
        for limit in sorted({c['max_new_tokens'] for c in cases}):
            group=[c for c in cases if c['max_new_tokens']==limit]
            for i in range(0,len(group),4):
                subset=group[i:i+4]
                if not any(c['id'] in evidence for c in subset):continue
                copied=[]
                for c in subset:
                    row=copy.deepcopy(c);row['max_new_tokens']=4096
                    row.setdefault('family','legacy_long_form')
                    copied.append(row);baseline[row['id']]=old[row['id']];included.add(row['id'])
                batches.append({'split':split,'original_limit':limit,'original_batch_index':i//4,'cases':copied})
    if not set(evidence)<=included:raise ValueError('Flagged prompt missing from frozen source data')
    return {'selected_step':32,'max_new_tokens':4096,'target_ids':sorted(evidence),'flag_evidence':evidence,
        'batch_size':4,'batches':batches,'cases':[c for b in batches for c in b['cases']],
        'selected32_baseline':baseline,'purpose':'Distinguish truncation from persistent repetition.',
        'control':'Preserve original batch membership, order, prompts, model, greedy decoding, thinking disabled, cache and repetition settings; change only the output limit.',
        'interpretation':'Completion and repetition are separate outcomes. Check exact output prefixes before attributing differences to the longer limit. Companion cases preserve batch/padding behavior; target counts remain separate.'}


def proposal(request):
    return {'resource':'One Secure Cloud H200 SXM, 141 GB VRAM',
        'gpu_usd_per_hour_ceiling':5.29,'temporary_disk_gb':200,
        'disk_usd_per_hour_estimate':lifecycle.DISK_RATE,'expected_minutes':[15,25],
        'maximum_minutes_from_create':MAX_MINUTES,'budget_cap_usd':CAP,
        'maximum_planned_usage_usd':(5.29+lifecycle.DISK_RATE)*MAX_MINUTES/60,
        'target_prompts':len(request['target_ids']),'batch_companions':len(request['cases'])-len(request['target_ids']),
        'max_new_tokens':4096,'scope':'Inference only on selected update-32 75% Qwen27 adapter; rerun all prompts flagged in this healing pilot, preserving original batches.',
        'training_performed':False,'termination':'Independent watchdog; delete on completion, failure or 35 minutes. No persistent volume or automatic replacement.',
        'authorization':'Pending explicit approval of this new bounded paid job; prior training authorization is complete.'}


def prepare():
    audit=json.loads((SELECTED/'completion-audit.json').read_text())
    if not audit['pod_absent_confirmed'] or audit['artifact_failures'] or audit['selected_step']!=32:
        raise ValueError('Selected checkpoint must be verified and previous pod deleted')
    RESULTS.mkdir(parents=True,exist_ok=True);JOB_ROOT.mkdir(parents=True,exist_ok=True)
    request=prepare_request();lifecycle.write_json(request,RESULTS/'loop-retry.json')
    paths={str(p.relative_to(PRIOR/'reproduction')):p for p in (PRIOR/'reproduction').rglob('*')
        if p.is_file() and p.name!='bundle-manifest.json'}
    paths['correctness_worker.py']=paths['scale_worker.py'];paths['scale_worker.py']=ROOT/'loop_retry_worker.py'
    paths['retry_healing_loops.py']=ROOT/'retry_healing_loops.py';paths['loop-retry.json']=RESULTS/'loop-retry.json'
    for p in (SELECTED/'healing-extended/adapter-best').iterdir():
        if p.is_file():paths['adapter/'+p.name]=p
    for folder in [PRIOR,SELECTED]:
        original=json.loads((folder/'artifact-manifest.json').read_text())
        for p in paths.values():
            if p.is_relative_to(folder) and sha256(p)!=original[str(p.relative_to(folder))]:
                raise ValueError('Original artifact differs: '+p.name)
    manifest={'sha256':{name:sha256(p) for name,p in paths.items()}}
    lifecycle.write_json(manifest,JOB_ROOT/'bundle-manifest.json');paths['bundle-manifest.json']=JOB_ROOT/'bundle-manifest.json'
    bundle=JOB_ROOT/'loop-retry-bundle.tar.gz'
    with tarfile.open(bundle,'w:gz',compresslevel=1) as archive:
        for name,p in paths.items():archive.add(p,arcname=name)
    lifecycle.write_json(proposal(request)|{'bundle_sha256':sha256(bundle),'files':manifest},RESULTS/'job-proposal.json')
    return bundle


def execute(bundle,budget):
    if budget!=CAP:raise ValueError('This new job requires its exact $3.50 spending approval')
    request=json.loads((RESULTS/'loop-retry.json').read_text())
    lifecycle.JOB_ROOT=JOB_ROOT;lifecycle.RESULTS=RESULTS;lifecycle.JOB_PREFIX='mc-loop-'
    lifecycle.BUDGET=CAP;lifecycle.MAX_MINUTES=MAX_MINUTES;lifecycle.BUNDLE_UPLOAD_TIMEOUT=180
    lifecycle.proposal=lambda:proposal(request)
    lifecycle.execute(bundle,budget)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--quote',action='store_true')
    ap.add_argument('--execute',action='store_true');ap.add_argument('--approved-budget-usd',type=float)
    args=ap.parse_args()
    if args.quote:lifecycle.write_json({'checked_at':time.time(),'H200_secure_usd_hr':lifecycle.fresh_quote()},RESULTS/'price-quote.json')
    bundle=prepare()
    if args.execute:execute(bundle,args.approved_budget_usd)
    else:print(json.dumps(proposal(json.loads((RESULTS/'loop-retry.json').read_text())),indent=2));print('Prepared',bundle,'— no paid action performed.')
