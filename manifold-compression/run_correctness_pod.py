"""Prepare the correctness healing job; paid launch requires fresh approval.

Default is offline bundle preparation. --quote is a read-only current GPU price
check. --execute is an operator switch, never a substitute for user approval.
Reuses the previously exercised independent watchdog and lifecycle controller.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

import run_scale_pod as lifecycle

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results/correctness-healing-20261025'
JOB_ROOT=ROOT.parent/'target/manifold-healing'
RECIPE=ROOT/'results/scale-probe-20261024/mc-scale-9b5ca68c9da9/qwen27'
FILES=['correctness_heal.py','correctness_eval.py','correctness_summary.py','horizon_data.py',
       'correctness_prepare.py','correctness_verify.py','horizon_fetch.py',
       'scale_probe.py','scale_data.py','projection_models.py','audit_rotation.py',
       'healing_data.py','summarize_scale_probe.py','scale_requirements.txt']


def proposal():
    preparation=json.loads((RESULTS/'preparation.json').read_text())
    return {'resource':'1 NVIDIA H200 SXM, 141 GB VRAM, Secure Cloud on-demand',
        'gpu_usd_per_hour_ceiling':5.29,'temporary_disk_gb':200,
        'disk_usd_per_hour_estimate':200*.10/(30*24),
        'expected_minutes': [60,90],'maximum_minutes_from_create':90,
        'maximum_planned_usage_usd':(5.29+200*.10/(30*24))*1.5,'budget_cap_usd':10.,
        'image':lifecycle.IMAGE,'source':{'repo':'Qwen/Qwen3.5-27B','revision':'fc05daec18b0a78c049392ed2e771dde82bdf654'},
        'scope':'One correctness-focused LoRA pilot on the saved 3824-wide Qwen27 projection; stock/full-width/75%-before/selected-after controls.',
        'limits':{'max_updates':32,'train_rows':preparation['rows'],
                  'assistant_tokens':sum(preparation['assistant_tokens_by_kind'].values()),
                  'input_tokens':preparation['input_tokens'],
                  'development_scored_cases':64,'test_scored_cases':128,
                  'early_stop':'development regression or time reserved for final evaluation'},
        'outputs':'Best/last and intermediate adapters, last optimizer, pinned projection recipe, data provenance, full generations and scores. Full merged weights can be reconstructed; not copied from this pod.',
        'termination':'Independent local watchdog deletes only this newly created pod after completion, failure or 90-minute deadline. No persistent volume. Never extend time to finish copying.',
        'authorization':'pending explicit approval of this bounded paid job',
        'data_sha256':preparation['data_sha256']}


def prepare_bundle():
    JOB_ROOT.mkdir(parents=True,exist_ok=True)
    paths={name:ROOT/name for name in FILES}
    paths['scale_worker.py']=ROOT/'correctness_worker.py'
    paths['data.json']=RESULTS/'data.json'
    paths['preparation.json']=RESULTS/'preparation.json'
    paths['verification-audit.json']=RESULTS/'verification-audit.json'
    paths['legacy-data.json']=ROOT/'results/scale-probe-20261024/data.json'
    for name in ['basis.pt','rms-scales.json','result.json']:paths['recipe/'+name]=RECIPE/name
    manifest={'sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}}
    preparation=json.loads(paths['preparation.json'].read_text())
    verification=json.loads(paths['verification-audit.json'].read_text())
    if preparation['data_sha256']!=manifest['sha256']['data.json'] or verification['data_sha256']!=preparation['data_sha256']:
        raise ValueError('Frozen dataset checksum differs')
    if verification['failed_ids'] or verification['verifier_sha256']!=manifest['sha256']['horizon_data.py']:
        raise ValueError('Current verifier must pass every selected reference example')
    for name,digest in preparation['recipe_sha256'].items():
        if manifest['sha256']['recipe/'+name]!=digest:raise ValueError('Projection recipe differs')
    lifecycle.write_json(manifest,JOB_ROOT/'bundle-manifest.json')
    paths['bundle-manifest.json']=JOB_ROOT/'bundle-manifest.json'
    bundle=JOB_ROOT/'correctness-bundle.tar.gz'
    with tarfile.open(bundle,'w:gz') as archive:
        for name,path in paths.items():archive.add(path,arcname=name)
    value=proposal()|{'bundle_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'files':manifest}
    quote=RESULTS/'price-quote.json'
    if quote.exists():value['read_only_quote']=json.loads(quote.read_text())
    lifecycle.write_json(value,RESULTS/'job-proposal.json')
    return bundle


def execute(bundle,budget):
    lifecycle.JOB_ROOT=JOB_ROOT;lifecycle.RESULTS=RESULTS
    lifecycle.JOB_PREFIX='mc-heal-';lifecycle.proposal=proposal
    lifecycle.execute(bundle,budget)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--quote',action='store_true')
    ap.add_argument('--approved-budget-usd',type=float)
    args=ap.parse_args()
    if args.quote:
        lifecycle.write_json({'checked_at_unix':time.time(),'H200_secure_usd_hr':lifecycle.fresh_quote()},RESULTS/'price-quote.json')
    bundle=prepare_bundle()
    if args.execute:execute(bundle,args.approved_budget_usd)
    else:
        print(json.dumps(proposal(),indent=2))
        print('Prepared',bundle,'— no paid action performed.')


if __name__=='__main__':main()
