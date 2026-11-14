"""Prepare a replacement pod continuation. Paid execution needs explicit approval.

The old pod is deleted. This proposal supersedes the now-unavailable in-place
extension, and caps the combined first run plus continuation at $15.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

import run_scale_pod as lifecycle
from run_correctness_pod import ROOT, RESULTS

PRIOR=RESULTS/'mc-heal-4d3614e4c060'
JOB_ROOT=ROOT.parent/'target/manifold-healing-resume'
TOTAL_CAP=15.
PRIOR_RESERVE=8.
NEW_CAP=TOTAL_CAP-PRIOR_RESERVE
MAX_MINUTES=75


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def proposal():
    return {'resource':'One replacement Secure Cloud H200 SXM, 141 GB VRAM',
        'gpu_usd_per_hour_ceiling':5.29,'temporary_disk_gb':200,
        'disk_usd_per_hour_estimate':lifecycle.DISK_RATE,
        'expected_minutes':[50,65],'maximum_minutes_from_create':MAX_MINUTES,
        'maximum_planned_usage_usd':(5.29+lifecycle.DISK_RATE)*MAX_MINUTES/60,
        'budget_cap_usd':NEW_CAP,'total_budget_cap_usd':TOTAL_CAP,
        'prior_run_cost_reserve_usd':PRIOR_RESERVE,
        'expected_combined_cost_usd':[12.06,13.39],
        'prior_job':PRIOR.name,'prior_pod_deleted':True,
        'scope':'Resume the frozen 75%-retained Qwen27 pilot from update 16 to at most 32; reuse completed controls, preserve development selection and regression gates. No new data or objective.',
        'termination':'Independent watchdog; delete on completion, failure or 75 minutes from create intent. No persistent volume, no automatic replacement.',
        'authorization':'pending explicit approval of this replacement pod and $15 combined cap'}


def validate_prior():
    audit=json.loads((PRIOR/'completion-audit.json').read_text())
    report=json.loads((PRIOR/'healing/result.json').read_text())
    life=json.loads((PRIOR/'job-lifecycle.json').read_text())
    if not life.get('terminated') or not audit.get('pod_absent_confirmed'):
        raise ValueError('Previous pod must be deleted before replacement')
    if audit['artifact_failures'] or not report.get('adapter_reload_exact'):
        raise ValueError('Prior checkpoint verification failed')
    billing=json.loads((PRIOR/'billing-final.json').read_text())['records']
    posted=sum(float(r['amount']) for r in billing)
    if max(audit['estimated_cost_usd'],posted)>PRIOR_RESERVE:
        raise ValueError('Prior spending exceeds its reserve; reprice before approval')
    if report['status']!='complete' or report.get('stop_reason')=='development regression gate':
        raise ValueError('Cannot bypass incomplete run or regression gate')
    if len(report['steps'])!=16:raise ValueError('This proposal resumes exactly update 16')
    return life


def bundle_paths():
    validate_prior()
    paths={str(p.relative_to(PRIOR/'reproduction')):p for p in (PRIOR/'reproduction').rglob('*')
           if p.is_file() and p.name!='bundle-manifest.json'}
    paths['correctness_worker.py']=paths['scale_worker.py']
    paths['scale_worker.py']=ROOT/'correctness_restart_worker.py'
    for name in ['resume_correctness.py','correctness_resume_worker.py']:paths[name]=ROOT/name
    for name in ['result.json','optimizer-last.pt']:
        paths['prior/'+name]=PRIOR/'healing'/name
    for label in ['stock','full_width','unhealed_75']:
        for suffix in ['dev.json','test.json','legacy72.json']:
            name=label+'-'+suffix;paths['prior/'+name]=PRIOR/'healing'/name
    for label in ['adapter-best','adapter-last']:
        for p in (PRIOR/'healing'/label).iterdir():
            if p.is_file():paths['prior/'+label+'/'+p.name]=p
    original=json.loads((PRIOR/'artifact-manifest.json').read_text())
    # Verify every reused original input, including the optimizer and adapters.
    for p in paths.values():
        if p.is_relative_to(PRIOR):
            if sha256(p)!=original[str(p.relative_to(PRIOR))]:raise ValueError('Prior artifact changed: '+p.name)
    return paths


def prepare():
    paths=bundle_paths();JOB_ROOT.mkdir(parents=True,exist_ok=True)
    manifest={'sha256':{name:sha256(p) for name,p in paths.items()}}
    lifecycle.write_json(manifest,JOB_ROOT/'bundle-manifest.json')
    paths['bundle-manifest.json']=JOB_ROOT/'bundle-manifest.json'
    bundle=JOB_ROOT/'continuation-bundle.tar.gz'
    with tarfile.open(bundle,'w:gz',compresslevel=1) as archive:
        for name,p in paths.items():archive.add(p,arcname=name)
    lifecycle.write_json(proposal()|{'bundle_sha256':sha256(bundle),'files':manifest},RESULTS/'restart-proposal.json')
    return bundle


def execute(bundle,approved_total):
    if approved_total!=TOTAL_CAP:raise ValueError('Requires explicit approval of the $15 combined cap')
    prior=validate_prior()
    if lifecycle.matching_pod(prior) is not None:raise ValueError('Prior pod still exists')
    lifecycle.JOB_ROOT=JOB_ROOT;lifecycle.RESULTS=RESULTS
    lifecycle.JOB_PREFIX='mc-heal-resume-';lifecycle.proposal=proposal
    lifecycle.BUDGET=NEW_CAP;lifecycle.MAX_MINUTES=MAX_MINUTES
    lifecycle.BUNDLE_UPLOAD_TIMEOUT=300  # Includes the 1.7-GB saved optimizer/adapters.
    lifecycle.execute(bundle,NEW_CAP)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true');ap.add_argument('--quote',action='store_true')
    ap.add_argument('--approved-total-budget-usd',type=float);args=ap.parse_args()
    if args.quote:
        lifecycle.write_json({'checked_at':time.time(),'H200_secure_usd_hr':lifecycle.fresh_quote()},RESULTS/'restart-price-quote.json')
    bundle=prepare()
    if args.execute:execute(bundle,args.approved_total_budget_usd)
    else:print(json.dumps(proposal(),indent=2));print('Prepared',bundle,'— no paid action performed.')
