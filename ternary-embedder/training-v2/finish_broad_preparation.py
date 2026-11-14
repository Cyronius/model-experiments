"""Finish authorized free preparation; paid launch is deliberately not part of this script."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from broad_data_fetch import atomic,digest
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
REQUIRED=('gooaq','yahoo','hotpot','trivia','squad-exact','stack-balanced','horizon-structured','msmarco-full','nq-recovered','trivia-recovered','paq-00','paq-01','paq-02','paq-03')


def phase(name,command,resources=('cpu',),eta='20m'):
    log=ROOT/'work/broad-data'/(name+'.log');env=dict(os.environ,DIBS_AGENT='broad_'+name,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    atomic(ROOT/'broad-data/active-work.json',dict(status='Preparing '+name,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log=str(log),paid_pod_active=False,new_spending_authorized=False))
    with log.open('a') as output:
        subprocess.run(['dibs','run',*resources,'--label','Broad embedding corpus '+name,'--eta',eta,'--',sys.executable,*command],check=True,stdout=output,stderr=subprocess.STDOUT,env=env)


def main():
    deadline=time.monotonic()+2*3600
    while True:
        pending=[]
        for name in REQUIRED:
            p=ROOT/'work/broad-data/candidates'/name/'manifest.json'
            if not p.exists() or not json.loads(p.read_text()).get('complete'):pending.append(name)
        if not pending:break
        atomic(ROOT/'broad-data/active-work.json',dict(status='Fetching and curating',pending_sources=pending,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),paid_pod_active=False,new_spending_authorized=False))
        if time.monotonic()>deadline:raise RuntimeError('Free data preparation exceeded its two-hour wait; inspect source logs')
        time.sleep(20)
    if not (ROOT/'work/broad-data/ready/audit.json').exists():phase('build',[str(ROOT/'broad_data_build.py')],eta='35m')
    if not (ROOT/'broad-data/runpod-job.json').exists():phase('job_plan',[str(ROOT/'prepare_broad_job.py')],eta='10m')
    if not (ROOT/'broad-data/preflight.json').exists():phase('preflight',[str(ROOT/'broad_preflight.py')],eta='10m')
    if not (ROOT/'work/broad-data-run/package.json').exists():phase('package',[str(ROOT/'pair_runpod.py'),'prepare','--profile','broad-data'],resources=('cpu','build'),eta='15m')
    if not (ROOT/'broad-data/preparation-complete.json').exists():phase('documentation',[str(ROOT/'document_broad_data.py')],eta='10m')
    audit=json.loads((ROOT/'work/broad-data/ready/audit.json').read_text());plan=json.loads((ROOT/'broad-data/runpod-job.json').read_text());package=json.loads((ROOT/'work/broad-data-run/package.json').read_text())
    assert package['plan_sha256']==digest(ROOT/'broad-data/runpod-job.json')
    atomic(ROOT/'broad-data/active-work.json',dict(status='Prepared, verified and packaged; awaiting NEW $6 spending approval',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),train_rows=audit['train_rows'],validation_rows=audit['validation_rows'],paid_pod_active=False,new_spending_authorized=False,proposed_cap_usd=6,package_sha256=package['sha256'],package_bytes=package['bytes']))
    print('All free preparation completed; new $6 approval required before paid launch.',flush=True)


if __name__=='__main__':main()
