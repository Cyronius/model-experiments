"""Adopt the existing pod only after approval of the concrete $15 total cap.

No pod is created. The original controller is replaced while independent
watchdogs remain active. The original worker finishes and saves its optimizer;
then a continuation processes the remainder of the frozen 32-update plan.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tarfile
import time

import run_scale_pod as pod

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results/correctness-healing-20261025'
JOB=ROOT.parent/'target/manifold-healing/mc-heal-4d3614e4c060'
CAP=15.;MAX_MINUTES=150


def proposal():
    return {'resource':'Continue existing Secure Cloud NVIDIA H200, 141 GB VRAM',
        'pod_id':'do2mevn0r9ac89','gpu_usd_per_hour':5.29,'temporary_disk_gb':200,
        'disk_usd_per_hour_estimate':pod.DISK_RATE,'total_budget_cap_usd':CAP,
        'max_total_minutes_from_original_create':MAX_MINUTES,
        'maximum_planned_total_usd':(5.29+pod.DISK_RATE)*MAX_MINUTES/60,
        'expected_total_minutes':[125,145],
        'scope':'Finish up to 32 frozen training updates using the saved optimizer; original development regression gate remains in force. No second pod, new data or changed hyperparameters.',
        'authorization':'Pending explicit approval of $15 TOTAL, replacing the original $10 cap'}


def prepare():
    paths={n:ROOT/n for n in ['resume_correctness.py','correctness_resume_worker.py']}
    manifest={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()}
    bundle=JOB/'extension.tar.gz'
    with tarfile.open(bundle,'w:gz') as archive:
        for name,path in paths.items():archive.add(path,arcname=name)
    pod.write_json(proposal()|{'source_sha256':manifest,'bundle_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest()},
                   RESULTS/'extension-proposal.json')
    return bundle


def validate_adoption(m,approved_cap,old_pid):
    if approved_cap!=CAP:raise ValueError('Requires explicit approval of the $15 total cap')
    if m['name']!=JOB.name or m.get('pod_id')!='do2mevn0r9ac89':raise ValueError('Wrong job identity')
    if m.get('terminated') or m.get('terminate_requested') or m.get('extension_adopted'):
        raise ValueError('Job cannot be adopted')
    if time.time()>m['deadline']-180:raise ValueError('Too close to the original cutoff to transfer safely')
    args=Path(f'/proc/{old_pid}/cmdline').read_bytes().decode().split('\0')
    if not any(a.endswith('/run_correctness_pod.py') for a in args) or '--execute' not in args:
        raise ValueError('Original controller PID identity differs')


def execute(bundle,approved_cap,old_pid):
    path=JOB/'pod.json';m=json.loads(path.read_text())
    validate_adoption(m,approved_cap,old_pid)
    live=pod.matching_pod(m)
    if live is None or live['name']!=m['name']:raise ValueError('Original pod is no longer available')
    if float(live.get('costPerHr') or 5.29)>5.29+pod.DISK_RATE+.01:raise ValueError('Rate exceeds proposal')
    command=['ssh',*pod.ssh_options(JOB,JOB/'ssh-key'),'-p',str(live['portMappings']['22'])]
    host='root@'+live['publicIp'];output=RESULTS/JOB.name
    remaining=math.ceil((m['created_intent']+MAX_MINUTES*60-time.time())/60)
    subprocess.run(['dibs','hold','gpu','network','--pid',str(os.getpid()),'--eta',f'{remaining}m',
                    '--label','Approved H200 correctness continuation; $15 total cap'],check=True)
    # Stop the old controller before changing shared lifecycle state, so its
    # finally block cannot overwrite the new deadline or delete a resumed pod.
    os.kill(old_pid,signal.SIGSTOP)
    adopted=False
    def sync():
        try:
            return subprocess.run(['rsync','-az','--partial','--timeout=15','--exclude=teacher_logits/','--exclude=*.tmp',
                '-e',shlex.join(command),host+':/workspace/scale/results/',str(output)+'/' ],
                timeout=40,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
        except subprocess.TimeoutExpired:return False
    try:
        # Read again after freezing the controller; reject any concurrent exit.
        m=json.loads(path.read_text())
        if m.get('terminated') or m.get('terminate_requested'):raise ValueError('Original controller was already exiting')
        m.update(original_deadline=m['deadline'],deadline=m['created_intent']+MAX_MINUTES*60,
            budget_cap_usd=CAP,extension_adopted=True,extension_authorized_at=time.time(),
            controller_pid=os.getpid(),previous_controller_pid=old_pid,extension_proposal=proposal())
        pod.write_json(m,path)
        with (JOB/'extension-watchdog.log').open('a') as log:
            subprocess.Popen([sys.executable,str(ROOT/'run_scale_pod.py'),'--watchdog',str(path)],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        adopted=True;os.kill(old_pid,signal.SIGKILL)
        with bundle.open('rb') as archive:
            subprocess.run([*command,host,'tar xzf - -C /workspace/scale'],stdin=archive,timeout=30,check=True)
        print('Adopted existing pod',m['pod_id'],'total deadline',m['deadline'],'total cap',CAP,flush=True)
        continuation=False
        while time.time()<m['deadline']-30:
            sync();state=json.loads((output/'status.json').read_text())
            print(state,flush=True)
            if state['stage']=='failed':break
            if state['stage']=='complete':
                if continuation:
                    if state.get('continuation') or state.get('continuation_skipped'):break
                else:
                    result=json.loads((output/'healing/result.json').read_text())
                    if result.get('stop_reason')=='development regression gate' or len(result['steps'])>=32:break
                    setup='cd /workspace/scale && SCALE_JOB_DEADLINE='+str(m['deadline']-120)+\
                        ' env/bin/python -u correctness_resume_worker.py > results/resume-worker.log 2>&1'
                    remote='nohup bash -c '+shlex.quote(setup)+' > /workspace/scale/resume-launch.log 2>&1 < /dev/null &'
                    subprocess.run([*command,host,remote],timeout=20,check=True);continuation=True
            time.sleep(20)
        m['final_sync_succeeded']=sync()
        manifest=output/'artifact-manifest.json'
        if manifest.exists():
            hashes=json.loads(manifest.read_text())
            m['artifact_checksums_verified']=all((output/n).is_file() and hashlib.sha256((output/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    finally:
        if not adopted:
            os.kill(old_pid,signal.SIGCONT)
        else:
            m['terminate_requested']=True;pod.write_json(m,path)
            pod.terminate(m)
            if pod.matching_pod(m) is None:m['terminated']=True;m['terminated_at']=time.time()
            pod.write_json(m,path)
            pod.write_json({k:v for k,v in m.items() if k not in ('proposal','extension_proposal')},output/'job-lifecycle.json')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true');ap.add_argument('--approved-total-budget-usd',type=float)
    ap.add_argument('--original-controller-pid',type=int,default=1564314)
    args=ap.parse_args();bundle=prepare()
    if args.execute:execute(bundle,args.approved_total_budget_usd,args.original_controller_pid)
    else:print(json.dumps(proposal(),indent=2));print('Prepared; no lifecycle or spending change.')
