"""Controller for the user's approved single A5000 / one-hour / $0.50 pilot.

An independent watchdog deletes this pod at 55 minutes, reserving time for API
retries. Only one active pod; a setup retry shares the original deadline and budget.
API credentials stay on this local machine.
"""
import argparse,hashlib,json,os,shlex,stat,subprocess,sys,tarfile,time,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
WORK=ROOT/'work';STATE=WORK/'pod.json';REMOTE='/workspace/ternary-pilot'
PLAN='pilot'
PROFILES={'pilot':{'hours':1,'work_minutes':45,'watchdog_minutes':55,'budget':.50,'workload':'run_pilot.sh'},
          'balanced-round2':{'hours':2,'work_minutes':105,'watchdog_minutes':115,'budget':.75,'workload':'run_balanced_round2.sh'}}
PROFILE=PROFILES[PLAN]
CLOUD='SECURE'

class RunPodError(RuntimeError):
    def __init__(self,code,message):
        self.code=code;super().__init__(message)

def api(method,route,body=None):
    path=Path.home()/'.config/runpod/credentials.env'
    if stat.S_IMODE(path.stat().st_mode)!=0o600 or stat.S_IMODE(path.parent.stat().st_mode)!=0o700:
        raise RuntimeError('Credential permissions must be 0600/0700')
    token=next(l.split('=',1)[1].strip().strip('\"\'') for l in path.read_text().splitlines() if l.startswith('RUNPOD_API_KEY='))
    req=urllib.request.Request('https://rest.runpod.io/v1/'+route,
          data=json.dumps(body).encode() if body is not None else None,
          headers={'Authorization':'Bearer '+token,'Content-Type':'application/json','User-Agent':'TernaryPilot'},method=method)
    try:
        with urllib.request.urlopen(req,timeout=25) as r:
            raw=r.read();return json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        detail=error.read().decode(errors='replace').replace(token,'[redacted]')
        raise RunPodError(error.code,f'RunPod {method} {route}: HTTP {error.code}: {detail[:1200]}') from None

def save(state):
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(STATE)

def delete(state,reason):
    for _ in range(12):
        try:
            api('DELETE','pods/'+state['id']);break
        except RunPodError as e:
            if e.code==404:break
            time.sleep(3)
        except urllib.error.HTTPError as e:
            if e.code==404:break
            time.sleep(3)
        except Exception:time.sleep(3)
    else:raise RuntimeError('Pod deletion failed; watchdog will retry')
    state.update(deleted=True,end_time=time.time(),reason=reason)
    save(state);print('Pod deleted:',state['id'],reason,flush=True)

def options():return ['-i',str(WORK/'pilot-key'),'-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new',
                        '-o','UserKnownHostsFile='+str(WORK/'known_hosts'),'-o','ConnectTimeout=10',
                        '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2']
def remote(state,command,timeout=60):
    return subprocess.check_output(['ssh',*options(),'-p',str(state['ssh_port']),
          'root@'+state['publicIp'],command],stderr=subprocess.STDOUT,timeout=timeout).decode()
def scp(state,src,dst,upload=False,timeout=240):
    address='root@'+state['publicIp']+':'
    args=[str(src),address+str(dst)] if upload else [address+str(src),str(dst)]
    subprocess.run(['scp',*options(),'-P',str(state['ssh_port']),*args],check=True,timeout=timeout)

def bundle():
    path=WORK/'code.tar.gz'
    with tarfile.open(path,'w:gz') as t:
        for p in ROOT.glob('*.py'):t.add(p,arcname='ternary-embedder/training-v2/'+p.name)
        for p in ROOT.glob('*.sh'):t.add(p,arcname='ternary-embedder/training-v2/'+p.name)
        t.add(PROJECT/'tern-embed/model.py',arcname='ternary-embedder/tern-embed/model.py')
        t.add(PROJECT/'tokenizer',arcname='ternary-embedder/tokenizer')
        t.add(PROJECT/'retrieval-diagnostics/float-control/runs/float150/model.pt',
              arcname='ternary-embedder/retrieval-diagnostics/float-control/runs/float150/model.pt')
        if PLAN=='balanced-round2':
            t.add(PROJECT/'tern-embed/pulls/grid1/runs/2a_384/model.pt',arcname='ternary-embedder/tern-embed/pulls/grid1/runs/2a_384/model.pt')
    return path

def recover(state):
    command="""cd /workspace/ternary-pilot && python - <<'PACK'
import pathlib,hashlib,json,tarfile
root=pathlib.Path('ternary-embedder/training-v2')
files=[p for p in root.rglob('*') if p.is_file() and ('runs' in p.parts or p.name.endswith(('.py','.sh','.status','.txt')))]
files += [p for p in (root/'work').glob('*.json')]
for name in ('pilot.log',):
 p=pathlib.Path(name)
 if p.exists(): files.append(p)
checks={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
pathlib.Path('checks.json').write_text(json.dumps(checks,indent=2))
with tarfile.open('artifacts.tar.gz','w:gz') as t:
 for p in files+[pathlib.Path('checks.json')]:t.add(p)
print('packed',len(files),'files',flush=True)
PACK"""
    print(remote(state,command,timeout=120),flush=True)
    dest=ROOT/'runs'/('approved-pilot' if PLAN=='pilot' else PLAN);dest.mkdir(parents=True,exist_ok=True)
    scp(state,REMOTE+'/artifacts.tar.gz',dest/'artifacts.tar.gz')
    with tarfile.open(dest/'artifacts.tar.gz') as t:t.extractall(dest,filter='data')
    checks=json.loads((dest/'checks.json').read_text())
    for name,h in checks.items():
        if hashlib.sha256((dest/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Artifact hash mismatch: '+name)
    print('Verified',len(checks),'recovered artifacts',flush=True)
    state['artifacts_verified']=len(checks);save(state)

def watchdog():
    while True:
        state=json.loads(STATE.read_text())
        if state.get('deleted'):return
        if time.time()>=state['deadline']:
            try:delete(state,f"independent {PROFILE['watchdog_minutes']}-minute watchdog");return
            except Exception:time.sleep(3)
        else:time.sleep(5)

def launch(retry_setup=False):
    previous=None
    if STATE.exists():
        previous=json.loads(STATE.read_text())
        if not retry_setup or not previous.get('deleted') or time.time()-previous['started']>600:
            raise RuntimeError('Existing pilot; only a deleted setup failure may retry within original budget')
        archive=WORK/('attempt-'+previous['id']+'.json')
        archive.write_text(json.dumps(previous,indent=2)+'\n')
    WORK.mkdir(exist_ok=True)
    # Approval: one A5000, <=$0.30 GPU/hr, <=1 hour, <=$0.50 total.
    # Live rate was checked separately: Secure Cloud A5000 $0.27/hr.
    started=previous['started'] if previous else time.time()
    attempt_started=time.time()
    pod=api('POST','pods',dict(name='ternary-v2-'+PLAN,cloudType=CLOUD,computeType='GPU',gpuCount=1,
          gpuTypeIds=['NVIDIA RTX A5000'],templateId='runpod-torch-v280',containerDiskInGb=60,volumeInGb=0,
          minRAMPerGPU=24,minVCPUPerGPU=2,minDownloadMbps=200,supportPublicIp=True,ports=['22/tcp'],
          env={'PUBLIC_KEY':(WORK/'pilot-key.pub').read_text().strip(),'TERNARY_WORKLOAD':PROFILE['workload']}))
    state={'id':pod['id'],'started':started,'attempt_started':attempt_started,
           'prior_compute_usd':previous.get('estimated_compute_usd',0.) if previous else 0.,'deadline':started+PROFILE['watchdog_minutes']*60,'work_deadline':started+PROFILE['work_minutes']*60,
           'rate':pod.get('costPerHr'),'approved_total_cap':PROFILE['budget'],'approved_max_hours':PROFILE['hours'],'plan':PLAN,'cloud':CLOUD,'deleted':False}
    save(state)
    with (WORK/'watchdog.log').open('a') as f:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'watchdog','--plan',PLAN],stdout=f,stderr=f,start_new_session=True)
    print('Created',state['id'],'rate',state['rate'],'watchdog PID',child.pid,flush=True)
    try:
        if state['rate'] is None or float(state['rate'])>.30:raise RuntimeError('Unexpected rate; terminate immediately')
        while time.time()<started+600:
            live=api('GET','pods/'+state['id'])
            if live.get('publicIp') and (live.get('portMappings') or {}).get('22'):
                state.update(publicIp=live['publicIp'],ssh_port=live['portMappings']['22']);save(state)
                try:
                    print(remote(state,'echo ready',timeout=20),flush=True);break
                except Exception:pass
            time.sleep(10)
        else:raise RuntimeError('Pod setup timed out')
        path=bundle();remote(state,'mkdir -p '+REMOTE)
        scp(state,path,REMOTE+'/code.tar.gz',upload=True)
        command=f"cd {REMOTE} && tar xzf code.tar.gz --no-same-owner && (nohup timeout {PROFILE['work_minutes']-5}m bash ternary-embedder/training-v2/pod_entry.sh {PROFILE['workload']} > pilot.log 2>&1 < /dev/null & echo $! > pilot.pid)"
        remote(state,command);print('Pilot launched',flush=True)
        while time.time()<state['work_deadline']:
            status=remote(state,f'cd {REMOTE}; cat ternary-embedder/training-v2/pilot.status 2>/dev/null; /bin/echo; tail -n 3 pilot.log',timeout=30)
            print(status,flush=True)
            stage=status.splitlines()[0] if status.strip() else ''
            if stage in ('done','failed'):break
            time.sleep(20)
        else:
            remote(state,f'cd {REMOTE}; kill -TERM -- -$(cat pilot.pid) 2>/dev/null || kill $(cat pilot.pid) 2>/dev/null || true')
        recover(state)
    finally:
        current=json.loads(STATE.read_text())
        if not current.get('deleted'):
            state.update(current);delete(state,'pilot finished or failed; no retained paid volume')
    state=json.loads(STATE.read_text())
    state['elapsed_hours']=(state['end_time']-state['started'])/3600
    state['estimated_compute_usd']=state['prior_compute_usd']+(state['end_time']-state['attempt_started'])/3600*float(state['rate'])
    save(state);print(json.dumps({k:state[k] for k in ('id','elapsed_hours','estimated_compute_usd','deleted')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','watchdog','status','delete']);p.add_argument('--retry-setup',action='store_true');p.add_argument('--plan',choices=list(PROFILES),default='pilot');p.add_argument('--cloud',choices=['SECURE','COMMUNITY'],default='SECURE');a=p.parse_args()
    PLAN=a.plan;PROFILE=PROFILES[PLAN];CLOUD=a.cloud
    if PLAN!='pilot':STATE=WORK/('pod-'+PLAN+'.json')
    assert PROFILE['work_minutes']<PROFILE['watchdog_minutes']<PROFILE['hours']*60
    assert .30*PROFILE['hours']+60*.1/720*PROFILE['hours']<PROFILE['budget']
    if a.action=='launch':launch(a.retry_setup)
    elif a.action=='watchdog':watchdog()
    elif a.action=='delete':delete(json.loads(STATE.read_text()),'explicit controller cleanup')
    else:print(json.dumps(json.loads(STATE.read_text()),indent=2))
