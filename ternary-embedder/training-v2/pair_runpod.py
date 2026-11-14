"""Bounded direct-pair jobs with independent deletion watchdog and off-pod backups.

prepare is free/local. launch requires a recorded, explicit approval matching
this proposed job; no API credential is copied into the package or pod.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,tarfile,time,uuid
from pathlib import Path
import pilot_controller as pod
from pair_runpod_api import api as lifecycle_api
pod.api=lifecycle_api
from pair_training_state import atomic_json,digest
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;WORK=ROOT/'work/data-first-run';STATE=WORK/'pod.json';REMOTE='/workspace/ternary-pilot';PLAN=ROOT/'data-first/runpod-job.json'
pod.STATE=STATE
FILES=['pair_train.py','pair_validation.py','pair_training_state.py','data_first_build.py','grow_depth.py','contrastive_cache.py','test_contrastive_cache.py','test_positive_mask.py','learned_row_scales.py','test_learned_row_scales.py','nli_auxiliary.py','test_nli_auxiliary.py','run_pair_plan.py','run_data_first.sh','pair_job_entry.sh']
PROFILE='data-first';RUN_GLOB='data-first-*';WORKLOAD='run_data_first.sh';BASE_FILES=tuple(FILES)

def configure(profile):
 global PROFILE,WORK,STATE,PLAN,RUN_GLOB,WORKLOAD,FILES,SNAPSHOT
 PROFILE=profile;WORK=ROOT/'work'/(profile+'-run');STATE=WORK/'pod.json';PLAN=ROOT/profile/'runpod-job.json'
 RUN_GLOB=('scales' if profile=='scales-rest' else profile)+'-*';WORKLOAD={'data-first':'run_data_first.sh','depth3':'run_depth3.sh','batch1024':'run_batch1024.sh','hardneg':'run_hardneg.sh','nli-aux':'run_nli_aux.sh','science-mix':'run_science_mix.sh','science-float':'run_science_float.sh','science-ternary':'run_science_ternary.sh','scales':'run_scales.sh','scales-rest':'run_scales_rest.sh','human-mix':'run_human_mix.sh','broad-data':'run_broad_data.sh'}[profile]
 FILES=[f for f in BASE_FILES if f!='run_data_first.sh']+[WORKLOAD]
 SNAPSHOT=BASE_SNAPSHOT.replace("glob('data-first-*')",f"glob('{RUN_GLOB}')")
 pod.STATE=STATE

def prepare():
 WORK.mkdir(parents=True,exist_ok=True)
 audit=json.loads((ROOT/'data-first/capfit-audit.json').read_text())
 for split,expected in audit['output_sha256'].items():
  if digest(ROOT/'work/data-first/capfit'/(split+'.jsonl'))!=expected:raise RuntimeError('Prepared data differs from audited input: '+split)
 files=[ROOT/f for f in FILES]+[PROJECT/'tern-embed/model.py']
 files+=list((PROJECT/'tokenizer').glob('*'))+list((ROOT/'data-first').glob('*.json'))
 plan=json.loads(PLAN.read_text())
 if plan.get('run_namespace',PROFILE)+'-*'!=RUN_GLOB:raise RuntimeError('Unexpected continuation run namespace')
 for spec in plan.get('resume_inputs',[]):
  path=(PROJECT/spec['path']).resolve()
  if not path.is_relative_to((ROOT/'runs').resolve()) or digest(path)!=spec['sha256']:raise RuntimeError('Resume input path/hash mismatch')
  files.append(path)
 if plan.get('data_inputs'):
  for spec in plan['data_inputs'].values():
   path=(PROJECT/spec['path']).resolve()
   if not path.is_relative_to(PROJECT.resolve()) or digest(path)!=spec['sha256']:raise RuntimeError('Data input path/hash mismatch')
   files.append(path)
  files += [PROJECT/f for f in plan.get('data_audit_files',[])]
 else:files += [ROOT/'work/data-first/capfit'/f for f in ('train.jsonl','dev.jsonl','audit.json')]
 if PROFILE!='data-first':
  starts=list((ROOT/'work'/PROFILE/'starts').glob('*.pt'))
  if not starts:raise RuntimeError('No prepared starting checkpoints for '+PROFILE)
  metadata=list((ROOT/PROFILE).glob('*.json'))
  if PROFILE=='broad-data':metadata=[p for p in metadata if p.name not in ('active-work.json','session-notes.json','preparation-complete.json')]
  files+=starts+metadata
 else:files += [PROJECT/'tern-embed/pulls/grid1/runs/2a_384/model.pt',PROJECT/'retrieval-diagnostics/float-control/runs/float150/model.pt']
 manifest={str(p.relative_to(PROJECT)):digest(p) for p in files if p.is_file()};atomic_json(WORK/'package-manifest.json',manifest)
 path=WORK/'package.tar.gz'
 with tarfile.open(path,'w:gz',compresslevel=1) as t:
  for p in files:
   if p.is_file():t.add(p,arcname='ternary-embedder/'+str(p.relative_to(PROJECT)))
  t.add(WORK/'package-manifest.json',arcname='package-manifest.json')
 atomic_json(WORK/'package.json',{'path':str(path),'sha256':digest(path),'bytes':path.stat().st_size,'files':len(manifest),'profile':PROFILE,'plan_sha256':digest(PLAN)})
 print('Package ready:',path.stat().st_size,'bytes;',len(manifest),'verified source/input files',flush=True)

SNAPSHOT=r'''
import pathlib,json,os,hashlib,shutil,uuid,time
root=pathlib.Path('/workspace/ternary-pilot');snapshot=root/('pull-'+uuid.uuid4().hex);snapshot.mkdir();entries=[]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
for run in (root/'ternary-embedder/training-v2/runs').glob('data-first-*'):
 latest=run/'latest.json'
 if latest.exists():
  for attempt in range(4):
   target=None
   try:
    info=json.loads(latest.read_text());target=snapshot/run.name/info['file'];target.parent.mkdir(exist_ok=True);target.unlink(missing_ok=True);os.link(run/info['file'],target)
    if digest(target)!=info['sha256']:raise FileNotFoundError('Checkpoint publication changed')
    break
   except FileNotFoundError:
    if target is not None:target.unlink(missing_ok=True)
    if attempt==3:shutil.rmtree(snapshot);raise
    time.sleep(.05)
  pointer=snapshot/run.name/'latest.json';pointer.write_text(json.dumps(info))
  for pinned in (target,pointer):entries.append({'remote':str(pinned),'relative':run.name+'/'+pinned.name,'sha256':digest(pinned),'bytes':pinned.stat().st_size})
 for src in run.iterdir():
  if src.suffix not in ('.pt','.json','.jsonl'):continue
  if src.name.endswith('.tmp') or src.name=='latest.json' or src.name.startswith('checkpoint-'):continue
  dst=snapshot/run.name/src.name;dst.parent.mkdir(exist_ok=True)
  if dst.exists():continue
  try:
   if src.suffix=='.pt':os.link(src,dst)
   else:dst.write_bytes(src.read_bytes())
  except FileNotFoundError:continue
  entries.append({'remote':str(dst),'relative':run.name+'/'+src.name,'sha256':digest(dst),'bytes':dst.stat().st_size})
for src in (root/'pilot.log',root/'ternary-embedder/training-v2/pair-environment.txt',root/'ternary-embedder/training-v2/pair.status'):
 if src.exists():
  dst=snapshot/src.name;dst.write_bytes(src.read_bytes());entries.append({'remote':str(dst),'relative':src.name,'sha256':digest(dst),'bytes':dst.stat().st_size})
print(json.dumps({'snapshot':str(snapshot),'files':entries}))
'''

BASE_SNAPSHOT=SNAPSHOT

def backup(state):
 payload=json.loads(pod.remote(state,"python - <<'SNAPSHOT'\n"+SNAPSHOT+"\nSNAPSHOT",timeout=120));dest=WORK/'backups';dest.mkdir(exist_ok=True);receipt_path=dest/'verified.json';receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
 try:
  for item in payload['files']:
   path=dest/item['relative'];old=receipt.get(item['relative']);path.parent.mkdir(parents=True,exist_ok=True)
   if old and old['sha256']==item['sha256'] and path.exists():continue
   temp=path.with_suffix(path.suffix+'.download');pod.scp(state,item['remote'],temp,timeout=240)
   if digest(temp)!=item['sha256']:temp.unlink(missing_ok=True);raise RuntimeError('Local backup checksum mismatch')
   temp.replace(path);receipt[item['relative']]=dict(item,verified_unix=time.time());atomic_json(receipt_path,receipt)
   print('Verified local backup:',item['relative'],flush=True)
  for run in dest.glob(RUN_GLOB):
   for old in sorted(run.glob('checkpoint-*.pt'))[:-2]:old.unlink();old.with_suffix('.json').unlink(missing_ok=True)
 finally:
  # Snapshot paths are generated by the fixed trusted script, not user input.
  if not payload['snapshot'].startswith(REMOTE+'/pull-'):raise RuntimeError('Unexpected snapshot directory')
  pod.remote(state,"rm -rf -- '"+payload['snapshot']+"'",timeout=30)
 return any('/checkpoint-' in f['relative'] and f['relative'].endswith('.pt') for f in payload['files'])

def watchdog():
 while True:
  state=json.loads(STATE.read_text())
  if state.get('deleted'):return
  if time.time()>=state['deadline']:
   pod.delete(state,'independent '+PROFILE+' budget watchdog');return
  time.sleep(5)

def launch():
 approval_path=WORK/'approval.json'
 if not approval_path.exists():raise RuntimeError('Explicit job-specific spending approval has not been recorded')
 approval=json.loads(approval_path.read_text());plan=json.loads(PLAN.read_text())
 duration=plan.get('work_duration_minutes',plan['work_deadline_minutes']-10)
 if not 0<duration<plan['work_deadline_minutes']<plan['watchdog_delete_minutes']<=plan['maximum_elapsed_hours']*60:raise RuntimeError('Invalid nested work/deletion deadlines')
 if (plan['maximum_gpu_rate_usd_per_hour']+plan['disk_gb']*.1/720)*plan['maximum_elapsed_hours']>plan['proposed_total_cap_usd']:raise RuntimeError('Resource bounds exceed total cap')
 if approval.get('consumed'):raise RuntimeError('This job-specific spending approval has already been consumed')
 if not approval.get('authorized') or approval.get('cap_usd')!=plan['proposed_total_cap_usd'] or approval.get('max_hours')!=plan['maximum_elapsed_hours']:raise RuntimeError('Approval does not match this job')
 if STATE.exists():raise RuntimeError('This bounded job already has pod state; use a separate campaign reservation/profile for another job')
 if approval.get('campaign_cap_usd'):
  from campaign_budget import guard
  guard(PROFILE,plan,approval)
 package=json.loads((WORK/'package.json').read_text());path=Path(package['path'])
 if package.get('profile')!=PROFILE or package.get('plan_sha256')!=digest(PLAN):raise RuntimeError('Prepared package plan is stale; prepare again before allocation')
 if digest(path)!=package['sha256']:raise RuntimeError('Prepared package changed')
 cloud=os.environ.get('TERNARY_RUNPOD_CLOUD','SECURE')
 if cloud not in ('SECURE','COMMUNITY'):raise RuntimeError('Invalid cloud choice')
 started=time.time();name='ternary-'+PROFILE+'-'+str(int(started));created=None
 try:
  created=pod.api('POST','pods',dict(name=name,cloudType=cloud,computeType='GPU',gpuCount=1,gpuTypeIds=[plan.get('gpu_type_id','NVIDIA RTX A5000')],templateId='runpod-torch-v280',allowedCudaVersions=plan.get('cuda_versions',[]),dataCenterIds=plan.get('data_center_ids',[]),containerDiskInGb=plan['disk_gb'],volumeInGb=0,supportPublicIp=True,minRAMPerGPU=plan.get('minimum_host_ram_gb',8),ports=['22/tcp'],env={'PUBLIC_KEY':(pod.WORK/'pilot-key.pub').read_text().strip()}))
 except Exception:
  # Recover a pod created despite a dropped API response; never touch other jobs.
  try:
   for live in pod.api('GET','pods'):
    if live.get('name')==name:pod.api('DELETE','pods/'+live['id']);print('Deleted uncertain creation',live['id'],flush=True)
  finally:raise
 state={'id':created['id'],'started':started,'attempt_started':started,'rate':created.get('costPerHr'),'deadline':started+plan['watchdog_delete_minutes']*60,'work_deadline':started+plan['work_deadline_minutes']*60,'approved_total_cap':approval['cap_usd'],'approved_max_hours':approval['max_hours'],'deleted':False,'plan':PROFILE,'cloud':cloud,'package_sha256':package['sha256']};atomic_json(STATE,state)
 with (WORK/'watchdog.log').open('a') as f:child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'watchdog','--profile',PROFILE],stdout=f,stderr=f,start_new_session=True)
 print('Created',state['id'],'rate',state['rate'],'watchdog PID',child.pid,flush=True)
 last_checkpoint_backup=time.time();last_attempt=0
 try:
  if state['rate'] is None or float(state['rate'])>plan['maximum_gpu_rate_usd_per_hour']:raise RuntimeError('GPU rate exceeds approval')
  while time.time()<started+600:
   live=pod.api('GET','pods/'+state['id'])
   if live.get('publicIp') and (live.get('portMappings') or {}).get('22'):
    state.update(publicIp=live['publicIp'],ssh_port=live['portMappings']['22']);atomic_json(STATE,state)
    try:pod.remote(state,'echo ready',timeout=20);break
    except Exception:pass
   time.sleep(10)
  else:raise RuntimeError('Pod setup timed out')
  pod.remote(state,'mkdir -p '+REMOTE);pod.scp(state,path,REMOTE+'/package.tar.gz',upload=True,timeout=plan.get('package_transfer_timeout_seconds',900))
  command="cd /workspace/ternary-pilot && tar xzf package.tar.gz --no-same-owner && python - <<'VERIFY'\nimport pathlib,json,hashlib\nfor name,h in json.loads(pathlib.Path('package-manifest.json').read_text()).items():\n p=pathlib.Path('ternary-embedder')/name\n assert hashlib.sha256(p.read_bytes()).hexdigest()==h,name\nprint('Upload checksums verified')\nVERIFY"
  print(pod.remote(state,command,timeout=180),flush=True)
  pod.remote(state,'cd '+REMOTE+f" && (nohup timeout --signal=TERM --kill-after=120s {duration}m bash ternary-embedder/training-v2/pair_job_entry.sh {WORKLOAD} > pilot.log 2>&1 < /dev/null & echo $! > pilot.pid)")
  last_checkpoint_backup=time.time()
  while time.time()<state['work_deadline']:
   try:
    status=pod.remote(state,'cd '+REMOTE+'; cat ternary-embedder/training-v2/pair.status 2>/dev/null; tail -n 2 pilot.log',timeout=30);print(status,flush=True)
    if time.time()-last_attempt>=60:
     last_attempt=time.time()
     try:
      if backup(state):last_checkpoint_backup=time.time()
     except Exception as e:print('Backup retry needed:',type(e).__name__,flush=True)
    if status.splitlines() and status.splitlines()[0] in ('done','failed'):break
    if time.time()-last_checkpoint_backup>600:raise RuntimeError('Off-pod backups unavailable for ten minutes; stopping training')
   except (subprocess.CalledProcessError,subprocess.TimeoutExpired):print('SSH interruption; retaining verified local checkpoints',flush=True)
   if time.time()-last_checkpoint_backup>600:raise RuntimeError('Off-pod backups unavailable for ten minutes; stopping job')
   time.sleep(20)
  else:print('Training work deadline reached',flush=True)
  pod.remote(state,"pkill -TERM -f '[p]air_train.py' || true",timeout=30)
  for _ in range(6):
   try:backup(state);break
   except Exception:time.sleep(5)
 finally:
  # Recover on errors too. Credentials stay in this process on the local host.
  try:
   if state.get('ssh_port'):backup(state)
  except Exception as e:print('Final recovery unavailable:',type(e).__name__,flush=True)
  pod.delete(state,PROFILE+' job finished or failed; verified local backups retained')
  try:pod.api('GET','pods/'+state['id']);raise RuntimeError('Pod still exists after deletion')
  except pod.RunPodError as e:
   if e.code!=404:raise
  state=json.loads(STATE.read_text());state['deletion_verified_http_status']=404;state['elapsed_hours']=(state['end_time']-started)/3600;state['estimated_compute_usd']=state['elapsed_hours']*float(state['rate']);state['estimated_disk_usd']=state['elapsed_hours']*plan['disk_gb']*.1/720;state['estimated_total_usd']=state['estimated_compute_usd']+state['estimated_disk_usd'];atomic_json(STATE,state)
  if approval.get('campaign_cap_usd'):
   from campaign_budget import sync
   sync()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','watchdog','backup','status','delete']);p.add_argument('--profile',choices=['data-first','depth3','batch1024','hardneg','nli-aux','science-mix','science-float','science-ternary','scales','scales-rest','human-mix','broad-data'],default='data-first');a=p.parse_args();configure(a.profile);WORK.mkdir(parents=True,exist_ok=True)
 if a.action=='prepare':prepare()
 elif a.action=='launch':launch()
 elif a.action=='watchdog':watchdog()
 elif a.action=='backup':backup(json.loads(STATE.read_text()))
 elif a.action=='delete':pod.delete(json.loads(STATE.read_text()),'explicit data-first cleanup')
 else:print(STATE.read_text() if STATE.exists() else 'No pod has been allocated')
