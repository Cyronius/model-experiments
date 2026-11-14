import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path('manifold-compression').resolve()))
import run_scale_pod as pod
job=Path('target/manifold-loop-retry/mc-loop-52581c3d31d1')
m=json.loads((job/'pod.json').read_text())
print(json.dumps({'elapsed_min':round((time.time()-m['created_intent'])/60,1),'remaining_min':round((m['deadline']-time.time())/60,1),'terminated':m.get('terminated',False)}))
if m.get('terminated'):sys.exit(0)
p=pod.matching_pod(m)
if p is None:print('Pod absent');sys.exit(0)
if not p.get('publicIp') or not (p.get('portMappings') or {}).get('22'):print('SSH endpoint not ready');sys.exit(0)
script='''import json,subprocess
from pathlib import Path
root=Path('/workspace/scale/results')
s=root/'status.json'
if s.exists():print(s.read_text())
p=root/'loop-retry/result.json'
if p.exists():
 r=json.loads(p.read_text());print(json.dumps({'status':r['status'],'completed_batches':len(r.get('batches',[])),'targets':r.get('target_summary')}))
 p=root/'loop-retry/generations.json'
 if p.exists():
  targets=set(json.loads(Path('/workspace/scale/loop-retry.json').read_text())['target_ids'])
  d=json.loads(p.read_text());print('completed cases',len(d['cases']))
  for c in d['cases']:
   if c['id'] in targets:print(json.dumps({k:c[k] for k in ['id','generated_tokens','ended','combined_loop_flag']}))
 q=root/'original-limit-control/result.json'
 if q.exists():
  z=json.loads(q.read_text());print('control',json.dumps({'status':z['status'],'completed_batches':len(z.get('batches',[]))}))
  a=json.loads((root/'original-limit-control/generations.json').read_text())['cases']
  b={c['id']:c for c in json.loads((root/'loop-retry/generations.json').read_text())['cases']}
  print('same-pod prefix agreement',sum(b[c['id']]['answer'].startswith(c['answer']) for c in a),'of',len(a))
else:
 p=root/'worker.log'
 if p.exists():print(p.read_text(errors='replace')[-350:])
print('GPU:',subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip())
'''
r=subprocess.run(['ssh',*pod.ssh_options(job,job/'ssh-key'),'-p',str(p['portMappings']['22']),'root@'+p['publicIp'],'python3 -'],input=script,text=True,capture_output=True,timeout=20)
print(r.stdout)
if r.returncode:print(r.returncode,r.stderr[-300:])
