import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path('manifold-compression').resolve()))
import run_scale_pod as pod
job=Path('target/manifold-healing-resume/mc-heal-resume-12789a78a60b')
m=json.loads((job/'pod.json').read_text())
print(json.dumps({'elapsed_min':round((time.time()-m['created_intent'])/60,1),'remaining_min':round((m['deadline']-time.time())/60,1),'terminated':m.get('terminated',False)}))
if m.get('terminated'):sys.exit(0)
p=pod.matching_pod(m)
if p is None:print('Pod absent');sys.exit(0)
script='''import json,subprocess
from pathlib import Path
root=Path('/workspace/scale/results');s=json.loads((root/'status.json').read_text());p=root/'healing-extended/result.json'
if p.exists():
 r=json.loads(p.read_text());steps=r.get('steps',[])
 print(json.dumps({'stage':r.get('stage'),'status':r.get('status'),'completed_updates':len(steps),'selected_step':r.get('selected_step'),'last_update':steps[-1] if steps else None}))
 for p in sorted((root/'healing-extended').rglob('*.json')):
  if p.name=='dev.json' or p.name.startswith('healed-selected'):
   d=json.loads(p.read_text());print(json.dumps({'file':str(p.relative_to(root/'healing-extended')),'cases':len(d.get('cases',[])),'summary':d.get('summary',{})}))
else:print(json.dumps(s))
if s.get('stage')=='failed':print(s)
print('GPU utilization:',subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader'],text=True).strip())
'''
r=subprocess.run(['ssh',*pod.ssh_options(job,job/'ssh-key'),'-p',str(p['portMappings']['22']),'root@'+p['publicIp'],'python3 -'],input=script,text=True,capture_output=True,timeout=20)
print(r.stdout)
if r.returncode:print(r.returncode,r.stderr[-300:])
