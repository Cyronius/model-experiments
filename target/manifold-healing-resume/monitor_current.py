import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path('manifold-compression').resolve()))
import run_scale_pod as p
job=Path('target/manifold-healing-resume/mc-heal-resume-12789a78a60b')
m=json.loads((job/'pod.json').read_text())
print(json.dumps({'elapsed_minutes':round((time.time()-m['created_intent'])/60,2),'remaining_minutes':round((m['deadline']-time.time())/60,2),'terminated':m.get('terminated',False)}))
if m.get('terminated'):sys.exit(0)
pod=p.matching_pod(m)
if not pod:print('Pod absent');sys.exit(0)
ip,port=pod.get('publicIp'),(pod.get('portMappings') or {}).get('22')
if not ip or not port:print('SSH endpoint not ready');sys.exit(0)
script='''import json,subprocess
from pathlib import Path
root=Path('/workspace/scale/results')
for name in ['status.json','worker.log']:
 p=root/name
 if p.exists():print(name,p.read_text(errors='replace')[-250:])
p=root/'healing-extended/result.json'
if p.exists():
 r=json.loads(p.read_text());print('progress',json.dumps({k:r[k] for k in ['status','stage','continuation_seconds','selected_step'] if k in r}));print('last_updates',json.dumps(r.get('steps',[])[-1:]))
for p in sorted((root/'healing-extended').rglob('*.json')):
 if p.name=='dev.json' or p.name.startswith('healed-selected'):
  x=json.loads(p.read_text());print('generation_progress',str(p.relative_to(root/'healing-extended')),len(x.get('cases',[])),x.get('summary',{}))
try:print(subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw','--format=csv,noheader'],text=True))
except Exception as e:print(type(e).__name__)
'''
r=subprocess.run(['ssh',*p.ssh_options(job,job/'ssh-key'),'-p',str(port),'root@'+ip,'python3 -'],input=script,text=True,capture_output=True,timeout=20)
print(r.stdout)
if r.returncode:print('SSH exit',r.returncode,r.stderr[-400:])
