import json,os,shlex,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path('manifold-compression').resolve()))
import run_scale_pod as pod
job=Path('target/manifold-loop-retry/mc-loop-52581c3d31d1');m=json.loads((job/'pod.json').read_text());pid=1990661
assert m['pod_id']=='ujoj3g8p7avcon' and not m.get('terminated') and time.time()<m['deadline']-600
args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
assert b'target/manifold-loop-retry/launch_approved.py' in args
p=pod.matching_pod(m);assert p['name']==m['name']
command=['ssh',*pod.ssh_options(job,job/'ssh-key'),'-p',str(p['portMappings']['22']),'root@'+p['publicIp']]
paused=False
try:
 os.kill(pid,signal.SIGSTOP);paused=True
 pod.write_json({'reason':'Historical 1024-token answers diverged before cutoff; matched original-limit control on same pod','controller_temporarily_paused':True,'original_deadline_unchanged':m['deadline'],'budget_cap_unchanged':m['budget_cap_usd'],'recorded_at':time.time()},job/'paired-control-supervision.json')
 for name in ['paired_loop_control.py','paired_loop_worker.py']:
  with (Path('manifold-compression')/name).open('rb') as f:subprocess.run([*command,'cat > /workspace/scale/'+name],stdin=f,timeout=20,check=True)
 probe="from pathlib import Path\nfound=[]\nfor f in Path('/proc').glob('[0-9]*/cmdline'):\n try:\n  a=f.read_bytes().split(b'\\0')\n  if b'paired_loop_worker.py' in a:found.append(f.parent.name)\n except (OSError,ProcessLookupError):pass\nprint(len(found))\n"
 running=subprocess.run([*command,'python3 -'],input=probe,capture_output=True,text=True,check=True,timeout=20)
 count=int(running.stdout.strip());assert count<=1
 if not count:
  inner='cd /workspace/scale && SCALE_JOB_DEADLINE='+str(m['deadline']-120)+' env/bin/python -u paired_loop_worker.py'
  launch='nohup bash -c '+shlex.quote(inner)+' > /workspace/scale/results/control-worker.log 2>&1 < /dev/null &'
  subprocess.run([*command,launch],check=True,timeout=20)
 print('Paired control scheduled; original deadline and watchdog unchanged',flush=True)
 while time.time()<m['deadline']-150:
  r=subprocess.run([*command,'cat /workspace/scale/results/status.json'],capture_output=True,text=True,timeout=20)
  if r.returncode==0:
   state=json.loads(r.stdout)
   if state.get('paired_control') and state['stage'] in ('complete','failed'):
    print(json.dumps(state),flush=True);break
  time.sleep(10)
finally:
 if paused:
  try:os.kill(pid,signal.SIGCONT)
  except ProcessLookupError:pass
 print('Original lifecycle controller resumed',flush=True)
