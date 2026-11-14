import datetime,hashlib,json,subprocess,sys,time,urllib.parse
from pathlib import Path
sys.path.insert(0,str(Path('manifold-compression').resolve()))
import run_scale_pod as pod
base=Path('manifold-compression/results/correctness-healing-20261025')
job=base/'mc-heal-resume-12789a78a60b';prior=base/'mc-heal-4d3614e4c060'
life=json.loads((job/'job-lifecycle.json').read_text())
assert life['pod_id']=='m15lynbrinlrdu' and life['name']==job.name
report=json.loads((job/'healing-extended/result.json').read_text())
assert report['status']=='complete' and report['adapter_reload_exact']
assert life.get('terminated') and pod.matching_pod(life) is None
manifest=json.loads((job/'artifact-manifest.json').read_text());failed=[]
for name,digest in manifest.items():
 p=job/name;h=hashlib.sha256()
 if not p.is_file():failed.append(name);continue
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 if h.hexdigest()!=digest:failed.append(name)
assert not failed,failed
old=json.loads((prior/'job-lifecycle.json').read_text());estimates={};billing={}
for folder,m in [(prior,old),(job,life)]:
 estimates[folder.name]=(m['terminated_at']-m['created_intent'])/3600*(5.29+pod.DISK_RATE)
 q={'podId':m['pod_id'],'grouping':'podId','bucketSize':'hour','startTime':datetime.datetime.fromtimestamp(m['created_intent']-3600,datetime.timezone.utc).isoformat().replace('+00:00','Z'),'endTime':datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z')}
 records=pod.api('GET','/billing/pods?'+urllib.parse.urlencode(q))
 pod.write_json({'query':q,'records':records,'checked_at':time.time()},folder/'billing-final.json')
 billing[folder.name]=sum(float(r['amount']) for r in records)
audit={'pod_id':life['pod_id'],'pod_absent_confirmed':True,'confirmed_at':time.time(),'duration_minutes':(life['terminated_at']-life['created_intent'])/60,'artifact_files_checked':len(manifest),'artifact_failures':failed,'adapter_reload_exact':report['adapter_reload_exact'],'selected_step':report['selected_step'],'updates_completed':len(report['steps']),'estimated_cost_usd':estimates[job.name],'combined_estimated_cost_usd':sum(estimates.values()),'posted_billing_usd':billing,'billing_note':'Posted billing may still be incomplete; elapsed-time cost includes GPU and estimated temporary disk.','combined_budget_cap_usd':15}
pod.write_json(audit,job/'completion-audit.json')
py='target/manifold-audit/env/bin/python'
subprocess.run([py,'manifold-compression/audit_healing_intervals.py',str(job/'healing-extended'),'--out',str(job/'interval-audit.json')],check=True)
subprocess.run([py,'manifold-compression/review_healing_scores.py','--data',str(base/'data.json'),'--folder',str(job/'healing-extended'),'--out',str(job/'syntax-review.json')],check=True)
family={'post_hoc':True,'used_for_selection':False,'arms':{}}
for name,path in [('stock',prior/'healing/stock-test.json'),('unhealed_75',prior/'healing/unhealed_75-test.json'),('step16',prior/'healing/healed-selected-test.json'),('continued_selected',job/'healing-extended/healed-selected-test.json')]:
 d=json.loads(path.read_text());stats={};long=[]
 for c in d['cases']:
  if c['grade'] is None:long.append(c);continue
  f=stats.setdefault(c['family'],{'correct':0,'scored':0,'wrong_ids':[]});f['correct']+=int(c['grade']['correct']);f['scored']+=1
  if not c['grade']['correct']:f['wrong_ids'].append(c['id'])
 assert sum(f['correct'] for f in stats.values())==d['summary']['correct']
 family['arms'][name]={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'families':stats,'long_form_cases':len(long),'long_form_loop_flags':sum(c['combined_loop_flag'] for c in long),'long_form_ended':sum(c['ended'] for c in long)}
pod.write_json(family,job/'family-review.json')
print(json.dumps(audit,indent=2));print(json.dumps(family,indent=2))
