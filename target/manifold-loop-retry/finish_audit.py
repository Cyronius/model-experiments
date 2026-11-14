import datetime,hashlib,json,sys,time,urllib.parse
from pathlib import Path
sys.path.insert(0,str(Path('manifold-compression').resolve()))
import run_scale_pod as pod
base=Path('manifold-compression/results/loop-retry-20261025');job=base/'mc-loop-52581c3d31d1'
life=json.loads((job/'job-lifecycle.json').read_text())
assert life['pod_id']=='ujoj3g8p7avcon' and life['name']==job.name
control=json.loads((job/'original-limit-control/result.json').read_text())
assert control['status']=='complete'
report=json.loads((job/'loop-retry/result.json').read_text());request=json.loads((base/'loop-retry.json').read_text())
assert report['status']=='complete' and report['target_summary']['cases']==8
assert life.get('terminated') and pod.matching_pod(life) is None
manifest=json.loads((job/'artifact-manifest.json').read_text());failed=[]
for name,digest in manifest.items():
 p=job/name;h=hashlib.sha256()
 if not p.is_file():failed.append(name);continue
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 if h.hexdigest()!=digest:failed.append(name)
assert not failed,failed
q={'podId':life['pod_id'],'grouping':'podId','bucketSize':'hour','startTime':datetime.datetime.fromtimestamp(life['created_intent']-3600,datetime.timezone.utc).isoformat().replace('+00:00','Z'),'endTime':datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z')}
records=pod.api('GET','/billing/pods?'+urllib.parse.urlencode(q));pod.write_json({'query':q,'records':records,'checked_at':time.time()},job/'billing-final.json')
seconds=life['terminated_at']-life['created_intent']
audit={'pod_id':life['pod_id'],'pod_absent_confirmed':True,'confirmed_at':time.time(),'duration_minutes':seconds/60,'artifact_files_checked':len(manifest),'artifact_failures':failed,'estimated_cost_usd':seconds/3600*(5.29+pod.DISK_RATE),'posted_billing_usd':sum(float(r['amount']) for r in records),'billing_note':'Posted billing may be incomplete; estimate includes GPU and temporary disk.','approved_job_cap_usd':3.5,'completed_batches':len(report['batches']),'target_summary':report['target_summary'],'historical_prefix_matches':sum(r['previous_answer_is_exact_prefix'] for r in report['comparisons']),'same_pod_prefix_matches':sum(r['previous_answer_is_exact_prefix'] for r in control['paired_4096_comparisons']),'same_pod_control_complete':True}
pod.write_json(audit,job/'completion-audit.json');print(json.dumps(audit,indent=2))
targets=set(request['target_ids'])
for row in control['paired_4096_comparisons']:
 if row['id'] in targets:print(json.dumps(row))
