"""Read-only progress plus conservative budget reconciliation for owned jobs."""
import datetime,json,time
from pathlib import Path
from campaign_budget import sync
ROOT=Path(__file__).resolve().parent

def last_json(path):
 if not path.exists():return None
 with path.open('rb') as f:
  f.seek(0,2);size=f.tell();f.seek(max(0,size-10000));lines=f.read().splitlines()
 for line in reversed(lines):
  try:return json.loads(line)
  except (ValueError,UnicodeDecodeError):continue
 return None

def main():
 ledger=sync();now=time.time();spent=0.;jobs=[]
 for job in ledger['jobs']:
  state_path=Path(job['state_path']);profile=job['name']
  if not state_path.exists():continue
  state=json.loads(state_path.read_text());plan=json.loads((ROOT/profile/'runpod-job.json').read_text());elapsed=(state.get('end_time',now)-state['started'])/60
  cost=state.get('estimated_total_usd',elapsed/60*(float(state['rate'])+plan['disk_gb']*.1/720));spent+=cost
  if state.get('deleted'):continue
  backups=state_path.parent/'backups';receipt_path=backups/'verified.json';receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {};checkpoints=[(k,v) for k,v in receipt.items() if '/checkpoint-' in k and k.endswith('.pt')];latest=max(checkpoints,key=lambda x:x[1]['verified_unix']) if checkpoints else None
  run=backups/latest[0].split('/')[0] if latest else None;pointer=json.loads((run/'latest.json').read_text()) if run and (run/'latest.json').exists() else None;log=last_json(run/'log.jsonl') if run else None
  namespace=plan.get('run_namespace',profile)
  row={'profile':profile,'pod':state['id'],'elapsed_minutes':round(elapsed,1),'estimated_usd':round(cost,3),'cap_usd':state['approved_total_cap'],'completed_arms':len(list(backups.glob(namespace+'-*/completed.json'))),'planned_arms':len(plan['training']['arms']),'latest_verified_run':run.name if run else None,'verified_step':pointer['step'] if pointer else None,'total_steps':log['total_steps'] if log else None,'backup_age_seconds':round(now-latest[1]['verified_unix']) if latest else None}
  jobs.append(row)
 print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'estimated_campaign_spent_usd':round(spent,3),'full_commitment_upper_bound_usd':ledger['committed_upper_bound_usd'],'approved_campaign_usd':ledger['authorized_total_usd'],'active_jobs':jobs},indent=2))
if __name__=='__main__':main()
