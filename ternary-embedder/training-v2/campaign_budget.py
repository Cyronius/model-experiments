"""Conservative cumulative spending reservations for the explicitly approved campaign."""
import contextlib,fcntl,json,math,os
from pathlib import Path
def atomic_json(path,data):
 tmp=path.with_suffix('.'+str(os.getpid())+'.tmp')
 with tmp.open('w') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(path)
ROOT=Path(__file__).resolve().parent;LEDGER=ROOT/'campaign/ledger.json'

@contextlib.contextmanager
def locked():
 with LEDGER.with_suffix('.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)

def refresh(ledger):
 # Durable per-job reservations also protect against an older, already-running
 # controller publishing a ledger snapshot without a newly reserved job.
 directory=LEDGER.parent/'reservations';directory.mkdir(exist_ok=True)
 jobs={j['name']:j for j in ledger['jobs']}
 for path in directory.glob('*.json'):
  record=json.loads(path.read_text());name=record['name']
  if name in jobs and jobs[name]['reserved_cap_usd']!=record['reserved_cap_usd']:raise RuntimeError('Immutable reservation cap differs from ledger')
  jobs.setdefault(name,record)
 ledger['jobs']=list(jobs.values())
 for name,job in jobs.items():
  path=directory/(name+'.json')
  if not path.exists():atomic_json(path,{k:job[k] for k in ('name','reserved_cap_usd','max_hours','gpu','state_path','status') if k in job})
 realized=reserved=0.
 for job in ledger['jobs']:
  state_path=Path(job['state_path']) if job.get('state_path') else ROOT/'work'/(job['name']+'-run')/'pod.json'
  if state_path.exists():
   state=json.loads(state_path.read_text())
   if state.get('deleted') and 'estimated_total_usd' in state:
    job.update(status='completed',pod_id=state['id'],actual_estimated_usd=state['estimated_total_usd']);realized+=state['estimated_total_usd'];continue
   job.update(status='active' if not state.get('deleted') else 'reconciling',pod_id=state['id'])
  reserved+=job['reserved_cap_usd']
 ledger['realized_estimated_usd']=realized;ledger['outstanding_reserved_usd']=reserved;ledger['committed_upper_bound_usd']=realized+reserved;ledger['uncommitted_usd']=ledger['authorized_total_usd']-realized-reserved
 return ledger

def sync():
 with locked():
  ledger=refresh(json.loads(LEDGER.read_text()));atomic_json(LEDGER,ledger);return ledger

def reserve(job):
 cap=job['reserved_cap_usd']
 if not math.isfinite(cap) or cap<=0:raise ValueError('Invalid reservation cap')
 with locked():
  ledger=refresh(json.loads(LEDGER.read_text()))
  if any(j['name']==job['name'] for j in ledger['jobs']):raise RuntimeError('Campaign reservation already exists')
  if ledger['committed_upper_bound_usd']+cap>ledger['authorized_total_usd']+1e-9:raise RuntimeError('Reservation would exceed approved total budget')
  atomic_json(LEDGER.parent/'reservations'/(job['name']+'.json'),job);ledger['jobs'].append(job);ledger=refresh(ledger);atomic_json(LEDGER,ledger);return ledger

def guard(profile,plan,approval):
 if not approval.get('campaign_cap_usd'):return
 ledger=sync();matches=[j for j in ledger['jobs'] if j['name']==profile]
 if len(matches)!=1:raise RuntimeError('Campaign reservation missing or ambiguous')
 job=matches[0]
 if job.get('status')=='completed':raise RuntimeError('Campaign job approval has already been used')
 if job['reserved_cap_usd']!=plan['proposed_total_cap_usd']:raise RuntimeError('Job cap differs from campaign reservation')
 if ledger['committed_upper_bound_usd']>ledger['authorized_total_usd']+1e-9:raise RuntimeError('Campaign would exceed approved total budget')
 if approval['campaign_cap_usd']!=ledger['authorized_total_usd']:raise RuntimeError('Campaign authorization mismatch')
