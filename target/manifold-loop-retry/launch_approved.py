import sys,json
from pathlib import Path
sys.path.insert(0,str(Path('manifold-compression').resolve()))
import run_loop_retry_pod as launch
bundle=launch.JOB_ROOT/'loop-retry-bundle.tar.gz'
p=json.loads((launch.RESULTS/'job-proposal.json').read_text())
a=json.loads((launch.RESULTS/'authorization.json').read_text())
if a['budget_cap_usd']!=3.5:raise ValueError('Wrong authorization')
if launch.sha256(bundle)!=p['bundle_sha256']:raise ValueError('Prepared bundle changed')
for name,source in [('scale_worker.py','loop_retry_worker.py'),('retry_healing_loops.py','retry_healing_loops.py')]:
 if launch.sha256(launch.ROOT/source)!=p['files']['sha256'][name]:raise ValueError('Worker changed after preparation')
if launch.sha256(launch.RESULTS/'loop-retry.json')!=p['files']['sha256']['loop-retry.json']:raise ValueError('Diagnostic request changed')
launch.execute(bundle,3.5)
