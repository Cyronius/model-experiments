import sys,json
from pathlib import Path
sys.path.insert(0,str(Path('manifold-compression').resolve()))
import restart_correctness_pod as launch
bundle=launch.JOB_ROOT/'continuation-bundle.tar.gz'
p=json.loads((launch.RESULTS/'restart-proposal.json').read_text())
a=json.loads((launch.RESULTS/'restart-authorization.json').read_text())
if a['total_budget_cap_usd']!=15 or a['new_job_cap_usd']!=7:raise ValueError('Wrong authorization')
if launch.sha256(bundle)!=p['bundle_sha256']:raise ValueError('Prepared bundle changed')
for name,path in launch.bundle_paths().items():
 if launch.sha256(path)!=p['files']['sha256'][name]:raise ValueError('Prepared input changed: '+name)
launch.execute(bundle,15)
