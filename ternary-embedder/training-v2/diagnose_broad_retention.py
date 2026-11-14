"""Free, explicitly exploratory early-stop/weight-retention diagnostics."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from pair_training_state import atomic_json,atomic_torch_save,digest

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
OUT=ROOT/'broad-retention/results'


def main():
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(PROJECT/'tern-embed'))
    from model import Student,tern_codes
    OUT.mkdir(parents=True,exist_ok=True)
    frozen_path=OUT/'frozen.json'
    if not frozen_path.exists():
        original=json.loads((ROOT/'broad-data/results/frozen.json').read_text())
        plan=json.loads((ROOT/'broad-data/runpod-job.json').read_text())
        models={};drift={}
        for precision in ('float','ternary'):
            start=PROJECT/next(v['checkpoint'] for v in plan['training']['arms'] if v['id']==precision+'-broad')
            selected=original['models'][precision+'-broad'];child_path=Path(selected['path'])
            assert digest(child_path)==selected['sha256']
            parent=torch.load(start,map_location='cpu',weights_only=False)
            child=torch.load(child_path,map_location='cpu',weights_only=False)
            assert parent['dims']==child['dims'] and parent['quant']==child['quant']==(precision=='ternary')
            assert parent['state'].keys()==child['state'].keys()
            early=child_path.parent/'milestone-0000500.pt'
            checkpoint=torch.load(early,map_location='cpu',weights_only=False)
            assert checkpoint['meta']['step']==500 and checkpoint['quant']==parent['quant']
            models[precision+'-early500']=dict(path=str(early),sha256=digest(early),precision=precision,layers=3,eligible_for_promotion=False,step=500)
            drift[precision]={}
            for fraction in (.25,.5):
                state={k:parent['state'][k]*(1-fraction)+child['state'][k]*fraction for k in parent['state']}
                name=precision+'-blend'+str(int(fraction*100));path=OUT/(name+'.pt')
                atomic_torch_save(path,dict(dims=parent['dims'],quant=parent['quant'],state=state,
                    meta=dict(exploratory_diagnostic=True,trained_fraction=fraction,parent_sha256=digest(start),child_sha256=selected['sha256'])))
                model=Student.load(path)
                assert model.quantized==(precision=='ternary') and sum(p.numel() for p in model.parameters())==17241984
                models[name]=dict(path=str(path),sha256=digest(path),precision=precision,layers=3,eligible_for_promotion=False,trained_fraction=fraction)
            for prefix in ('word.','out.','layers.0.','layers.1.','layers.2.'):
                keys=[k for k in parent['state'] if k.startswith(prefix)]
                magnitude=sum(float(parent['state'][k].double().square().sum()) for k in keys)
                change=sum(float((child['state'][k]-parent['state'][k]).double().square().sum()) for k in keys)
                record=dict(relative_l2=(change/max(magnitude,1e-30))**.5)
                if precision=='ternary' and prefix.startswith('layers.'):
                    codes=[k for k in keys if k.endswith('.weight') and parent['state'][k].ndim==2]
                    total=sum(parent['state'][k].numel() for k in codes)
                    changed=sum(int((tern_codes(parent['state'][k])[0]!=tern_codes(child['state'][k])[0]).sum()) for k in codes)
                    record['ternary_code_flip_fraction']=changed/total
                drift[precision][prefix]=record
        atomic_json(OUT/'weight-drift.json',drift)
        atomic_json(frozen_path,dict(models=models,local_only=True,paid_resource_allocated=False,
            frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            scope='Exploratory diagnostic chosen after observing the original failure. Existing external tasks are development; no new reserved judgments may be opened from this diagnostic.'))
    frozen=json.loads(frozen_path.read_text())
    if not (OUT/'summary.json').exists():
        subprocess.run([sys.executable,str(ROOT/'score_campaign.py'),'--profile','broad-retention','--arms',*frozen['models'],
                        '--local-frozen','--dibs-inner'],check=True)
    result=json.loads((OUT/'summary.json').read_text())
    parent=json.loads((ROOT/'broad-data/results/development-summary.json').read_text())['models']
    for name,entry in result['models'].items():
        baseline=parent['parent-'+entry['selection']['precision']]
        entry['versus_immediate_parent']=dict(core_delta=entry['core_retrieval']-baseline['core_retrieval'],
                                             sts_delta=entry['stsbenchmark']-baseline['stsbenchmark'])
    atomic_json(OUT/'diagnostic-summary.json',result)
    print(json.dumps({n:dict(core=v['core_retrieval'],sts=v['stsbenchmark'],delta=v['versus_immediate_parent']) for n,v in result['models'].items()},indent=2),flush=True)


if __name__=='__main__':main()
