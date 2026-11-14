"""Wider follow-up only for the blend improving both parent core and STS points."""
import json
from pathlib import Path
import subprocess
import sys
from pair_training_state import atomic_json,digest
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'broad-retention/results'


def main():
    report=json.loads((OUT/'diagnostic-summary.json').read_text())
    passing=[name for name,entry in report['models'].items() if all(v>0 for v in entry['versus_immediate_parent'].values())]
    if not passing:print('No blend/early checkpoint improved both parent point estimates.');return
    name=max(passing,key=lambda n:report['models'][n]['core_retrieval'])
    receipt=OUT/'followup-selection.json'
    entry=dict(candidate=name,selection=report['models'][name]['selection'],core_diagnostic_sha256=digest(OUT/'diagnostic-summary.json'),
               rule='Highest core point estimate among diagnostic candidates improving both immediate-parent core and STS. Existing tasks are observed development.',new_reserved_judgments_permitted=False)
    if receipt.exists():assert json.loads(receipt.read_text())==entry
    else:atomic_json(receipt,entry)
    wide=OUT/'mteb'/name
    with (OUT/'followup-wide.log').open('a') as log:
        subprocess.run([str(PROJECT/'.eval-venv/bin/python'),str(ROOT/'evaluate_round2_mteb.py'),'--freeze',str(OUT/'frozen.json'),
                        '--branch',name,'--output',str(wide)],check=True,stdout=log,stderr=subprocess.STDOUT)
    with (OUT/'followup-extra.log').open('a') as log:
        subprocess.run([sys.executable,str(ROOT/'evaluate_observed_extra.py'),'--freeze',str(OUT/'frozen.json'),'--model',name,
                        '--output',str(OUT/'observed-retrieval')],check=True,stdout=log,stderr=subprocess.STDOUT)
    print('Wider diagnostic follow-up completed for',name,flush=True)


if __name__=='__main__':main()
