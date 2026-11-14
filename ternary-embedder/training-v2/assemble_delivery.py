"""Select the final development family, copy exact artifacts, and score new exports."""
import json,subprocess,sys,time
from pathlib import Path
from pair_training_state import atomic_json,digest
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
def main():
 final=ROOT/'final-float-blends/results/summary.json'
 while not final.exists():time.sleep(5)
 baseline_path=ROOT/'float-blends/results/summary.json';baseline=json.loads(baseline_path.read_text())['models']['float-25']
 candidates=[(baseline_path,'float-25',baseline)]
 for name,model in json.loads(final.read_text())['models'].items():
  if model['core_retrieval']>=baseline['core_retrieval']-.1 and model['stsbenchmark']>=baseline['stsbenchmark']-.2:
   candidates.append((final,name,model))
 summary,arm,selected=max(candidates,key=lambda item:item[2]['mteb_12_mean'])
 output=PROJECT/'models/campaign-20261025'
 decision=dict(scope='Development selection, not independent confirmation. Retain float retrieval within0.1 and STS within0.2 of the strong float blend, then maximize the fixed12-task mean. Ternary chosen for retrieval with retained broad quality.',
               float_summary=str(summary),float_arm=arm,ternary_summary=str(ROOT/'ternary-blends/results/summary.json'),ternary_arm='ternary-3-25',
               float_sha256=selected['selection']['sha256'],cost_ledger=str(ROOT/'campaign/ledger.json'))
 atomic_json(ROOT/'delivery/selection.json',decision)
 command=[sys.executable,str(ROOT/'prepare_precision_family.py'),'--float-summary',str(summary),'--float-arm',arm,
          '--ternary-summary',str(ROOT/'ternary-blends/results/summary.json'),'--ternary-arm','ternary-3-25','--out',str(output),'--q4-group','16']
 if selected['selection']['sha256']==baseline['selection']['sha256']:
  command+=['--q4-summary',str(ROOT/'q4-groups/results/summary.json'),'--q4-arm','q4-g16',
            '--q8-summary',str(ROOT/'provisional-family/results/summary.json'),'--q8-arm','q8']
 subprocess.run(['dibs','run','cpu','--label','Assemble final experimental precision family','--eta','3m','--',*command],check=True)
 subprocess.run([sys.executable,str(ROOT/'score_precision_family.py'),'--freeze',str(output/'frozen.json'),
                 '--output',str(ROOT/'delivery/family-results')],check=True)
 frozen=json.loads((output/'frozen.json').read_text())
 # Benchmark only actual float/ternary CPU inference; packed reference formats
 # do not establish optimized low-bit execution speed.
 atomic_json(ROOT/'delivery/benchmark-frozen.json',dict(models={k:v for k,v in frozen['models'].items() if k in ('float','ternary')}))
 atomic_json(ROOT/'delivery/quality-complete.json',dict(family=str(output/'frozen.json'),sha256=digest(output/'frozen.json')))
 print('Final family quality complete:',output,flush=True)
if __name__=='__main__':main()
