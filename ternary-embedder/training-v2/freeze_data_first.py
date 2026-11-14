"""Freeze both branches using only the predeclared held-source validation rule."""
import datetime,json,sys
from pathlib import Path
import torch
from pair_training_state import digest
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent

def main():
 work=ROOT/'work/data-first-run';out=ROOT/'data-first/results';out.mkdir(exist_ok=True)
 if (out/'frozen.json').exists():raise RuntimeError('Candidates already frozen')
 state=json.loads((work/'pod.json').read_text());assert state['deleted'] and state['deletion_verified_http_status']==404
 expected_data=json.loads((ROOT/'data-first/capfit-audit.json').read_text())['output_sha256']['train'];result={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selected_before_external_scores':True,'rule':'Best full-source macro nDCG@10; every source within 2 points of initialization; initialization is null control','models':{}}
 for mode in ('float','ternary'):
  run=work/'backups'/('data-first-'+mode);path=run/'best.pt';ck=torch.load(path,map_location='cpu',weights_only=False);meta=ck['meta'];config=meta['config'];valid=meta['validation']
  assert ck['quant']==(mode=='ternary') and ck['dims']['d']==384 and ck['dims']['layers']==2
  assert config['data_sha256']==expected_data and config['teacher_targets'] is False
  assert valid['source_floor_passed']
  completed=json.loads((run/'completed.json').read_text()) if (run/'completed.json').exists() else None
  result['models'][mode]={'path':str(path.resolve()),'sha256':digest(path),'step':meta['step'],'validation':valid,'completion':completed,'initial_checkpoint_sha256':config['checkpoint_sha256']}
 (out/'frozen.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
