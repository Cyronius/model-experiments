"""Freeze all completed campaign arms before exposing external development scores."""
import argparse,datetime,json
from pathlib import Path
import torch
from pair_training_state import atomic_json,digest
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--arms',nargs='+',required=True);a=p.parse_args()
 work=ROOT/'work'/(a.profile+'-run');out=ROOT/a.profile/'results';out.mkdir(parents=True,exist_ok=True)
 if (out/'frozen.json').exists():raise RuntimeError('Candidates already frozen')
 state=json.loads((work/'pod.json').read_text());assert state['deleted'] and state['deletion_verified_http_status']==404
 continuation=work/'continuations.json';continuation_states=[]
 if continuation.exists():
  for record in json.loads(continuation.read_text()):
   path=Path(record['state_path']);assert digest(path)==record['state_sha256']
   extra=json.loads(path.read_text());assert extra['deleted'] and extra['deletion_verified_http_status']==404
   continuation_states.append(extra)
 plan=json.loads((ROOT/a.profile/'runpod-job.json').read_text())
 expected_data=plan['training'].get('data_sha256') or json.loads((ROOT/'data-first/capfit-audit.json').read_text())['output_sha256']['train']
 expected_dev=plan.get('data_inputs',{}).get('dev',{}).get('sha256') or json.loads((ROOT/'data-first/capfit-audit.json').read_text())['output_sha256']['dev']
 result={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selected_before_external_scores':True,'continuation_resources':continuation_states,'rule':plan['training'].get('selection','Best full-source macro nDCG@10 at full depth, every source within 2 points of common parent. Missing eligible best: predetermined final diagnostic, not promoted.'),'models':{}}
 for arm in a.arms:
  run=work/'backups'/(a.profile+'-'+arm);completion=json.loads((run/'completed.json').read_text());config=json.loads((run/'config.json').read_text())
  arm_spec=next((v for v in plan['training']['arms'] if isinstance(v,dict) and v['id']==arm),{})
  arm_data=plan['data_inputs'][arm_spec['data_key']]['sha256'] if arm_spec.get('data_key') else expected_data
  assert completion['step']==config['total_steps'] and config['teacher_targets'] is False and config['data_sha256']==arm_data
  assert config['dev_sha256']==expected_dev and config['limit'] is None and config['loaded_rows']==plan['training']['train_rows']
  assert config['epochs']==plan['training']['epochs_each'] and config['steps'] is None
  if plan['training'].get('auxiliary_nli'):
   assert config['auxiliary_sha256']==plan['data_inputs']['aux_train']['sha256'] and config['auxiliary_dev_sha256']==plan['data_inputs']['aux_dev']['sha256']
  best=run/'best.pt';eligible=best.exists();path=best if eligible else run/'model.pt';ck=torch.load(path,map_location='cpu',weights_only=False);meta=ck['meta'];valid=meta['validation']
  assert meta.get('growth_alpha',1.)==1. and ck['dims']['d']==384 and ck['dims']['layers']==completion['layers']
  assert ck['quant']==(config['precision']=='ternary') and (not eligible or valid['source_floor_passed'])
  assert not eligible or valid.get('auxiliary_floor_passed',True)
  result['models'][arm]={'path':str(path.resolve()),'sha256':digest(path),'step':meta['step'],'precision':config['precision'],'layers':ck['dims']['layers'],'eligible_for_promotion':eligible,'validation':valid,'completion':completion,'initial_checkpoint_sha256':config['checkpoint_sha256']}
  if eligible and meta['step']==0 and not config.get('learn_row_scales') and arm_spec.get('checkpoint'):
   initial_path=(ROOT.parent/arm_spec['checkpoint']).resolve();initial=torch.load(initial_path,map_location='cpu',weights_only=False)
   if initial.get('meta',{}).get('query_projection') is None:
    assert initial['dims']==ck['dims'] and initial['quant']==ck['quant'] and initial['state'].keys()==ck['state'].keys()
    assert all(torch.equal(value,ck['state'][key]) for key,value in initial['state'].items())
    selection=result['models'][arm];selection.update(path=str(initial_path),sha256=digest(initial_path),selected_identity=True)
    parents_path=ROOT/a.profile/'parent-selection.json'
    if parents_path.exists():
     parents=json.loads(parents_path.read_text());matches=[v for v in parents.values() if v['prepared_path']==arm_spec['checkpoint']]
     assert len(matches)==1 and matches[0]['selection']['sha256']==selection['sha256']
     source=matches[0];assert digest(Path(source['source_summary']))==source['source_summary_sha256']
     selection.update(source_summary=source['source_summary'],source_arm=source['source_arm'],source_summary_sha256=source['source_summary_sha256'])
 atomic_json(out/'frozen.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
