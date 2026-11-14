"""Run sequential, independently resumable direct-supervision arms from a job plan."""
import argparse,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent

def command(plan,arm):
 t=plan['training'];profile=plan.get('run_namespace',plan['profile'])
 if not re.fullmatch(r'[a-z0-9-]+',profile) or not re.fullmatch(r'[a-z0-9.-]+',arm['id']):raise ValueError('Invalid run name')
 if t['teacher_targets'] is not False:raise ValueError('This runner supports direct data supervision only')
 out=ROOT/'runs'/(profile+'-'+arm['id']);checkpoint=PROJECT/arm['checkpoint'];train=PROJECT/plan['data_inputs'][arm.get('data_key','train')]['path'];dev=PROJECT/plan['data_inputs']['dev']['path']
 for path in (checkpoint,train,dev):
  if not path.resolve().is_relative_to(PROJECT.resolve()):raise ValueError('Input path outside project')
 args=[sys.executable,str(ROOT/'pair_train.py'),'--data',str(train),'--dev',str(dev),'--checkpoint',str(checkpoint),'--out',str(out),'--expected-layers',str(arm['layers']),'--epochs',str(t['epochs_each']),'--batch',str(arm['batch']),'--micro-batch',str(arm['micro_batch']),'--lr',str(arm['lr']),'--temperature',str(arm.get('temperature',.05)),'--seed',str(t.get('seed',29)),'--eval-every',str(t.get('eval_every',200)),'--source-floor',str(t.get('source_floor',2.)),'--save-every',str(t.get('save_every',100)),'--save-seconds','120','--device','cuda','--amp','bf16']
 if t.get('validation_negatives'):args+=['--validation-negatives']
 if arm.get('use_negatives'):args+=['--use-negatives']
 if arm.get('learn_row_scales'):args+=['--learn-row-scales','--scale-lr-multiplier',str(arm.get('scale_lr_multiplier',100.))]
 if arm.get('train_row_scales_only'):args+=['--train-row-scales-only']
 if t.get('auxiliary_nli'):
  aux_train=PROJECT/plan['data_inputs']['aux_train']['path'];aux_dev=PROJECT/plan['data_inputs']['aux_dev']['path']
  if not all(v.resolve().is_relative_to(PROJECT.resolve()) for v in (aux_train,aux_dev)):raise ValueError('Auxiliary input outside project')
  args+=['--aux-data',str(aux_train),'--aux-dev',str(aux_dev),'--aux-encoder-weight',str(arm['aux_encoder_weight']),'--aux-batch',str(t.get('aux_batch',64)),'--aux-lr-multiplier',str(t.get('aux_lr_multiplier',25.)),'--aux-warmup-steps',str(t.get('aux_warmup_steps',100))]
 if (out/'latest.json').exists():
  info=json.loads((out/'latest.json').read_text());args+=['--resume',str(out/info['file'])]
 return out,args

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());arms=plan['training']['arms'];assert len({v['id'] for v in arms})==len(arms)
 for arm in arms:
  out,args=command(plan,arm)
  if (out/'completed.json').exists():continue
  subprocess.run(args,check=True,cwd=PROJECT)
  config=json.loads((out/'config.json').read_text());latest=json.loads((out/'latest.json').read_text());assert latest['step']==config['total_steps']
  (out/'completed.json').write_text(json.dumps({'step':latest['step'],'precision':config['precision'],'layers':arm['layers']})+'\n')
if __name__=='__main__':main()
