"""Freeze completed arms, score observed tasks, and retain a wider-quality check."""
import argparse,concurrent.futures,json,os,shutil,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent

def mteb_scores(path):
 scores={}
 for p in path.rglob('*.json'):
  if p.name=='model_meta.json':continue
  d=json.loads(p.read_text());test=d.get('scores',{}).get('test',[])
  if test:scores[d['task_name']]=test[0]['main_score']*100
 return scores

def main():
 p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--arms',nargs='+',required=True);p.add_argument('--mteb',action='store_true');p.add_argument('--dibs-inner',action='store_true');p.add_argument('--local-frozen',action='store_true');p.add_argument('--wait-complete',action='store_true');a=p.parse_args()
 if not a.dibs_inner:
  # Do not queue a CPU reservation while our remote-work hold owns network:
  # another CPU owner may need network before releasing CPU.
  while True:
   owned=[]
   for state_path in (ROOT/'work').glob('*-run/pod.json'):
    state=json.loads(state_path.read_text())
    if state.get('plan')==a.profile and not state.get('deleted'):owned.append(state_path)
   missing=a.wait_complete and any(not (ROOT/'work'/(a.profile+'-run')/'backups'/(a.profile+'-'+arm)/'completed.json').exists() for arm in a.arms)
   if not owned and not missing:break
   time.sleep(10)
  env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',DIBS_AGENT='quality-'+a.profile)
  subprocess.run(['dibs','run','cpu','--label','Ternary '+a.profile+' cached quality benchmarks','--eta','35m','--',sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--dibs-inner'],check=True,env=env);return
 out=ROOT/a.profile/'results';out.mkdir(parents=True,exist_ok=True);freeze=out/'frozen.json'
 if a.local_frozen and not freeze.exists():raise RuntimeError('Freeze the local candidate grid before scoring')
 if not freeze.exists():subprocess.run([sys.executable,str(ROOT/'freeze_campaign.py'),'--profile',a.profile,'--arms',*a.arms],check=True,stdout=(out/'freeze.log').open('w'))
 frozen=json.loads(freeze.read_text());assert set(a.arms)==set(frozen['models'])
 if a.local_frozen:assert frozen['local_only'] is True
 logs=out/'logs';logs.mkdir(exist_ok=True)
 def worker(arm):
  quality=out/(arm+'-quality.json')
  selection=frozen['models'][arm]
  if selection.get('source_summary'):
   from pair_training_state import digest
   source=Path(selection['source_summary']);previous=json.loads(source.read_text());old_arm=selection['source_arm']
   assert digest(Path(selection['path']))==selection['sha256']==previous['models'][old_arm]['selection']['sha256']
   if selection.get('source_summary_sha256'):assert digest(source)==selection['source_summary_sha256']
   score=json.loads((source.parent/(old_arm+'-quality.json')).read_text());assert score['selection']['sha256']==selection['sha256'] and score['token_cap']==128
   score.update(selection=selection,reused_sha_identical_source=str(source));quality.write_text(json.dumps(score,indent=2)+'\n')
   shutil.copyfile(source.parent/(old_arm+'-rows.json'),out/(arm+'-rows.json'))
   if a.mteb:
    assert len(previous['models'][old_arm]['mteb_12'])==12
    shutil.copytree(source.parent/'mteb'/old_arm/'candidate',out/'mteb'/arm/'candidate',dirs_exist_ok=True)
   print('Reused SHA-identical model scores for',arm,flush=True);return
  if quality.exists():assert json.loads(quality.read_text())['selection']['sha256']==frozen['models'][arm]['sha256']
  else:
   with (logs/(arm+'-quality.log')).open('w') as f:subprocess.run([sys.executable,str(ROOT/'evaluate_data_first.py'),'--freeze',str(freeze),'--output',str(out),'--model',arm],check=True,stdout=f,stderr=subprocess.STDOUT)
  if a.mteb:
   with (logs/(arm+'-mteb.log')).open('a') as f:subprocess.run([str(PROJECT/'.eval-venv/bin/python'),str(ROOT/'evaluate_round2_mteb.py'),'--freeze',str(freeze),'--branch',arm,'--output',str(out/'mteb'/arm)],check=True,stdout=f,stderr=subprocess.STDOUT)
  print('Scored',arm,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(worker,arm) for arm in a.arms]
  for f in futures:f.result()
 from summarize_pair_results import interval
 reference=json.loads((ROOT/'round2/query-rows.json').read_text());results={}
 for arm in a.arms:
  score=json.loads((out/(arm+'-quality.json')).read_text());rows=json.loads((out/(arm+'-rows.json')).read_text());wide=mteb_scores(out/'mteb'/arm/'candidate')
  if a.mteb:
   assert len(wide)==12
   assert abs(wide['STSBenchmark']-score['stsbenchmark_spearman'])<.02
   for task in ('SciFact','NFCorpus'):assert abs(wide[task]-score['tasks'][task]['ndcg10'])<.02
  precision=frozen['models'][arm]['precision']
  prior=reference['original'] if precision=='ternary' else {t:json.loads((ROOT/'analysis'/f'precision-float-{t}-rows.json').read_text()) for t in ('SciFact','NFCorpus')}
  results[arm]={'selection':frozen['models'][arm],'core_retrieval':score['mean_core_retrieval'],'stsbenchmark':score['stsbenchmark_spearman'],'tasks':score['tasks'],'gates':score['gates'],'core_delta':{name:interval(reference[name],rows) for name in ('original','bge_micro','minilm')},'original_reference_precision':'ternary','matched_original_delta':interval(prior,rows),'mteb_12':wide,'mteb_12_mean':statistics.mean(wide.values()) if wide else None}
 contrasts={}
 for precision in ('float','ternary'):
  left='2-'+precision;right='3-'+precision
  if left in results and right in results:
   contrasts[precision]=interval(json.loads((out/(left+'-rows.json')).read_text()),json.loads((out/(right+'-rows.json')).read_text()))
 recipe_contrasts={}
 pairs=[(f'{p}-pairs',f'{p}-hard') for p in ('float','ternary')]
 pairs += [(f'{p}-0.0',f'{p}-{weight}') for p in ('float','ternary') for weight in ('0.1','0.25')]
 pairs += [(f'{p}-repeat',f'{p}-fresh') for p in ('float','ternary')]
 pairs += [('fixed','learned'),('fixed','scales-only'),('3-learned','3-scales-only')]
 for left,right in pairs:
  if left not in results or right not in results:continue
  recipe_contrasts[right+' minus '+left]={'core':interval(json.loads((out/(left+'-rows.json')).read_text()),json.loads((out/(right+'-rows.json')).read_text())),
   'stsbenchmark_delta':results[right]['stsbenchmark']-results[left]['stsbenchmark'],
   'mean12_delta':results[right]['mteb_12_mean']-results[left]['mteb_12_mean'] if a.mteb else None}
 parent_contrasts={};parent_path=ROOT/a.profile/'parent-selection.json'
 if parent_path.exists():
  parents=json.loads(parent_path.read_text());plan=json.loads((ROOT/a.profile/'runpod-job.json').read_text())
  for spec in plan['training']['arms']:
   arm=spec['id']
   if arm not in results:continue
   matches=[v for v in parents.values() if v['prepared_path']==spec['checkpoint']]
   if not matches:continue
   assert len(matches)==1
   parent=matches[0];source=Path(parent['source_summary'])
   from pair_training_state import digest
   assert digest(source)==parent['source_summary_sha256']
   prior=json.loads(source.read_text())['models'][parent['source_arm']]
   assert prior['selection']['sha256']==results[arm]['selection']['initial_checkpoint_sha256']
   parent_contrasts[arm+' minus parent']={
    'parent_summary':str(source),'parent_arm':parent['source_arm'],
    'core':interval(json.loads((source.parent/(parent['source_arm']+'-rows.json')).read_text()),json.loads((out/(arm+'-rows.json')).read_text())),
    'stsbenchmark_delta':results[arm]['stsbenchmark']-prior['stsbenchmark'],
    'mean12_delta':results[arm]['mteb_12_mean']-prior['mteb_12_mean'] if a.mteb else None,
    'scope':'Continuation plus this arm recipe; parent contrast alone does not isolate learned scales from additional training.'}
 resources={'local_only':True,'paid_resource_allocated':False} if a.local_frozen else json.loads((ROOT/'work'/(a.profile+'-run')/'pod.json').read_text())
 summary={'scope':'Frozen checkpoints, observed development tasks. Wider means cover only the fixed 12-task subset. Query confidence intervals do not measure training-seed variation.','reserved_fiqa_scidocs_scored':False,'resources':resources,'continuation_resources':frozen.get('continuation_resources',[]),'models':results,'matched_depth_contrasts':contrasts,'matched_recipe_contrasts':recipe_contrasts,'parent_contrasts':parent_contrasts}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({name:{'core':v['core_retrieval'],'sts':v['stsbenchmark'],'mean12':v['mteb_12_mean']} for name,v in results.items()},indent=2),flush=True)
if __name__=='__main__':main()
