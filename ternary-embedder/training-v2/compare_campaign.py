"""Compare completed frozen development results without opening reserved labels."""
import datetime,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def wide(path):
 scores={}
 for p in path.rglob('*.json'):
  if p.name=='model_meta.json':continue
  d=json.loads(p.read_text());test=d.get('scores',{}).get('test',[])
  if test:scores[d['task_name']]=test[0]['main_score']*100
 return scores

def main():
 baseline=json.loads((ROOT/'round2/quality.json').read_text())['models'];refs={}
 for name in ('bge_micro','minilm'):
  scores=wide(ROOT/'campaign/mteb-references'/name);assert len(scores)==12
  refs[name]={'mean12':statistics.mean(scores.values()),'core':baseline[name]['mean_core_retrieval'],'tasks':baseline[name]['tasks']}
 rows=[]
 for profile in ('depth3','batch1024','nli-aux','hardneg','science-float','science-ternary','scales','float-blends','ternary-blends','projection','projection-blend','human-mix','q4-mse','q4-groups','role-projection','final-float-blends','delivery'):
  path=ROOT/profile/('family-results/summary.json' if profile=='delivery' else 'results/summary.json')
  if not path.exists():continue
  report=json.loads(path.read_text())
  for name,m in report['models'].items():
   if m['mteb_12_mean'] is None:continue
   precision=m['selection']['precision'];each_core=all(m['tasks'][t]['ndcg10']>max(v['tasks'][t]['ndcg10'] for v in refs.values()) for t in ('SciFact','NFCorpus'));wider=m['mteb_12_mean']>max(v['mean12'] for v in refs.values())
   row={'name':profile+'/'+name,'precision':precision,'layers':m['selection']['layers'],'core':m['core_retrieval'],'SciFact':m['tasks']['SciFact']['ndcg10'],'NFCorpus':m['tasks']['NFCorpus']['ndcg10'],'STSBenchmark':m['stsbenchmark'],'mean12':m['mteb_12_mean'],'beats_both_refs_each_core':each_core,'beats_both_refs_mean12':wider,'internal_guard_passed':m['selection']['eligible_for_promotion'],'selection':m['selection']}
   row['fixed_subset_target_passed']=each_core and wider and row['internal_guard_passed'];rows.append(row)
 rows.sort(key=lambda r:(r['precision'],-r['mean12'],-r['core']));out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Observed development scores only; fixed12-task subset, not full MTEB. Selecting across experiments is model development; reserved labels and training-seed replication remain separate confirmation.','references':refs,'models':rows,'reserved_labels_opened_by_this_script':False}
 (ROOT/'campaign/comparison.json').write_text(json.dumps(out,indent=2)+'\n')
 lines=['# Frozen campaign comparisons','','Observed development results,128-token cap. The12-task mean is not full MTEB.','', '| Candidate | Core retrieval | SciFact | NFCorpus | STSBenchmark |12-task mean | Both core references |','|---|---:|---:|---:|---:|---:|---|']
 for r in rows:lines.append(f"| {r['name']} | {r['core']:.2f} | {r['SciFact']:.2f} | {r['NFCorpus']:.2f} | {r['STSBenchmark']:.2f} | {r['mean12']:.2f} | {'yes' if r['beats_both_refs_each_core'] else 'no'} |")
 lines+=['','Reference core means: bge-micro46.21, MiniLM46.92. Wider means: bge-micro70.90, MiniLM71.48. The strict core gate requires a higher score on each task individually, not only their average.','', 'This script does not score FiQA/SCIDOCS judgments or establish seed stability. The separate frozen Q8 confirmation is recorded in [delivery results](../delivery/RESULTS.md). Default deployed weights are unchanged.',''];(ROOT/'campaign/COMPARISON.md').write_text('\n'.join(lines))
 print(json.dumps([{k:v for k,v in r.items() if k in ('name','precision','core','mean12','fixed_subset_target_passed')} for r in rows],indent=2))
if __name__=='__main__':main()
