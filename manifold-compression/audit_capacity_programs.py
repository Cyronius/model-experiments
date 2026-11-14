"""Supplementary edge-case audit; declared after inspecting stock-r16 final code."""
import json
from capacity_projection import OUT,ARMS
from local_projection_probe import grade_program,save

TESTS={
 'ctest-dot':[[{'left':[-2],'right':[3]},-6],[{'left':[0,2,4],'right':[4,-3,1]},-2],[{'left':[],'right':[]},0]],
 'ctest-unique':[[{'text':'z'},'z'],[{'text':'aabb'},None],[{'text':'ññø'},'ø'],[{'text':'abacb'},'c'],[{'text':''},None]],
 'ctest-streak':[[{'values':[1]},1],[{'values':[0,1,0]},1],[{'values':[-1,2]},1],[{'values':[-1,2,0,3,4,0,5]},2],[{'values':[-3,0,-2]},0],[{'values':[1,1]},2]],
 'ctest-overlap':[[{'text':'aaa','pattern':'aaaa'},0],[{'text':'a','pattern':'a'},1],[{'text':'banana','pattern':'ana'},2],[{'text':'','pattern':'a'},0],[{'text':'aaaa','pattern':'aa'},3]],
}

def main():
 assert json.loads((OUT/'program-edge-audit-protocol.json').read_text())['tests']==TESTS
 result={'scope':'Supplementary post-hoc edge-case audit, applied to every arm and baseline. Original primary scores unchanged. Passing these checks still is not proof of correctness.','arms':{}}
 for arm in ARMS:
  result['arms'][arm]={}
  for step in [0,8]:
   raw=json.loads((OUT/arm/f'test-{step}.json').read_text());rows=[]
   for c in raw['cases']:
    if c['id'] not in TESTS:continue
    grade=grade_program(c['answer'],TESTS[c['id']])
    rows.append({'id':c['id'],'original_grade':c['grade'],'edge_grade':grade,'passes_both':c['grade']['correct'] and grade['correct']})
   result['arms'][arm][step]={'cases':rows,'passes_both':sum(c['passes_both'] for c in rows),'total':len(rows)}
 result['rank_truncation']={}
 for variant in ['full-rank-control','rank16-truncated']:
  path=OUT/'rank-truncation'/variant/'test.json'
  if not path.exists():continue
  rows=[]
  for c in json.loads(path.read_text())['cases']:
   if c['id'] not in TESTS:continue
   grade=grade_program(c['answer'],TESTS[c['id']]);rows.append({'id':c['id'],'original_grade':c['grade'],'edge_grade':grade,'passes_both':c['grade']['correct'] and grade['correct']})
  result['rank_truncation'][variant]={'cases':rows,'passes_both':sum(c['passes_both'] for c in rows),'total':len(rows)}
 save(result,OUT/'program-edge-audit.json')
 print(json.dumps({a:{s:r['passes_both'] for s,r in v.items()} for a,v in result['arms'].items()},indent=2))

if __name__=='__main__':main()
