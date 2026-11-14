"""Paired task-family reporting; checkpoint selection is already frozen."""
from collections import defaultdict
import json
from curriculum_healing import OUT, PARENT
from local_projection_probe import save


def read(path):return json.loads(path.read_text())

def audit(path):
    data=read(path);families=defaultdict(lambda:{'correct':0,'total':0});math={'correct':0,'total':0};retention={'correct':0,'total':0}
    for c in data['cases']:
        if c['grade'] is None:continue
        fam=families[c['family']];fam['total']+=1;fam['correct']+=int(c['grade']['correct'])
        if c['family']=='code_generation':continue
        group=math if c['family'].startswith('math_') else retention
        group['total']+=1;group['correct']+=int(c['grade']['correct'])
    return {'summary':data['summary'],'families':dict(families),'new_math':math,'retention':retention}

def paired(before,after):
    b={c['id']:c for c in read(before)['cases']};a={c['id']:c for c in read(after)['cases']}
    result={k:{'gains':[],'losses':[]} for k in ['closed','program_io']}
    assert b.keys()==a.keys()
    for id,c in b.items():
        if c['grade'] is None:continue
        k='program_io' if c['family']=='code_generation' else 'closed'
        if a[id]['grade']['correct'] and not c['grade']['correct']:result[k]['gains'].append(id)
        if c['grade']['correct'] and not a[id]['grade']['correct']:result[k]['losses'].append(id)
    return result

def main():
    result=read(OUT/'result.json')
    result['runner_metrics_before_content_audit']={k:result[k] for k in ['dev','fresh_before','fresh_after','legacy_before','legacy_after']}
    result['audits']={n:audit(OUT/(n+'-content-audit.json')) for n in ['dev-0','dev-8','dev-16','test-0','test-selected','legacy-selected','legacy-parent','stock-test','legacy-stock']}
    result['dev']={str(s):result['audits'][f'dev-{s}']['summary'] for s in [0,8,16]}
    for field,name in [('fresh_before','test-0'),('fresh_after','test-selected'),('legacy_before','legacy-parent'),('legacy_after','legacy-selected'),('stock_fresh','stock-test'),('stock_legacy','legacy-stock')]:
        result[field]=result['audits'][name]['summary']
    result['selection_content_audit']=read(OUT/'selection-content-audited.json')
    result['fresh_changes']=paired(OUT/'test-0-content-audit.json',OUT/'test-selected-content-audit.json')
    result['legacy_changes']=paired(OUT/'legacy-parent-content-audit.json',OUT/'legacy-selected-content-audit.json')
    save(result,OUT/'summary.json')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
