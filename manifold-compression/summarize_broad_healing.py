"""Summarize frozen outcomes; does not choose a checkpoint."""
import json
import ast
from collections import defaultdict
from pathlib import Path
from broad_healing import OUT, ROOT, sha
from local_projection_probe import save, extract_program, grade_program
from summarize_scale_probe import repeated_line_run


def audit(path):
    data=json.loads(path.read_text());families=defaultdict(lambda:{'correct':0,'total':0})
    review={}
    for c in data['cases']:
        if c['grade'] is not None:
            families[c['family']]['total']+=1
            families[c['family']]['correct']+=int(c['grade']['correct'])
        line_run=repeated_line_run(c['answer'])
        review[c['id']]={'ended':c['ended'],'word_loop_flag':c['loop_flag'],'repeated_line_run':line_run,
                         'combined_loop_flag':c['loop_flag'] or line_run>=8,'grade':c['grade']}
    return {'summary':data['summary'],'families':dict(families),'combined_loop_flags':sum(r['combined_loop_flag'] for r in review.values()),'review':review}


def positional_audit(path, cases):
    """Secondary diagnostic only: allow argument renaming, not code repair.

    Added after observing the development chunking interface failure. Applied
    uniformly to baseline and repaired outputs; never used for selection.
    """
    specs={c['id']:c for c in cases if c['family']=='code_generation'}
    result={}
    class Rename(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            self.generic_visit(node)
            if node.name=='main_solution':node.name='_broad_healing_positional_candidate'
            return node
        def visit_Name(self,node):
            if node.id=='main_solution':node.id='_broad_healing_positional_candidate'
            return node
    for c in json.loads(path.read_text())['cases']:
        if c['id'] not in specs:continue
        spec=specs[c['id']];tests=spec['check']['tests']
        names=list(tests[0][0])
        assert all(list(inputs)==names for inputs,_ in tests)
        try:
            tree=Rename().visit(ast.parse(extract_program(c['answer'])))
            code=ast.unparse(tree)
            code+='\ndef main_solution(**inputs):\n    return _broad_healing_positional_candidate(*(inputs[k] for k in '+repr(names)+'))\n'
            result[c['id']]=grade_program(code,tests,c['id']=='code-intervals')
        except SyntaxError:
            result[c['id']]={'correct':False,'syntax_pass':False,'passed_tests':0,'tests':len(tests)}
    return result


def main():
    result=json.loads((OUT/'result.json').read_text())
    files=['dev-0','dev-8','dev-16','test-0','test-selected','legacy-selected']
    audits={n:audit(OUT/(n+'.json')) for n in files}
    for name,a in audits.items():save(a,OUT/(name+'-audit.json'))
    data=json.loads((OUT/'data.json').read_text())
    positional={n:positional_audit(OUT/(n+'.json'),data['dev'] if n.startswith('dev') else data['test'] if n.startswith('test') else data['legacy']) for n in files}
    positional['legacy-0']=positional_audit(ROOT/'results/local-projection-20261025/shared-broad-generations.json',data['legacy'])
    save({'post_hoc_diagnostic':True,'selection_unchanged':True,'results':positional},OUT/'positional-code-audit.json')
    before=json.loads((OUT/'test-0.json').read_text())['cases'];after=json.loads((OUT/'test-selected.json').read_text())['cases']
    b={c['id']:c for c in before};a={c['id']:c for c in after}
    gains=[];losses=[]
    for id,c in b.items():
        if c['grade'] is None:continue
        if a[id]['grade']['correct'] and not c['grade']['correct']:gains.append(id)
        if c['grade']['correct'] and not a[id]['grade']['correct']:losses.append(id)
    result['fresh_gains']=gains;result['fresh_losses']=losses
    result['combined_loop_counts']={n:a['combined_loop_flags'] for n,a in audits.items()}
    save(result,OUT/'summary.json')
    manifest={}
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='artifact-manifest.json':manifest[str(p.relative_to(OUT))]={'bytes':p.stat().st_size,'sha256':sha(p)}
    save(manifest,OUT/'artifact-manifest.json')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
