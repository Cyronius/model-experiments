"""Post-hoc executable audit of the withheld interval-merging generation.

This never participates in training or checkpoint selection. Model-generated
code executes only through the networkless, filesystem-isolated verifier.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

from horizon_data import run_reference


CASES=[([],[]),([[1,3],[2,6],[8,10],[15,18]],[[1,6],[8,10],[15,18]]),
       ([[1,2],[2,4]],[[1,4]]),([[1,10],[2,3]],[[1,10]]),
       ([[8,10],[2,6],[1,3]],[[1,6],[8,10]]),
       ([[1,2],[1,2],[5,5]],[[1,2],[5,5]])]


def audit(answer):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',answer,re.S)
    errors=[]
    for block in blocks:
        try:tree=ast.parse(block)
        except SyntaxError as error:
            errors.append({'error':'SyntaxError','line':error.lineno});continue
        functions=[n.name for n in tree.body if isinstance(n,ast.FunctionDef)]
        if not functions:continue
        # Keep imports and definitions, excluding top-level demonstrations.
        definitions=ast.Module(body=[n for n in tree.body if isinstance(n,
            (ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef))],type_ignores=[])
        code=ast.unparse(definitions)
        harness='''
def main_solution():
    cases=CASES_LITERAL
    results=[]
    for values,expected in cases:
        inp=[list(x) for x in values]
        before=[list(x) for x in inp]
        try:
            got=FUNCTION_NAME(inp)
            normalized=[list(x) for x in got]
            results.append({'correct':normalized==expected,'input_unchanged':inp==before,'actual':normalized})
        except BaseException as error:
            results.append({'correct':False,'input_unchanged':inp==before,'error':type(error).__name__})
    return results
'''.replace('CASES_LITERAL',repr(CASES)).replace('FUNCTION_NAME',functions[0])
        result=run_reference(code+'\n'+harness,{})
        checks=result['result'] if result else None
        return {'function':functions[0],'source_sha256':hashlib.sha256(block.encode()).hexdigest(),
            'parse_errors_in_earlier_blocks':errors,'checks':checks,
            'all_passed':bool(checks and len(checks)==len(CASES) and all(c['correct'] and c['input_unchanged'] for c in checks))}
    return {'all_passed':False,'parse_errors':errors,'reason':'No parseable Python function found'}


def main(folder,out):
    report={'post_hoc':True,'used_for_selection':False,
            'case_id':'test-long-3','test_cases':CASES,'arms':{}}
    for arm in ['stock','full_width','unhealed_75','healed-selected']:
        path=folder/(arm+'-test.json')
        if not path.exists():continue
        row=next((r for r in json.loads(path.read_text())['cases'] if r['id']=='test-long-3'),None)
        if row:
            report['arms'][arm]={'generation_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                 **audit(row['answer'])}
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({arm:r['all_passed'] for arm,r in report['arms'].items()}))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folder',type=Path);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();main(args.folder,args.out)
