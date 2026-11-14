"""Post-hoc syntax-tolerant content review; never changes checkpoint selection.

The original grader remains the preregistered score. This supplement accepts a
whole Python literal (e.g. True instead of JSON true) or a whole whitespace-
separated numeric sequence when the expected answer is a numeric list. It does
not search arbitrary prose for an expected answer or reorder list elements.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

from correctness_eval import grade,unwrap
from horizon_data import equivalent


def review(answer,check):
    primary=grade(answer,check)
    if primary['correct']:return {'correct':True,'method':'primary'}
    expected=check['expected'];clean=unwrap(answer).strip()
    if check['kind'] in ('json','codeio','literal','numeric_list'):
        try:
            value=ast.literal_eval(clean)
            if equivalent(value,expected):return {'correct':True,'method':'whole_python_literal'}
        except (ValueError,SyntaxError,RecursionError):pass
        if isinstance(expected,list) and all(type(v) in (int,float) for v in expected):
            if re.fullmatch(r'[+-]?\d+(?:\.\d+)?(?:\s+[+-]?\d+(?:\.\d+)?)+',clean):
                value=[float(v) for v in clean.split()]
                if equivalent(value,expected):return {'correct':True,'method':'whole_numeric_sequence'}
    return {'correct':False,'method':'not_correct_under_supplement'}


def main(data_path,folder,out):
    data=json.loads(data_path.read_text())
    spec={c['id']:c['check'] for c in data['dev']+data['test']}
    paths=sorted(folder.glob('*-dev.json'))+sorted(folder.glob('*-test.json'))+sorted(folder.glob('step-*/dev.json'))
    if (folder/'adapter-last/dev.json').exists():paths.append(folder/'adapter-last/dev.json')
    report={'post_hoc':True,'used_for_selection':False,'data_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),
            'limits':'Only two explicit syntax relaxations; not a comprehensive human correctness judgment.','arms':{}}
    for path in paths:
        rows=json.loads(path.read_text())['cases'];scored=[r for r in rows if r['grade'] is not None]
        outcomes={r['id']:review(r['answer'],spec[r['id']]) for r in scored}
        added=[r['id'] for r in scored if not r['grade']['correct'] and outcomes[r['id']]['correct']]
        report['arms'][str(path.relative_to(folder))]={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'scored':len(scored),'primary_correct':sum(r['grade']['correct'] for r in scored),
            'supplement_correct':sum(v['correct'] for v in outcomes.values()),
            'additional_content_correct':{k:outcomes[k]['method'] for k in added},'outcomes':outcomes}
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{n:v[n] for n in ['scored','primary_correct','supplement_correct']} for k,v in report['arms'].items()},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--folder',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();main(args.data,args.folder,args.out)
