"""Uniform numeric-content correction declared before first checkpoint results.

The legacy parser can prefer an earlier '=' over an explicit final numeric
line, and cannot parse a mixed number. Preserve its result for transparency.
No expected value is consulted while extracting the model's stated answer.
"""
from fractions import Fraction
import json
import re
from pathlib import Path
from correctness_eval import numeric_value
from local_projection_probe import save


def final_numeric(answer):
    lines=[s.strip() for s in answer.strip().splitlines() if s.strip()]
    if not lines:raise ValueError('Empty answer')
    line=lines[-1].strip(' *$`').rstrip('.').strip()
    if line.startswith(r'\(') and line.endswith(r'\)'):line=line[2:-2].strip()
    line=re.sub(r'\\(?:d?frac)\{([+-]?\d+)\}\{(\d+)\}',r'\1/\2',line)
    boxed=re.fullmatch(r'\\boxed\{([^{}]+)\}',line)
    if boxed:line=boxed[1].strip()
    if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?',line):line=line.replace(',','')
    mixed=re.fullmatch(r'([+-]?\d+)\s+(\d+)\s*/\s*(\d+)',line)
    if mixed:
        whole,numerator,denominator=mixed.groups()
        sign=-1 if whole.startswith('-') else 1
        return sign*(abs(Fraction(whole))+Fraction(int(numerator),int(denominator))),'standalone_mixed_number'
    if re.fullmatch(r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*/\s*\d+)?',line):
        return Fraction(line.replace(' ','')),'standalone_final_number'
    return numeric_value(answer),'legacy_explicit_answer_parser'


def audit_file(path,cases):
    data=json.loads(Path(path).read_text());specs={c['id']:c for c in cases}
    changes=[]
    for c in data['cases']:
        check=specs[c['id']]['check']
        if not check or check['kind']!='numeric':continue
        c['original_grade']=dict(c['grade'])
        try:
            value,reason=final_numeric(c['answer'])
            c['grade']={'correct':value==Fraction(str(check['expected'])),'parsed':True,'format_pass':c['answer'].strip()==str(check['expected'])}
            c['numeric_audit']={'method':reason,'value':str(value)}
        except (ValueError,ZeroDivisionError,TypeError):
            c['grade']={'correct':False,'parsed':False,'format_pass':False}
            c['numeric_audit']={'method':'unparseable'}
        if c['grade']!=c['original_grade']:changes.append(c['id'])
    closed=[c for c in data['cases'] if c['family'] not in ['explanation','code_generation']]
    data['summary']['closed_correct']=sum(c['grade']['correct'] for c in closed)
    data['numeric_audit_changes']=changes
    return data


def select(out):
    out=Path(out);data=json.loads((out/'data.json').read_text());audits={}
    for step in [0,8,16]:
        a=audit_file(out/f'dev-{step}.json',data['dev']);audits[step]=a['summary'];save(a,out/f'dev-{step}-content-audit.json')
    baseline=audits[0]
    eligible={s:(a['closed_correct']+a['programs_correct'],-a['loop_flags'],a['ended'],-s) for s,a in audits.items()
              if a['loop_flags']<=baseline['loop_flags'] and a['ended']>=baseline['ended']-1}
    winner=max(eligible,key=eligible.get)
    provisional=json.loads((out/'selection.json').read_text())['selected_stage_step']
    result={'selected_stage_step':winner,'selected_total_step':16+winner,'provisional_stage_step':provisional,'winner_unchanged':winner==provisional,'dev':audits,'eligible_keys':eligible}
    save(result,out/'selection-content-audited.json')
    return result

if __name__=='__main__':
    from curriculum_healing import OUT
    print(json.dumps(select(OUT),indent=2))
