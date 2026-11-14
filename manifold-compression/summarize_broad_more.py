"""Compare the completed broad+more arm with the frozen 2x2 baselines."""
import argparse
import json
from pathlib import Path

from local_projection_broad_more import OUT,BASE,ARM,sha
from local_projection_probe import save,grade_case
from correctness_eval import grade
from summarize_scale_probe import repeated_line_run


def summarize(out=OUT,base=BASE):
    protocol=json.loads((out/'protocol.json').read_text())
    assert sha(out/'data.json')==protocol['data_sha256']
    assert sha(base/'data.json')==protocol['base_data_sha256']
    data=json.loads((out/'data.json').read_text())
    original=json.loads((base/'data.json').read_text())
    assert data['validation']==original['validation'] and data['cases']==original['cases']
    specs={c['id']:c for c in data['cases']}
    result=json.loads((out/f'{ARM}-result.json').read_text())
    assert result['status']=='complete' and result['cache']['numerical_pass']
    raw=json.loads((out/f'{ARM}-generations.json').read_text())['cases']
    assert len(raw)==len(specs) and {c['id'] for c in raw}==set(specs)
    manual=json.loads((out/'manual-review.json').read_text())
    reviewed={}
    for c in raw:
        c['grade']=grade_case(c['answer'],specs[c['id']])
        c['repeated_line_run']=repeated_line_run(c['answer'])
        c['combined_loop_flag']=c['loop_flag'] or c['repeated_line_run']>=8
        if c['family']=='explanation':assert 'core_correct' in manual[c['id']]
        if c['family']=='code_generation' and c['grade']['correct']:
            assert 'requirements_pass' in manual[c['id']]
        reviewed[c['id']]={k:c[k] for k in ('grade','loop_flag','repeated_line_run','combined_loop_flag')}
    save(reviewed,out/f'{ARM}-answer-review.json')
    closed=[c for c in raw if c['family'] not in ('explanation','code_generation')]
    programs=[c for c in raw if c['family']=='code_generation']
    families={}
    for c in closed:
        item=families.setdefault(c['family'],{'correct':0,'total':0})
        item['total']+=1;item['correct']+=int(c['grade']['correct'])
    row={'arm':ARM,'parameters':result['parameters'],'stored_scalars':result['stored_scalars'],
         'retained_fraction':result['stored_scalars']/protocol['plan']['source_parameters'],
         'closed_correct':sum(c['grade']['correct'] for c in closed),'closed_total':len(closed),
         'typed_closed_correct':sum(grade(c['answer'],specs[c['id']]['check'])['correct'] for c in closed),
         'closed_format_pass':sum(c['grade']['format_pass'] for c in closed),
         'programs_correct':sum(c['grade']['correct'] for c in programs),'programs_total':len(programs),
         'programs_with_reviewed_requirements':sum(c['grade']['correct'] and manual[c['id']]['requirements_pass'] for c in programs if c['grade']['correct']),
         'explanations_core_correct':sum(manual[c['id']]['core_correct'] for c in raw if c['family']=='explanation'),
         'ended':sum(c['ended'] for c in raw),'capped':sum(not c['ended'] for c in raw),
         'word_loop_flags':sum(c['loop_flag'] for c in raw),'loop_flags':sum(c['combined_loop_flag'] for c in raw),
         'mean_kl':result['validation']['mean_kl'],'target_nll':result['validation']['target_nll'],
         'families':families,'paired':{}}
    old=json.loads((base/'summary.json').read_text())
    names=['stock','shared-large','shared-extended','shared-broad']
    baselines=[next(r for r in old['arms'] if r['arm']==name) for name in names]
    for baseline in baselines:
        name=baseline['arm']
        previous=json.loads((base/f'{name}-answer-review.json').read_text())
        gain=[];loss=[]
        for c in closed:
            before=previous[c['id']]['grade']['correct'];after=c['grade']['correct']
            if after and not before:gain.append(c['id'])
            if before and not after:loss.append(c['id'])
        row['paired'][name]={'closed_gained':gain,'closed_lost':loss,'net':len(gain)-len(loss)}
    scores={r['arm']:r['closed_correct'] for r in baselines+[row]}
    summary={'protocol':protocol,'baselines':baselines,'combined':row,'manual_review':manual,
             'fit':json.loads((out/'fit-summary.json').read_text()),
             'observed_interaction_in_correct_answers':scores[ARM]-scores['shared-broad']-scores['shared-extended']+scores['shared-large'],
             'interaction_scope':'Difference of observed gains in this small fixed task set; not a statistical estimate of general intelligence.',
             'controls':{'inherited_fullwidth_and_reload':str(base),'fresh_cache':result['cache'],
                         'incremental_unit_test':(out/'incremental-test.txt').read_text()},
             'grading':'Same content grading as prior experiment. Execution tests and full program requirements are separate. Repetition flags require qualitative review; caps alone do not imply looping.',
             'cloud_spend_usd':0}
    save(summary,out/'summary.json')
    print(json.dumps({'combined':row,'interaction':summary['observed_interaction_in_correct_answers']},indent=2))


def manifest(out=OUT,base=BASE):
    files={}
    for p in out.iterdir():
        if p.name=='artifact-manifest.json' or not p.is_file():continue
        if p.suffix not in ('.json','.txt','.md','.pt'):continue
        files[p.name]={'bytes':p.stat().st_size,'sha256':sha(p)}
        if p.is_symlink():files[p.name]['source']=str(p.resolve())
    for name in ('broad-states.pt','shared-broad-recipe.pt','summary.json','artifact-manifest.json'):
        p=base/name
        files[str(p)]={'bytes':p.stat().st_size,'sha256':sha(p),'role':'Original frozen comparison or reused calibration'}
    for name in ('local_projection_broad_more.py','test_local_projection_broad_more.py','summarize_broad_more.py',
                 'local_projection_models.py','local_projection_probe.py','correctness_eval.py','horizon_data.py'):
        p=Path(__file__).parent/name
        files['../../'+name]={'bytes':p.stat().st_size,'sha256':sha(p)}
    save(files,out/'artifact-manifest.json')
    print(f'Hashed {len(files)} artifacts and dependencies')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['summarize','manifest'])
    args=ap.parse_args();globals()[args.stage]()
