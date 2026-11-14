"""Regrade saved outputs and summarize the local projection mechanism screen."""
import hashlib
import json
from pathlib import Path

from local_projection_probe import DEFAULT_OUT, grade_case, save
from correctness_eval import grade
from summarize_scale_probe import repeated_line_run


def summarize(out=DEFAULT_OUT):
    protocol=json.loads((out/'protocol.json').read_text())
    assert hashlib.sha256((out/'data.json').read_bytes()).hexdigest()==protocol['data_sha256']
    data=json.loads((out/'data.json').read_text())
    manual=json.loads((out/'manual-review.json').read_text())
    specs={c['id']:c for c in data['cases']}
    rows=[];generations={}
    for arm in protocol['arms']+protocol.get('additional_arms',[]):
        p=out/f'{arm}-result.json'
        if not p.exists() or json.loads(p.read_text()).get('status')!='complete':
            raise RuntimeError(f'Arm incomplete: {arm}')
        result=json.loads(p.read_text())
        raw=json.loads((out/f'{arm}-generations.json').read_text())
        assert len(raw['cases'])==len(specs)
        assert {c['id'] for c in raw['cases']}==set(specs), f'Case coverage differs: {arm}'
        grades={}
        for c in raw['cases']:
            if c['grade'] is not None:
                c['grade']=grade_case(c['answer'],specs[c['id']])
            c['repeated_line_run']=repeated_line_run(c['answer'])
            c['combined_loop_flag']=c['loop_flag'] or c['repeated_line_run']>=8
            grades[c['id']]={'grade':c['grade'],'word_loop_flag':c['loop_flag'],
                             'repeated_line_run':c['repeated_line_run'],'combined_loop_flag':c['combined_loop_flag']}
        # Preserve raw runtime results. The final grading adds explicit input
        # immutability checking and separates integer-string sorting format from
        # ordering correctness, uniformly across every arm, without regeneration.
        save(grades,out/f'{arm}-answer-review.json')
        closed=[c for c in raw['cases'] if c['family'] not in ('code_generation','explanation')]
        programs=[c for c in raw['cases'] if c['family']=='code_generation']
        for c in programs:
            if c['grade']['correct']:
                assert 'requirements_pass' in manual['arms'].get(arm,{}).get(c['id'],{}), f"Review requirements: {arm}/{c['id']}"
        for c in raw['cases']:
            if c['family']=='explanation':
                assert 'core_correct' in manual['arms'].get(arm,{}).get(c['id'],{}), f"Review explanation: {arm}/{c['id']}"
        families={}
        for c in closed:
            families.setdefault(c['family'],{'correct':0,'total':0})
            families[c['family']]['total']+=1
            families[c['family']]['correct']+=int(c['grade']['correct'])
        generations[arm]={c['id']:c for c in raw['cases']}
        rows.append({'arm':arm,'parameters':result['parameters'],'stored_scalars':result['stored_scalars'],
            'retained_fraction':result['stored_scalars']/protocol['plan']['source_parameters'],
            'closed_correct':sum(c['grade']['correct'] for c in closed),'closed_total':len(closed),
            'typed_closed_correct':sum(grade(c['answer'],specs[c['id']]['check'])['correct'] for c in closed),
            'runtime_closed_correct':result['generation']['closed_correct'],
            'closed_format_pass':sum(c['grade'].get('format_pass',False) for c in closed),
            'programs_correct':sum(c['grade']['correct'] for c in programs),'programs_total':len(programs),
            'programs_with_reviewed_requirements':sum(c['grade']['correct'] and manual['arms'].get(arm,{}).get(c['id'],{}).get('requirements_pass',False) for c in programs),
            'syntax_pass':sum(c['grade']['syntax_pass'] for c in programs),
            'ended':sum(c['ended'] for c in raw['cases']),'capped':sum(not c['ended'] for c in raw['cases']),
            'word_loop_flags':sum(c['loop_flag'] for c in raw['cases']),
            'loop_flags':sum(c['combined_loop_flag'] for c in raw['cases']),
            'mean_kl':result['validation']['mean_kl'],'target_nll':result['validation']['target_nll'],
            'target_top1':result['validation']['target_top1'],'families':families})
    for row in rows:
        arm=row['arm'];row['paired']={}
        for reference in ('stock','shared-large','shared-extended'):
            if reference not in generations:continue
            gained=[];lost=[]
            for case_id,c in generations[arm].items():
                if c['grade'] is None:continue
                before=generations[reference][case_id]['grade']['correct'];after=c['grade']['correct']
                if after and not before:gained.append(case_id)
                if before and not after:lost.append(case_id)
            row['paired'][reference]={
                'scope':'Closed answers plus program execution tests; algorithm-requirement review is separate.',
                'gained':gained,'lost':lost,
                'closed_gained':[i for i in gained if specs[i]['family']!='code_generation'],
                'closed_lost':[i for i in lost if specs[i]['family']!='code_generation']}
    reconstruction=json.loads((out/'reconstruction.json').read_text())
    reconstruction_extended=json.loads((out/'reconstruction-extended.json').read_text())
    assert len(reconstruction)==len(reconstruction_extended)==3, 'Decoder diagnostics incomplete'
    summary={'protocol':protocol,'arms':rows,'fullwidth':json.loads((out/'fullwidth-control.json').read_text()),
             'roundtrip':json.loads((out/'roundtrip.json').read_text()),
             'rms_fit':json.loads((out/'fit-summary.json').read_text()),
             'reconstruction':reconstruction,
             'reconstruction_extended':reconstruction_extended,
             'linear_decoder':json.loads((out/'linear-decoder-audit.json').read_text()),
             'manual_review':manual,
             'geometry':json.loads((out/'geometry-diagnostics.json').read_text()),
             'extended_calibration':json.loads((out/'extended-calibration-protocol.json').read_text()) if (out/'extended-calibration-protocol.json').exists() else None,
             'grading':'Independent answer grading, sandboxed program tests, interval input-immutability check. programs_correct counts execution passes; programs_with_reviewed_requirements additionally checks explicit algorithm requirements by manual review. Sorting accepts unambiguous integer strings for ordering correctness and records strict type/format separately, uniformly for all arms. Supplemental repetition scan applies the established project rule of >=8 identical nonempty lines; original word-n-gram flags are also retained. Explanation correctness requires manual review.',
             'cloud_spend_usd':0,'limits':'56 closed cases across seven synthetic families and four programming problems: a mechanism screen, not broad intelligence or a 27B result.'}
    save(summary,out/'summary.json')
    files={}
    for p in out.iterdir():
        if p.name=='artifact-manifest.json':continue
        if p.is_file() and p.suffix in ('.json','.txt','.md'):
            files[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        elif p.is_file() and (p.name.endswith('-recipe.pt') or p.name.startswith('decoder-site-')):
            digest=hashlib.sha256()
            with p.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
            files[p.name]={'bytes':p.stat().st_size,'sha256':digest.hexdigest()}
    roundtrip=summary['roundtrip']
    files[roundtrip['checkpoint']]={'bytes':roundtrip['checkpoint_bytes'],'sha256':roundtrip['sha256'],
                                     'verification':'Hashed after exact full-state reload audit.'}
    for name in ('local_projection_models.py','local_projection_probe.py','test_local_projection.py','summarize_local_projection.py',
                 'local_projection_diagnostics.py','local_projection_more_data.py','local_projection_linear.py'):
        p=Path(__file__).parent/name
        files['../../'+name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    save(files,out/'artifact-manifest.json')
    print(json.dumps([{k:r[k] for k in ('arm','retained_fraction','closed_correct','programs_correct','loop_flags','capped','mean_kl')} for r in rows],indent=2))


if __name__=='__main__':summarize()
