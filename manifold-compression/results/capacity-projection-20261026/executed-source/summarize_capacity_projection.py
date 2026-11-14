"""Summarize fixed-endpoint matched arms without choosing a checkpoint."""
import json
from collections import Counter
from capacity_projection import OUT, ARMS
from local_projection_probe import save

def summarize(data):
    groups={}
    for name,pred in [('math',lambda c:c['family'].startswith('math_')),('retention',lambda c:c['family'] not in ['code_generation','explanation'] and not c['family'].startswith('math_')),('programs',lambda c:c['family']=='code_generation')]:
        rows=[c for c in data['cases'] if pred(c)]
        groups[name]={'correct':sum(c['grade']['correct'] for c in rows),'total':len(rows)}
    return {**data['summary'],'groups':groups}

def paired(before,after):
    b={c['id']:c for c in before['cases']};gains=[];losses=[];retained=[]
    for a in after['cases']:
        if a['grade'] is None:continue
        old=b[a['id']]['grade']['correct'];new=a['grade']['correct']
        if new and not old:gains.append(a['id'])
        if old and not new:losses.append(a['id'])
        if old and new:retained.append(a['id'])
    return {'gains':gains,'losses':losses,'retained':retained}

def main():
    data={};result={'arms':{},'baseline_equality':{},'cloud_spend':0}
    for arm in ARMS:
        d=OUT/arm;data[arm]={step:json.loads((d/f'test-{step}-content-audit.json').read_text()) for step in [0,8]}
        result['arms'][arm]={'model':json.loads((d/'model.json').read_text()),'baseline':summarize(data[arm][0]),'final':summarize(data[arm][8]),'changes':paired(data[arm][0],data[arm][8]),'train_fit_ce':json.loads((d/'result.json').read_text())['train_fit_ce']}
    for architecture in ['projected','stock']:
        a,b=(data[architecture+'-r'+str(r)][0] for r in [16,64]);aa={c['id']:c for c in a['cases']};bb={c['id']:c for c in b['cases']}
        equal=all(all(aa[i][k]==bb[i][k] for k in ['answer','ended','generated_tokens','grade']) for i in aa)
        result['baseline_equality'][architecture]=equal
        if not equal:raise ValueError('Baseline rank arms differ; investigate before interpreting intervention')
    interactions={}
    for group in ['math','retention','programs']:
        score=lambda a:result['arms'][a]['final']['groups'][group]['correct']
        projected_delta=score('projected-r64')-score('projected-r16');stock_delta=score('stock-r64')-score('stock-r16')
        interactions[group]={'projected_rank_gain':projected_delta,'stock_rank_gain':stock_delta,'difference_in_rank_gains':projected_delta-stock_delta,'stock_minus_projected_r16':score('stock-r16')-score('projected-r16'),'stock_minus_projected_r64':score('stock-r64')-score('projected-r64')}
    result['capacity_interaction']=interactions
    closed=lambda d:{'cases':[c for c in d['cases'] if c['family'] not in ['explanation','code_generation']]}
    result['source_reference_closed']={arm:paired(closed(data['stock-r16'][0]),closed(data[arm][8])) for arm in ARMS}
    save(result,OUT/'summary.json');print(json.dumps({k:v for k,v in result.items() if k!='arms'},indent=2))
    for a,r in result['arms'].items():print(a, json.dumps({'baseline':r['baseline'],'final':r['final']}))

if __name__=='__main__':main()
