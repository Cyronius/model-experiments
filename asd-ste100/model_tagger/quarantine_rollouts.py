"""Exclude whole groups implicated by the canonical-text split audit, preserving payloads."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import numpy as np
from rollouts import records,sha256,fingerprints,input_paths


def quarantine(parent,pairs,output):
    source=json.loads((parent/'summary.json').read_text())
    validation=json.loads((parent/'validation.json').read_text())
    split_audit=json.loads((parent/'split-validation.json').read_text())
    if not validation['passed'] or validation['summary_sha256']!=sha256(parent/'summary.json') or split_audit['summary_sha256']!=sha256(parent/'summary.json'):
        raise ValueError('parent audit evidence differs')
    groups=set(split_audit['cross_split_groups'])|set(split_audit['conflicting_source_groups'])
    if not groups:raise ValueError('no implicated groups')
    if output.exists() and any(output.iterdir()):raise ValueError('use empty output')
    for name,expected in source['files'].items():
        if sha256(parent/name)!=expected:raise ValueError('parent file changed')
    output.mkdir(parents=True,exist_ok=True)
    oldindex={s:np.load(parent/f'{s}.index.npy',mmap_mode='r') for s in ('train','dev','test')}
    data={s:np.memmap(parent/f'{s}.data.i32',dtype='<i4',mode='r') for s in oldindex}
    newindex={s:np.empty_like(oldindex[s]) for s in oldindex};positions=Counter();offsets=Counter()
    streams={s:(output/f'{s}.data.i32').open('wb') for s in oldindex}
    rejections={r['id']:r for r in records(parent/'rejected.jsonl')}
    counts=Counter();passes=Counter();accepted_kinds=Counter();removed_kinds=Counter();removed_groups=Counter();removed=0
    with gzip.open(output/'pairs.jsonl.gz','wt',encoding='utf-8') as target:
        for row in records(parent/'pairs.jsonl.gz'):
            if row['group'] in groups:
                removed+=1;removed_kinds[row['kind']]+=1;removed_groups[row['group']]+=1
                rejections[row['id']]={'id':row['id'],'kind':row['kind'],'split':row['split'],'reason':'canonical_split_overlap'}
                continue
            split=row['split'];first=row['first_example'];number=row['examples']
            start=int(oldindex[split][first,0]);end=int(oldindex[split][first+number,0]) if first+number<len(oldindex[split]) else len(data[split])
            data[split][start:end].tofile(streams[split])
            current=oldindex[split][first:first+number].copy();current[:,0]+=offsets[split]-start
            newindex[split][positions[split]:positions[split]+number]=current
            row['first_example']=positions[split];positions[split]+=number;offsets[split]+=end-start
            target.write(json.dumps(row,ensure_ascii=False)+'\n');counts[split]+=1;passes[str(row['edit_passes'])]+=1;accepted_kinds[row['kind']]+=1
    for split,stream in streams.items():
        stream.close();np.save(output/f'{split}.index.npy',newindex[split][:positions[split]])
    reasons=Counter();scanned=0
    with (output/'rejected.jsonl').open('w') as stream:
        for path,_ in input_paths(pairs):
            for row in records(path):
                scanned+=1
                rejected=rejections.pop(row['id'],None)
                if rejected:reasons[rejected['reason']]+=1;stream.write(json.dumps(rejected)+'\n')
    if rejections or scanned!=source['scanned']:raise ValueError('quarantine accounting mismatch')
    summary=dict(source)
    slices=dict(source['slices'])
    for kind,n in removed_kinds.items():slices[f'{kind}/accepted']-=n;slices[f'{kind}/rejected']=slices.get(f'{kind}/rejected',0)+n
    summary.update(accepted_pairs=sum(counts.values()),rejected_pairs=sum(reasons.values()),pair_splits=dict(counts),
                   example_splits=dict(positions),edit_passes=dict(passes),reasons=dict(reasons),slices=slices,code_sha256=fingerprints(),
                   quarantine={'groups':sorted(groups),'accepted_pairs_removed':removed,'removed_groups':dict(removed_groups),
                               'parent_summary_sha256':sha256(parent/'summary.json'),'parent_split_audit_sha256':sha256(parent/'split-validation.json'),
                               'script_sha256':sha256(Path(__file__))},files={p.name:sha256(p) for p in output.iterdir() if p.is_file()})
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return {'accepted_pairs':summary['accepted_pairs'],'removed':removed,'groups':len(groups),'splits':dict(counts)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(quarantine(a.parent,a.pairs,a.out),indent=2))
