"""Audit official human STS train/validation splits; never ingest its test split."""
import collections
import json
from pathlib import Path
import numpy as np
from huggingface_hub import HfApi,hf_hub_download
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from broad_pairs import identity
from data_first_build import visible_key
from pair_training_state import atomic_json,digest

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
OUT=ROOT/'work/human-similarity'
REVISION='ab7a5ac0e35aa22088bdcf23e7fd99b220e53308'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'audit.json').exists():
        print('Human-similarity data already audited.',flush=True);return
    tokenizer=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
    cache=ROOT/'work/broad-data/reference-identities.npz'
    with np.load(cache,allow_pickle=False) as arrays:
        blocked={v.tobytes() for v in arrays['visible']}
        raw={bytes(v).decode('ascii') for v in arrays['raw']}
    qa=ROOT/'work/broad-data/ready/dev.jsonl'
    for line in qa.open():
        row=json.loads(line)
        blocked.update(bytes.fromhex(row[k]) for k in ('query_visible_key','positive_visible_key'))
    files={};rows={};counts=collections.defaultdict(collections.Counter)
    for split in ('train','validation'):
        name='data/'+split+'-00000-of-00001.parquet'
        path=Path(hf_hub_download('sentence-transformers/stsb',name,repo_type='dataset',revision=REVISION))
        files[split]=dict(path=name,sha256=digest(path))
        candidates=[];seen={};conflicts=set()
        source=pq.read_table(path).to_pylist()
        for index,row in enumerate(source):
            counts[split]['raw']+=1
            a,b=row['sentence1'],row['sentence2'];score=float(row['score'])
            assert 0<=score<=1
            ids=tokenizer([a,b],truncation=False,verbose=False)['input_ids']
            if max(map(len,ids))>128:counts[split]['overlength']+=1;continue
            keys=[visible_key(v) for v in ids]
            if any(k in blocked for k in keys) or any(identity(t) in raw for t in (a,b)):
                counts[split]['benchmark_or_qa_dev_input_overlap']+=1;continue
            if keys[0]==keys[1]:counts[split]['identical_visible_inputs']+=1;continue
            pair=tuple(sorted(keys))
            if pair in seen:
                counts[split]['duplicate_pair']+=1
                if abs(seen[pair]-score)>1e-8:conflicts.add(pair)
                continue
            seen[pair]=score
            candidates.append(dict(sentence1=a,sentence2=b,score=score,visible1=keys[0].hex(),visible2=keys[1].hex(),source_row=index))
        rows[split]=[r for r in candidates if tuple(sorted((bytes.fromhex(r['visible1']),bytes.fromhex(r['visible2'])))) not in conflicts]
        counts[split]['conflicting_pair_groups']=len(conflicts)
    dev_inputs={r[k] for r in rows['validation'] for k in ('visible1','visible2')}
    train=[]
    for row in rows['train']:
        if row['visible1'] in dev_inputs or row['visible2'] in dev_inputs:
            counts['train']['semantic_dev_input_overlap']+=1
        else:train.append(row)
    rows['train']=train
    outputs={}
    for split,items in rows.items():
        path=OUT/(split+'.jsonl')
        with path.open('w') as output:
            for row in items:output.write(json.dumps(row,ensure_ascii=False)+'\n')
        counts[split]['retained']=len(items);outputs[split]=digest(path)
    assert not {r[k] for r in rows['train'] for k in ('visible1','visible2')} & dev_inputs
    atomic_json(OUT/'audit.json',dict(source='sentence-transformers/stsb',revision=REVISION,
        human_labels=True,score_range=[0,1],test_split_downloaded=False,token_cap=128,
        qa_retention_dev_sha256=digest(qa),benchmark_exclusion_cache_sha256=digest(cache),
        source_files=files,counts={k:dict(v) for k,v in counts.items()},output_sha256=outputs,
        semantic_train_dev_visible_overlap=0,teacher_targets=False,
        limitations='Known benchmark and QA-dev input identities excluded. Historical pretraining exposure remains unknown. Held semantic scores select the diagnostic head; existing benchmark test scores remain observed development.'))
    print(json.dumps({k:dict(v) for k,v in counts.items()},indent=2),flush=True)


if __name__=='__main__':main()
