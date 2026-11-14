"""Recover TRAIN positives using original short-answer aliases, never test labels."""
import collections
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('RAYON_NUM_THREADS','2')
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer
from broad_pairs import clean,identity,answer_window,prepare_batch
from broad_data_fetch import atomic,digest
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent


def batches(repo,revision,files,columns):
    fs=HfFileSystem(token=False)
    for filename in files:
        assert '/train-' in filename
        with fs.open('datasets/'+repo+'@'+revision+'/'+filename,'rb',block_size=1024*1024) as f:
            pf=pq.ParquetFile(f)
            for group in range(pf.num_row_groups):
                yield filename,group,pf.read_row_group(group,columns=columns).to_pylist()


def main():
    lock=json.loads((ROOT/'broad-data/answer-annotations.lock.json').read_text())
    source_audit=json.loads((ROOT/'broad-data/source-audit.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
    definitions=[('nq-recovered','google-research-datasets/nq_open','nq_open/','sentence-transformers/natural-questions'),
                 ('trivia-recovered','mandarjoshi/trivia_qa','rc.nocontext/','sentence-transformers/trivia-qa')]
    for slug,annotation_repo,prefix,pair_repo in definitions:
        annotation=lock[annotation_repo];pair=lock.get(pair_repo) or source_audit[pair_repo]
        files=[f for f in annotation['files'] if f.startswith(prefix)]
        aliases={}
        for _,_,rows in batches(annotation_repo,annotation['revision'],files,['question','answer']):
            for row in rows:
                values=row['answer']
                if isinstance(values,dict):values=values.get('aliases',[])+[values.get('value','')]
                values=[clean(s) for s in values if isinstance(s,str) and len(clean(s))>=3]
                if values:aliases.setdefault(identity(row['question']),set()).update(values)
        output=ROOT/'work/broad-data/candidates'/slug;output.mkdir(parents=True,exist_ok=True)
        pair_files=[f for f in pair.get('files',pair.get('parquet_files',[])) if f.startswith('pair/') and '/train-' in f]
        provenance=dict(pair_repo=pair_repo,pair_revision=pair['revision'],annotation_repo=annotation_repo,annotation_revision=annotation['revision'],annotation_files=files,
                        policy='Original TRAIN short-answer aliases matched within originally positive passages; all questions sharing a full original passage stay in one component',
                        transformation_sha256=digest(Path(__file__)),window_sha256=digest(ROOT/'broad_pairs.py'))
        request=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest();counts=collections.Counter();parts=0
        for filename,group,rows in batches(pair_repo,pair['revision'],pair_files,['query','answer']):
            unit=hashlib.sha256((filename+':'+str(group)).encode()).hexdigest()[:24]
            meta=output/(unit+'.meta.json')
            if meta.exists():
                record=json.loads(meta.read_text());assert record['request_sha256']==request
                assert digest(output/record['part'])==record['sha256'];counts.update(record['counts']);parts+=1;continue
            items=[];local=collections.Counter(raw=len(rows))
            for offset,row in enumerate(rows):
                query,positive=clean(row['query']),clean(row['answer']);answers=aliases.get(identity(query))
                if not answers:local['no_train_answer_annotation']+=1;continue
                if len(positive)>500000:local['extreme_length']+=1;continue
                match=answer_window(positive,sorted(answers,key=lambda s:(-len(s),s)),tokenizer)
                if not match:local['answer_not_found_or_too_long']+=1;continue
                passage,evidence=match
                items.append(dict(query=query,positive=passage,raw_positive_key=identity(positive),group=identity(positive),answer_evidence=evidence,
                                  origin=dict(provenance,file=filename,row_group=group,row_in_group=offset)))
            accepted=[]
            for i in range(0,len(items),128):
                chunk,reasons=prepare_batch(items[i:i+128],tokenizer);accepted.extend(chunk);local.update(reasons)
            part=output/(unit+'.jsonl');temporary=part.with_suffix('.partial')
            with temporary.open('w') as f:
                for item in accepted:
                    # prepare_batch hashes its final passage; retain the full-source identity too.
                    item['raw_positive_key']=items[next(i for i,r in enumerate(items) if r['origin']==item['origin'])]['raw_positive_key']
                    item.update(source=pair_repo,relation='question positive passage with TRAIN short-answer evidence',source_group=item.pop('group'))
                    f.write(json.dumps(item,ensure_ascii=False)+'\n')
            temporary.replace(part);local['accepted']=len(accepted)
            record=dict(unit=unit,request_sha256=request,part=part.name,sha256=digest(part),rows=len(rows),accepted=len(accepted),counts=dict(local),selected_compressed_bytes=0,rejections={k:v for k,v in local.items() if k not in ('raw','accepted')})
            atomic(meta,record);counts.update(local);parts+=1
            if parts%10==0:print(slug,dict(counts),flush=True)
        atomic(output/'manifest.json',dict(complete=True,request_sha256=request,accepted=counts['accepted'],scanned=counts['raw'],parts=parts,counts=dict(counts),provenance=provenance))
        print('RECOVERED',slug,dict(counts),flush=True)


if __name__=='__main__':main()
