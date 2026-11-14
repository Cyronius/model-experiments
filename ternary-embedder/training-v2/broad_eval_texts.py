"""Pin text-only corpora for new reserved tests; never fetch their qrels."""
import json
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
import pyarrow.parquet as pq
from broad_data_fetch import atomic,digest
ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'work/broad-data/reserved-texts';out.mkdir(parents=True,exist_ok=True)
    api=HfApi(token=False);records={}
    for name in ('webis-touche2020','trec-covid'):
        repo='BeIR/'+name;info=api.dataset_info(repo);target=out/name;target.mkdir(exist_ok=True)
        text_files={}
        for kind in ('corpus','queries'):
            files=sorted(f.rfilename for f in info.siblings if f.rfilename.startswith(kind+'/') and f.rfilename.endswith('.parquet'))
            assert files
            destination=target/(kind+'.jsonl')
            if not destination.exists():
                temporary=destination.with_suffix('.partial')
                with temporary.open('w') as output:
                    for filename in files:
                        local=hf_hub_download(repo,filename,repo_type='dataset',revision=info.sha,token=False)
                        for batch in pq.ParquetFile(local).iter_batches(batch_size=2048):
                            for row in batch.to_pylist():
                                assert set(row).issubset({'_id','id','title','text'})
                                output.write(json.dumps(row,ensure_ascii=False)+'\n')
                temporary.replace(destination)
            text_files[destination.name]=dict(path=str(destination),sha256=digest(destination),bytes=destination.stat().st_size)
        records[name]=dict(repo=repo,revision=info.sha,text_files=text_files,judgments_read=False)
        print('Pinned text only',name,{k:v['bytes'] for k,v in text_files.items()},flush=True)
    atomic(ROOT/'broad-data/evaluation.lock.json',dict(scope='Two new reserved label sets, untouched. FiQA/SCIDOCS remain observed development tasks. Historical pretraining exposure unknown.',datasets=records))


if __name__=='__main__':main()
