"""Fetch a reproducible slice of public MS MARCO train triplets."""
import argparse,hashlib,json
from pathlib import Path
from huggingface_hub import HfApi

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rows',type=int,default=50000)
 p.add_argument('--out',type=Path,default=Path(__file__).parent/'work/train.jsonl');p.add_argument('--seed',type=int,default=0)
 a=p.parse_args()
 if a.rows<32:p.error('Need at least 32 training rows')
 from datasets import load_dataset
 dataset='sentence-transformers/msmarco-bm25';revision=HfApi().dataset_info(dataset).sha
 stream=load_dataset(dataset,'triplet',split='train',revision=revision,streaming=True).shuffle(seed=a.seed,buffer_size=10000)
 a.out.parent.mkdir(parents=True,exist_ok=True);count=0
 with a.out.open('w') as f:
  for row in stream:
   r={k:row[k].strip() for k in ('query','positive','negative')}
   if not all(r.values()) or r['positive']==r['negative']:continue
   f.write(json.dumps(r)+'\n');count+=1
   if count==a.rows:break
 if count!=a.rows:raise RuntimeError('Training source ended early')
 a.out.with_suffix('.source.json').write_text(json.dumps({'dataset':dataset,'revision':revision,'config':'triplet','split':'train',
  'rows':count,'seed':a.seed,'shuffle_buffer':10000,'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest()},indent=2)+'\n')
if __name__=='__main__':main()
