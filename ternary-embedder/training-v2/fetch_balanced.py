"""50% MS MARCO, 25% NLI, 25% duplicate-question public TRAIN triplets."""
import argparse,hashlib,json,random,urllib.request
from pathlib import Path
from huggingface_hub import HfApi
SOURCES=[('sentence-transformers/msmarco-bm25',.5),('sentence-transformers/all-nli',.25),('sentence-transformers/quora-duplicates',.25)]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rows',type=int,default=50000)
 p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=1)
 p.add_argument('--preview',action='store_true',help='Small live-schema smoke via dataset-viewer API')
 a=p.parse_args();rows=[];sources=[]
 if a.rows<32 or a.rows%4 or a.preview and a.rows>100:p.error('Use >=32 rows; preview is limited to 100')
 for dataset,fraction in SOURCES:
  count=int(a.rows*fraction);revision=HfApi().dataset_info(dataset).sha
  if dataset=='sentence-transformers/all-nli' and not a.preview:
   from refine_nli_coverage import distinct_nli
   nli,scanned=distinct_nli(count,revision,a.seed);rows.extend(nli)
   sources.append({'dataset':dataset,'config':'pair-class','split':'train','revision':revision,'rows':count,
                   'construction':'one entailment/contradiction triplet per distinct premise','unique_queries':count,'scanned_rows':scanned})
   continue
  if a.preview:
   url=f'https://datasets-server.huggingface.co/rows?dataset={dataset.replace("/","%2F")}&config=triplet&split=train&offset=0&length={count}'
   with urllib.request.urlopen(url) as r:stream=[x['row'] for x in json.load(r)['rows']]
  else:
   from datasets import load_dataset
   stream=load_dataset(dataset,'triplet',split='train',revision=revision,streaming=True).shuffle(seed=a.seed,buffer_size=10000)
  accepted=0
  for row in stream:
   item={'query':row.get('query',row.get('anchor','')).strip(),'positive':row['positive'].strip(),'negative':row['negative'].strip(),'source':dataset}
   if not all(item.values()) or item['positive']==item['negative']:continue
   rows.append(item);accepted+=1
   if accepted==count:break
  if accepted!=count:raise RuntimeError('Training source ended early: '+dataset)
  sources.append({'dataset':dataset,'config':'triplet','split':'train','revision':revision,'rows':accepted})
 random.Random(a.seed).shuffle(rows);a.out.parent.mkdir(parents=True,exist_ok=True)
 a.out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 a.out.with_suffix('.source.json').write_text(json.dumps({'split':'train','sources':sources,'seed':a.seed,'preview':a.preview,
   'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest()},indent=2)+'\n')
 print('Wrote',len(rows),'balanced training triplets')
if __name__=='__main__':main()
