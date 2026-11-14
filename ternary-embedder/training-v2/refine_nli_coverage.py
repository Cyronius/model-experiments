"""Replace combinatorial NLI triplets with one triplet per distinct train premise."""
import argparse,hashlib,json,random
from pathlib import Path
from huggingface_hub import HfApi
from datasets import load_dataset

def distinct_nli(count,revision,seed):
 stream=load_dataset('sentence-transformers/all-nli','pair-class',split='train',revision=revision,streaming=True).shuffle(seed=seed,buffer_size=10000)
 groups={};seen=set();out=[];scanned=0
 for row in stream:
  scanned+=1;query=row['premise'].strip();text=row['hypothesis'].strip();label=int(row['label'])
  if not query or not text or query in seen or label not in (0,2):continue
  group=groups.setdefault(query,{});group.setdefault(label,text)
  if 0 in group and 2 in group and group[0]!=group[2]:
   out.append({'query':query,'positive':group[0],'negative':group[2],'source':'sentence-transformers/all-nli'})
   seen.add(query);del groups[query]
   if len(out)%1000==0:print('Unique NLI premises:',len(out),'scanned:',scanned,flush=True)
   if len(out)==count:break
 if len(out)!=count:raise RuntimeError('Insufficient unique-premise NLI training pairs')
 return out,scanned

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 rows=[json.loads(l) for l in a.input.read_text().splitlines()];old=[r for r in rows if r['source']=='sentence-transformers/all-nli']
 revision=HfApi().dataset_info('sentence-transformers/all-nli').sha
 nli,scanned=distinct_nli(len(old),revision,1)
 rows=[r for r in rows if r['source']!='sentence-transformers/all-nli']+nli;random.Random(1).shuffle(rows)
 a.out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 meta=json.loads(a.input.with_suffix('.source.json').read_text())
 for source in meta['sources']:
  if source['dataset']=='sentence-transformers/all-nli':
   source.update(config='pair-class',revision=revision,construction='one entailment/contradiction triplet per distinct premise',
                 label_names=['entailment','neutral','contradiction'],unique_queries=len(nli),scanned_rows=scanned)
 meta.update(sha256=hashlib.sha256(a.out.read_bytes()).hexdigest(),coverage_refined=True)
 a.out.with_suffix('.source.json').write_text(json.dumps(meta,indent=2)+'\n')
 print('Refined',len(rows),'triplets; NLI unique premises:',len(nli))
if __name__=='__main__':main()
