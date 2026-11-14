"""Fetch bounded, pinned TRAIN-only positive-pair candidates without teacher targets."""
import argparse,collections,hashlib,html,json,re,time,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def clean(value):
 if not isinstance(value,str):return ''
 value=html.unescape(unicodedata.normalize('NFKC',value))
 value=re.sub(r'</?(?:p|br|div|span|a|b|i|em|strong|code|pre|li|ul|ol)\b[^>]*>',' ',value,flags=re.I)
 return ' '.join(value.replace('\x00',' ').split())
def key(text):return hashlib.sha256(clean(text).casefold().encode()).hexdigest()
def valid_pair(q,p,relation):
 if key(q)==key(p):return 'identical_text'
 if len(q)<3 or len(p)<8:return 'too_short'
 if len(q)>1500 or len(p)>16000:return 'too_long'
 minimum=3 if relation in ('duplicate question','entailment auxiliary; cap training weight') else 6
 if len(p.split())<minimum:return 'short_positive'
 if q.startswith(('http://','https://')) or p.startswith(('http://','https://')):return 'url_only_prefix'
 if p.count('\ufffd')>2 or q.count('\ufffd')>2:return 'replacement_characters'
 return None

def main():
 from datasets import load_dataset
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scale',type=float,default=1.);parser.add_argument('--seed',type=int,default=19)
 parser.add_argument('--out',type=Path,default=ROOT/'work/data-first/candidates');parser.add_argument('--only',nargs='*');a=parser.parse_args()
 if not 0<a.scale<=1:parser.error('Scale must be in (0,1]')
 sources=json.loads((ROOT/'data-first/sources.lock.json').read_text());a.out.mkdir(parents=True,exist_ok=True)
 for s in sources:
  slug=s['id'].split('/')[-1]
  if a.only and slug not in a.only:continue
  target=max(1,round(s['quota']*a.scale));path=a.out/(slug+'.jsonl');meta=path.with_suffix('.manifest.json')
  request={'source':s,'target':target,'seed':a.seed,'shuffle_buffer':8192}
  if path.exists() and meta.exists():
   previous=json.loads(meta.read_text())
   if previous['request']!=request:raise RuntimeError('Refusing to mix different requests: '+str(path))
   if hashlib.sha256(path.read_bytes()).hexdigest()!=previous['sha256']:raise RuntimeError('Existing source hash mismatch')
   print('Already verified',slug,previous['accepted'],flush=True);continue
  stream=load_dataset(s['id'],s['config'],revision=s['revision'],split='train',streaming=True).shuffle(seed=a.seed,buffer_size=8192)
  seen=set();reasons=collections.Counter();accepted=scanned=0;start=time.monotonic();temp=path.with_suffix('.partial')
  with temp.open('w') as f:
   for row in stream:
    scanned+=1
    if 'label' in s and row['label']!=s['label']:reasons['other_label']+=1;continue
    q,p=clean(row[s['query']]),clean(row[s['positive']]);reason=valid_pair(q,p,s['relation'])
    if reason:reasons[reason]+=1;continue
    qkey=key(q)
    if qkey in seen:reasons['duplicate_query']+=1;continue
    seen.add(qkey);item={'query':q,'positive':p,'source':s['id'],'relation':s['relation'],'query_key':qkey,'positive_key':key(p),'source_row':scanned}
    f.write(json.dumps(item,ensure_ascii=False)+'\n');accepted+=1
    if accepted%10000==0:print(slug,accepted,'/',target,'scanned',scanned,flush=True)
    if accepted>=target:break
  temp.replace(path)
  summary={'request':request,'accepted':accepted,'scanned':scanned,'exhausted_before_quota':accepted<target,'filtered':dict(reasons),'seconds':time.monotonic()-start,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':'candidate pairs; benchmark/truncation dedup and split audit pending'}
  meta.write_text(json.dumps(summary,indent=2)+'\n');print('DONE',slug,json.dumps(summary),flush=True)
if __name__=='__main__':main()
