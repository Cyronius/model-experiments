"""Collect model-visible text exclusions from the already-observed legacy suite.

Run in .eval-venv. The FiQA/SCIDOCS relevance-label reservation is unchanged.
"""
import json,sys
from pathlib import Path
import datasets,mteb
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'tern-embed'))
from evaluate import TASKS

def strings(obj):
 if isinstance(obj,str):
  if len(obj.strip())>=3:yield obj.strip()
 elif isinstance(obj,datasets.Dataset):
  for row in obj:yield from strings(row)
 elif isinstance(obj,dict):
  for key,value in obj.items():
   if key not in ('score','label','labels','id','_id','query_id','query-id','corpus-id'):yield from strings(value)
 elif isinstance(obj,(list,tuple)):
  for value in obj:yield from strings(value)

def test_strings(obj):
 if isinstance(obj,dict):
  if 'test' in obj:yield from strings(obj['test'])
  else:
   for key,value in obj.items():
    if key not in ('train','validation','dev'):yield from test_strings(value)
 else:yield from strings(obj)

def main():
 out=ROOT/'work/expanded';out.mkdir(parents=True,exist_ok=True);texts=set();counts={}
 for task in mteb.get_tasks(tasks=[t for t,v in TASKS.items() if v!='Retrieval'],languages=['eng']):
  task.load_data();subset=set(test_strings(task.dataset));texts.update(subset);counts[task.metadata.name]=len(subset);print(task.metadata.name,len(subset),flush=True)
 path=out/'observed-test-texts.jsonl'
 with path.open('w') as f:
  for text in sorted(texts):f.write(json.dumps(text,ensure_ascii=False)+'\n')
 import hashlib
 (out/'observed-test-texts.audit.json').write_text(json.dumps({'tasks':counts,'unique_strings':len(texts),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'mteb_version':mteb.__version__,'scope':'Only already-observed test inputs, including nested candidate text lists. Numeric labels ignored. FiQA/SCIDOCS qrels not opened.'},indent=2)+'\n')
if __name__=='__main__':main()
