"""Keep missing judged documents as unreturned misses, without dropping queries."""
import numpy as np
from diagnose import evaluate as original_evaluate

def evaluate(scores,data):
 present=set(data['cids'])
 missing=sorted({cid for q in data['qrels'].values() for cid in q if cid not in present})
 if missing:
  if len(present)<10:raise ValueError("Missing-document adapter requires at least ten real corpus documents")
  data=dict(data,cids=data['cids']+missing)
  scores=np.pad(scores,((0,0),(0,len(missing))),constant_values=np.finfo(np.float32).min)
 metrics,rows=original_evaluate(scores,data)
 metrics['judged_documents_absent_from_corpus']=len(missing)
 return metrics,rows
