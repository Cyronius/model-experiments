"""Conservative candidate cleanup; legacy retention examples are preserved separately."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
from broad_pairs import identity
ROOT=Path(__file__).resolve().parent
lock=json.loads((ROOT/'broad-data/quality-tools.lock.json').read_text())
for entry in lock.values():
    path=Path(entry['path'])
    if not path.is_absolute():path=ROOT.parent.parent/path
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    sys.path.insert(0,str(path))
from ftfy import fix_text


def repaired(text):
    if any(marker in text for marker in ('Ã','Â','â€','â\x80','ðŸ','ï¿','\ufffd')):
        return fix_text(text,unescape_html=False,uncurl_quotes=False,normalization='NFKC')
    return unicodedata.normalize('NFKC',text)


def quality_pair(row):
    query,positive=row['query'],row['positive'];flags=[]
    q,p=repaired(query),repaired(positive)
    if (q,p)!=(query,positive):flags.append('encoding_or_unicode_repaired')
    if p.startswith('[') and p.endswith(']') and len(p)<20000:
        try:values=ast.literal_eval(p)
        except (ValueError,SyntaxError,RecursionError):values=None
        if isinstance(values,list) and 1<=len(values)<=32 and all(isinstance(v,str) for v in values):
            p='; '.join(' '.join(v.split()) for v in values if v.strip());flags.append('serialized_list_to_text')
    source=row.get('source','IFM/SFT-Reasoning')
    if source=='IFM/SFT-Reasoning' and re.match(r"^(?:i[’']?m sorry|i am sorry|sorry[, .]|i (?:cannot|can[’']t|am unable)|as an ai(?: language model)?|i (?:do not|don[’']t) (?:have|know))\b",p,re.I):
        return None,'generic_model_refusal'
    evidence=row.get('answer_evidence')
    if evidence and evidence.get('answer'):
        answer=repaired(evidence['answer'])
        if answer.casefold() not in p.casefold():return None,'cleanup_lost_answer_evidence'
    if not flags:return row,None
    output=dict(row,query=q,positive=p,raw_query_key=identity(query),original_positive_key=identity(positive))
    output['origin']=dict(row.get('origin',{}),quality_changes=flags,quality_tool_versions={name:spec['version'] for name,spec in lock.items()})
    if evidence and evidence.get('answer'):
        output['answer_evidence']=dict(evidence,answer=repaired(evidence['answer']),original_answer=evidence['answer'])
    return output,flags
