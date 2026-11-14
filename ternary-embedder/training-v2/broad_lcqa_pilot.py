"""Adapt public Horizon LCQA examples and validate future generated retrieval pairs."""
import collections
import hashlib
import json
from pathlib import Path
import re
import urllib.request
from broad_pairs import answer_window,clean,identity,prepare_batch
from broad_data_fetch import atomic,digest
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent


def generated_pair(payload,passage,tokenizer):
    """Accept structured questions with a verbatim evidence quote; never execute text."""
    if not isinstance(payload,dict) or set(payload)!={'question','answer_quote'}:return None
    query,quote=payload['question'],payload['answer_quote']
    if not isinstance(query,str) or not isinstance(quote,str):return None
    query,quote,passage=clean(query),clean(quote),clean(passage)
    if len(quote)<3 or quote.casefold() not in passage.casefold():return None
    match=answer_window(passage,[quote],tokenizer)
    if not match:return None
    positive,evidence=match
    accepted,_=prepare_batch([dict(query=query,positive=positive,group=identity(passage),origin={},answer_evidence=evidence)],tokenizer)
    return accepted[0] if accepted else None


def main():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
    source=json.loads((ROOT/'broad-data/lcqa-source.json').read_text());raw=ROOT/'work/broad-data/lcqa-examples';raw.mkdir(exist_ok=True)
    files={}
    for name in ('input.jsonl','questions.jsonl','final.jsonl'):
        path=raw/name
        if not path.exists():
            url=f"https://raw.githubusercontent.com/{source['repo']}/{source['revision']}/lcqa-synthesis/data/{name}"
            with urllib.request.urlopen(url,timeout=45) as response:
                content=response.read(100*1024**2+1)
            if len(content)>100*1024**2:raise RuntimeError('Example file exceeds bounded download size')
            path.write_bytes(content)
        files[name]=digest(path)
    documents=[json.loads(line) for line in (raw/'input.jsonl').open()]
    questions={r['id']:r for r in (json.loads(line) for line in (raw/'questions.jsonl').open())}
    final={r['id']:r for r in (json.loads(line) for line in (raw/'final.jsonl').open())}
    output=ROOT/'work/broad-data/candidates/lcqa-pilot';output.mkdir(exist_ok=True);rows=[];counts=collections.Counter()
    for index,question in questions.items():
        q=(question.get('question') or {}).get('question')
        if not q or index not in final:counts['no_complete_pair']+=1;continue
        answer=final[index]['conversation'][-1].get('content','')
        # Restrict the pilot to quotes/bold spans actually present in a source snippet.
        spans=re.findall(r'\*\*([^*\n]{3,300})\*\*|"([^"\n]{3,500})"',answer)
        quotes=sorted({clean(a or b).strip(' .') for a,b in spans},key=lambda text:(-len(text),text))
        candidate=None
        for snippet in documents[index]:
            for quote in quotes:
                candidate=generated_pair(dict(question=q,answer_quote=quote),snippet['snippet'],tokenizer)
                if candidate:
                    candidate.update(raw_positive_key=identity(snippet['doc']),source_group=identity(snippet['doc']),source='ifm-ai/LCQA',relation='source-generated question with quote verified against supporting snippet',
                                     origin=dict(repo=source['repo'],revision=source['revision'],document_id=index,knobs=question.get('knobs'),source_files_sha256=files))
                    candidate.pop('group',None);break
            if candidate:break
        if candidate:rows.append(candidate)
        else:counts['no_short_grounded_pair']+=1
    part=output/'pilot.jsonl';part.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    atomic(output/'pilot.meta.json',dict(part=part.name,sha256=digest(part),accepted=len(rows),source=source))
    atomic(output/'manifest.json',dict(complete=True,accepted=len(rows),scanned=len(questions),counts=dict(counts),source=source,
                    limits='Small upstream example pilot, not a representative bulk dataset. Quote consistency does not prove factual correctness or complete relevance. No LLM API was called.'))
    print('LCQA short grounded example pairs',len(rows),'from',len(questions),flush=True)


if __name__=='__main__':main()
