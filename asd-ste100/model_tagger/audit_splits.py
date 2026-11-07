"""Check token-canonical source/target and generated intermediate text across splits."""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from tagger import Tags,SteTokenizer,source_units
from packed import unpack
from rollouts import records,sha256
from ste_tokenizer import model_tokens
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data'))
from text_utils import normalized
import hashlib


def key(text):return hashlib.sha256(normalized(' '.join(t.text for t in model_tokens(text))).encode()).hexdigest()


def audit(directory,vocab):
    tok=SteTokenizer(vocab);tags=Tags(tok)
    seen={};groups=set();examples=[];scanned=0;intermediates=0;conflicts={};conflict_groups=set()
    index={s:np.load(directory/f'{s}.index.npy',mmap_mode='r') for s in ('train','dev','test')}
    data={s:np.memmap(directory/f'{s}.data.i32',dtype='<i4',mode='r') for s in index}
    def register(text,row,role):
        code=key(text);old=seen.get(code)
        item=(row['split'],row['group'],row['id'],role)
        if old is None:seen[code]=item
        elif old[0]!=item[0]:
            groups.update((old[1],item[1]))
            if len(examples)<30:examples.append({'text':text,'first':old,'second':item})
        return code
    for row in records(directory/'pairs.jsonl.gz'):
        scanned+=1
        register(row['src'],row,'source');target=register(row['ste'],row,'target')
        source_key=(row['mode'],key(row['src']))
        old=conflicts.get(source_key)
        if old is None:conflicts[source_key]=(target,row['group'])
        elif old[0]!=target:conflict_groups.update((old[1],row['group']))
        if row['edit_passes']>1:
            current=row['src']
            for i in range(row['edit_passes']-1):
                encoded=tok.encode_pair({'src':current,'ste':row['ste'],'mode':row['mode']},.7)
                stored=unpack(data[row['split']],index[row['split']][row['first_example']+i])
                out=tags.apply(source_units(encoded),stored['labels'].tolist(),stored['label_cases'].tolist(),encoded['slots'])
                values,cases=map(list,zip(*out));current=tok.detokenize(values,encoded['slots'],cases)
                register(current,row,'intermediate');intermediates+=1
                source_key=(row['mode'],key(current));old=conflicts.get(source_key)
                if old is None:conflicts[source_key]=(target,row['group'])
                elif old[0]!=target:conflict_groups.update((old[1],row['group']))
        if scanned%100000==0:print(f'checked canonical splits: {scanned:,}',flush=True)
    result={'passed':not groups and not conflict_groups,'pairs':scanned,'intermediate_sources':intermediates,
            'cross_split_groups':sorted(groups),'conflicting_source_groups':sorted(conflict_groups),
            'examples':examples,'summary_sha256':sha256(directory/'summary.json'),'code_sha256':sha256(Path(__file__)),
            'scope':'Canonical surface token text including real copy-slot surfaces; not opaque request-local slot IDs.'}
    (directory/'split-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--vocab',type=Path,required=True)
    a=p.parse_args();print(json.dumps(audit(a.data,a.vocab),ensure_ascii=False,indent=2))
