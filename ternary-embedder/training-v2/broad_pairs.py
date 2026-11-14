"""Pure transformations for retrieval data; corpus programs are never executed."""
import hashlib
import html
import re


def clean(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'<(?:script|style)\b.*?</(?:script|style)>', ' ', text, flags=re.S|re.I)
    text = re.sub(r'<[^>\n]{1,500}>', ' ', text)
    return ' '.join(html.unescape(text).replace('\x00', ' ').split())


def identity(text):
    return hashlib.sha256(clean(text).casefold().encode()).hexdigest()


def bert_prefix(text, cap=128):
    """Preserve the first cap WordPieces without constructing long overflow encodings.

    Every whitespace-separated word containing ASCII letters/digits contributes
    at least one BERT token. Keeping cap such words therefore preserves all cap
    input tokens, including room for special tokens. Short/non-ASCII inputs are
    left intact. This bound is specific to the project's BERT WordPiece tokenizer.
    """
    count=0
    for match in re.finditer(r'\S+',text):
        if re.search('[A-Za-z0-9]',match.group()):
            count+=1
            if count>=cap:return text[:match.end()]
    return text


def instruction_pair(text):
    for a,b,c in [('### Instruction','### Reasoning','### Response'),('# Query','# Analysis','# Response')]:
        if text.startswith(a+'\n') and text.count('\n'+b+'\n')==1 and text.count('\n'+c+'\n')==1:
            query,rest=text[len(a):].split('\n'+b+'\n')
            _,answer=rest.split('\n'+c+'\n')
            query,answer=clean(query),clean(answer)
            if not re.match(r'^(?:what|why|how|who|when|where|which|can|does|do|is|are|explain|describe|define)\b',query,re.I):
                return None
            if re.search(r'NAME_\d|<\|im_|</?think>|for (?:Engineer|Analysis|Decision Making) for|previous (?:message|answer)|above (?:text|code)',query+'\n'+answer):
                return None
            return query,answer
    return None


def answer_window(context, answers, tokenizer, cap=128):
    """Crop only when an annotated answer survives, with exact character evidence."""
    context=clean(context)
    spans=[]
    for answer in answers:
        annotated_start=answer.get('start') if isinstance(answer,dict) else None
        answer=clean(answer['text'] if isinstance(answer,dict) else answer)
        if not answer:
            continue
        if annotated_start is not None:
            if context[annotated_start:annotated_start+len(answer)].casefold()==answer.casefold():
                spans.append((annotated_start,annotated_start+len(answer),answer))
            continue
        match=re.search(r'(?<!\w)'+re.escape(answer)+r'(?!\w)',context,re.I)
        if match:
            spans.append((match.start(),match.end(),answer))
    if not spans:
        return None
    encoded=tokenizer(context,add_special_tokens=False,return_offsets_mapping=True,verbose=False)
    offsets=encoded['offset_mapping'];width=cap-tokenizer.num_special_tokens_to_add(pair=False)
    for start,end,answer in spans:
        left=next((i for i,(a,b) in enumerate(offsets) if b>start),None)
        right=next((i for i,(a,b) in enumerate(offsets) if b>=end),None)
        if left is None or right is None or right-left+1>width:
            continue
        first=max(0,min(left-(width-(right-left+1))//2,len(offsets)-width))
        last=min(len(offsets),first+width)
        while first<=left and last>right:
            passage=context[offsets[first][0]:offsets[last-1][1]].strip()
            if len(tokenizer(passage,verbose=False)['input_ids'])<=cap and re.search(re.escape(answer),passage,re.I):
                return passage,dict(answer=answer,original_char_start=offsets[first][0],original_char_end=offsets[last-1][1])
            if last-1>right:
                last-=1
            else:
                first+=1
    return None


def candidates(row, spec, origin):
    """Return positives only. Lower-voted and mined answers are not negatives."""
    kind=spec.get('kind','pair')
    if kind=='stack':
        metadata=row.get('metadata') or []
        url=metadata[0] if metadata else ''
        if re.search(r'\.meta\.|//meta\.',url):
            return []
        query=clean(row.get('question'))
        answers=[a for a in row.get('answers',[]) if a.get('pm_score',0)>=2 or (a.get('selected') and a.get('pm_score',0)>=0)]
        answers.sort(key=lambda a:(not a.get('selected'),-a.get('pm_score',0),a.get('answer_id',0)))
        result=[]
        for answer in answers[:2]:
            if re.search(r'<(?:pre|code|img)\b',row.get('question','')+answer.get('text',''),re.I):
                continue
            attribution={k:answer.get(k) for k in ('answer_id','author','author_id','author_profile','pm_score','selected')}
            result.append(dict(query=query,positive=clean(answer.get('text')),group=identity(url or query),
                               origin=dict(origin,question_url=url,question_author_profile=metadata[2] if len(metadata)>2 else '',answer=attribution)))
        return result
    if kind=='horizon':
        pair=instruction_pair(row.get('text',''))
        if not pair:
            return []
        q,p=pair
        return [dict(query=q,positive=p,group=identity(q),origin=origin)]
    if kind=='squad':
        q,p=row['question'],row['context']
        normalized=clean(p);answers=[]
        for text,start in zip(row['answers']['text'],row['answers']['answer_start']):
            if p[start:start+len(text)]!=text:continue
            answer=clean(text);near=len(clean(p[:start]));position=normalized.find(answer,max(0,near-2),near+len(answer)+3)
            if position>=0:answers.append(dict(text=answer,start=position))
        return [dict(query=clean(q),positive=normalized,group=identity(p),origin=dict(origin,id=row['id'],title=row['title']),answers=answers)]
    q,p=row.get(spec['query']),row.get(spec['positive'])
    if not isinstance(q,str) or not isinstance(p,str):
        return []
    return [dict(query=clean(q),positive=clean(p),group=identity(q),origin=origin)]


def prepare_batch(items, tokenizer, cap=128):
    accepted=[];reasons={}
    def reject(reason):reasons[reason]=reasons.get(reason,0)+1
    reasonable=[]
    for item in items:
        q,p=item['query'],item['positive']
        if len(q)<5 or len(p.split())<6 or len(q)>3000 or len(p)>20000:
            reject('text_shape');continue
        # Each whitespace-separated ASCII word requires at least one BERT token.
        # This rejects only provably overlong unannotated inputs, before WordPiece.
        budget=cap-tokenizer.num_special_tokens_to_add(pair=False)
        if not item.get('answers') and sum(bool(re.search('[A-Za-z0-9]',word)) for word in p.split(maxsplit=cap))>budget:
            reject('positive_words_exceed_cap');continue
        if identity(q)==identity(p):
            reject('identical');continue
        if '\ufffd' in q+p or re.search(r'(?:https?://\S+\s*){3,}',p):
            reject('corrupt_or_link_list');continue
        reasonable.append(item)
    if not reasonable:
        return accepted,reasons
    ids=tokenizer([r[k] for r in reasonable for k in ('query','positive')],truncation=False,verbose=False)['input_ids']
    for i,item in enumerate(reasonable):
        qi,pi=ids[2*i:2*i+2]
        if len(qi)>cap:
            reject('query_too_long');continue
        item=dict(item,raw_positive_key=identity(item['positive']))
        if len(pi)>cap:
            if item.get('answers'):
                result=answer_window(item['positive'],item['answers'],tokenizer,cap)
                if result:
                    item['positive'],item['answer_evidence']=result
                    pi=tokenizer(item['positive'],verbose=False)['input_ids']
                else:
                    reject('no_answer_preserving_window');continue
            else:
                reject('positive_too_long');continue
        if len(pi)>cap:
            raise AssertionError('Window length exceeded cap')
        item.pop('answers',None)
        item['query_tokens']=len(qi);item['positive_tokens']=len(pi)
        accepted.append(item)
    return accepted,reasons
