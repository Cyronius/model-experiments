"""Fresh answer-level checks; correctness and presentation are separate."""
import ast
from fractions import Fraction
import json
import random
import re

from horizon_data import equivalent


def unwrap(text):
    text=text.strip()
    if text.startswith('```') and text.endswith('```'):
        text=re.sub(r'^```[^\n]*\n','',text)[:-3].strip()
    return text


def json_value(text):
    text=unwrap(text)
    try:return json.loads(text)
    except ValueError:pass
    values=[]
    for m in re.finditer(r'[\[{]',text):
        try:
            obj,n=json.JSONDecoder().raw_decode(text[m.start():])
            if isinstance(obj,(dict,list)):values.append((m.start(),m.start()+n,obj))
        except ValueError:pass
    # Nested fragments are not independent answers; require one outer object.
    outer=[v for v in values if not any(a<v[0] and b>=v[1] for a,b,_ in values)]
    if len(outer)==1:return outer[0][2]
    raise ValueError('No unique JSON answer')


def numeric_value(text):
    text=unwrap(text).strip()
    # Normalize elementary LaTeX before looking for one explicit final answer.
    text=re.sub(r'\\(?:d?frac)\{([+-]?\d+)\}\{(\d+)\}',r'\1/\2',text)
    boxed=re.findall(r'\\boxed\{([^{}]+)\}',text)
    if len(boxed)>1:raise ValueError('Ambiguous boxed answers')
    if boxed:text=boxed[0]
    else:
        markers=list(re.finditer(r'(?i)(?:final\s+)?(?:answer|result)(?:\s+is|\s*:)|=',text))
        if markers:text=text[markers[-1].end():]
        elif '\n' in text:text=text.splitlines()[-1]
    text=text.strip(' *$`\n').rstrip('.').strip()
    text=re.sub(r'(?i)\s*(?:milliliters|millilitres|ml)\.?$','',text).strip(' *$`')
    if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?',text):text=text.replace(',','')
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*/\s*\d+)?',text):
        raise ValueError('No unambiguous numeric final answer')
    return Fraction(text.replace(' ',''))


def grade(answer, check):
    """Return content correctness, strict presentation, and extraction status.

    These are fixed closed-answer tasks. Unparseable is kept distinct for review
    but counts as not-correct in aggregate. No fuzzy keyword accuracy scoring.
    """
    text=answer.strip(); clean=unwrap(text).strip(' *$\n')
    kind=check['kind'];expected=check['expected']
    try:
        if kind in ('json','codeio'):
            value=json_value(text)
            content=equivalent(value,expected)
            try:strict=equivalent(json.loads(text),expected)
            except ValueError:strict=False
        elif kind=='numeric':
            content=numeric_value(text)==Fraction(str(expected))
            strict=text==str(expected)
        elif kind=='numeric_list':
            try:value=json_value(text)
            except ValueError:value=[int(x.strip()) for x in clean.rstrip('.').split(',')]
            content=equivalent(value,expected);strict=content and not text.startswith('```')
        elif kind=='literal':
            try:value=json_value(text)
            except ValueError:
                value=re.sub(r'(?is)^.*?(?:prints|output is|output:)\s*','',clean).strip(' `')
                value=ast.literal_eval(unwrap(value).rstrip('.'))
            content=equivalent(value,expected);strict=content and text==repr(expected)
        elif kind=='identifier':
            values=set(re.findall(r'(?<![A-Z0-9])[A-Z2-9]{8}(?![A-Z0-9])',text))
            if len(values)!=1:raise ValueError('No unique parcel code')
            content=values=={expected};strict=text==expected
        elif kind=='text':
            value=clean.rstrip('.!').strip()
            content=value.casefold()==str(expected).casefold();strict=text==str(expected)
        elif kind=='exact':
            content=clean==expected;strict=text==expected
        else:raise ValueError('unknown check')
        return {'correct':bool(content),'format_pass':bool(strict),'parsed':True}
    except (ValueError,TypeError,SyntaxError,ZeroDivisionError,OverflowError):
        return {'correct':False,'format_pass':False,'parsed':False}


def synthetic_cases(seed, per_family, exclude=()):
    rng=random.Random(seed);cases=[];seen=set(exclude);counts={}
    def add(family,prompt,kind,expected,messages=None):
        messages=messages or [{'role':'user','content':prompt}]
        key=json.dumps(messages,sort_keys=True)
        if counts.get(family,0)>=per_family or key in seen:return
        seen.add(key);counts[family]=counts.get(family,0)+1
        cases.append({'id':f'{seed}-{family}-{len(cases)}','family':family,
            'messages':messages,
            'check':{'kind':kind,'expected':expected},'max_new_tokens':256})
    while len(cases)<7*per_family:
        a,b=rng.randint(20,120),rng.randint(11,39)
        add('arithmetic',f'What is {a} multiplied by {b}? Return only the final number.','numeric',str(a*b))
        values=rng.sample(range(-80,81),5)
        add('sorting',f'Sort these integers from smallest to largest: {", ".join(map(str,values))}. Return only a JSON array.','json',sorted(values))
        x,y,z=[rng.randint(2,25) for _ in range(3)]
        which=rng.randrange(4)
        code,expected=[(f'a=[{x},{y}]; b=a; b.append({z}); print(a)',[x,y,z]),
                       (f'xs=[{x},{y},{z}]; print(sum(xs[1:]))',y+z),
                       (f'xs=[{x},{y},{z}]; print(xs[::-1])',[z,y,x]),
                       (f'd={{"a":{x}}}; print(d.get("b",{y})+d["a"])',x+y)][which]
        add('python',f'What does this Python program print?\n```python\n{code}\n```\nReturn only the printed value.','literal',expected)
        liters=Fraction(rng.randint(1,99),10)
        add('units',f'Convert {float(liters):g} liters to milliliters. Return only the number.','numeric',str(liters*1000))
        label='item-'+str(rng.randrange(100000,999999));n=rng.randint(1,999)
        add('json',f'Return only a JSON object with label set to "{label}", count set to {n}, and active set to true.','json',{'label':label,'count':n,'active':True})
        city=rng.choice(['Marlow','Denton','Linden','Harper']);day=rng.choice(['Monday','Tuesday','Wednesday','Thursday'])
        desks=rng.randint(3,90)
        add('grounding',f'Use only this notice: The {city} library reopens on {day} after repairs. It received {desks} new desks.\nReturn a JSON object containing the reopening day and number of new desks, using keys "day" and "desks".','json',{'day':day,'desks':desks})
        tag=''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(8))
        add('memory','', 'identifier',tag,[{'role':'user','content':f'Remember my parcel code: {tag}.'},
            {'role':'assistant','content':f'Your parcel code is {tag}.'},
            {'role':'user','content':'What is my parcel code? Output only the code.'}])
    return cases


LONG_PROMPTS=[
    'Write a Python function that groups transaction dictionaries by currency and sums their amounts. Include three runnable tests and explain how you handle missing fields.',
    'Explain merge sort to a beginner. Give correct Python code, an example, and its time and space complexity.',
    'Compare renting and buying a home using a fictional numerical example. State your assumptions and avoid claiming one option is universally better.',
    'Write a coherent six-paragraph story about a botanist repairing a greenhouse after a storm. Give it a clear ending.',
    'Explain how to troubleshoot a home Wi-Fi connection that works near the router but fails upstairs. Give an ordered diagnostic procedure.',
    'Write a Python function that merges overlapping intervals without changing its input. Include tests for empty input, touching intervals and nested intervals.',
    'Explain photosynthesis and cellular respiration, including how they differ and how they are connected. Use examples suitable for a secondary-school student.',
    'Plan a two-hour introductory workshop on spreadsheets. Include timing, exercises and ways to check whether attendees understood each skill.',
]


def long_cases(prefix, prompts):
    return [{'id':f'{prefix}-long-{i}','family':'long_form','messages':[{'role':'user','content':p}],
             'check':None,'max_new_tokens':1024} for i,p in enumerate(prompts)]
