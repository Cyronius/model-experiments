"""Prepare auditable Horizon targets; never execute corpus code outside bwrap.

Math targets are recomputed for explicitly supported problem grammars. CodeI/O
targets are checked against the supplied reference program, not against an LLM.
Instruction responses require unambiguous section boundaries and are labelled
source-curated, not independently verified. Whole underlying problems/functions
are assigned to one split before token filtering or sampling.
"""
import argparse
import ast
import collections
import concurrent.futures
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import tempfile

from healing_data import encode_complete, generation_metrics
from scale_data import load_tokenizer


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def arithmetic(text):
    """A deliberately small numeric grammar; no eval/sympify on corpus strings."""
    tree = ast.parse(text.replace('^', '**'), mode='eval')
    if len(list(ast.walk(tree))) > 64:
        raise ValueError('expression too large')
    def calc(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value) or abs(node.value) > 10**15:
                raise ValueError('large constant')
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(node.op, ast.USub) else 1) * calc(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = calc(node.left), calc(node.right)
            if isinstance(node.op, ast.Add): v = a+b
            elif isinstance(node.op, ast.Sub): v = a-b
            elif isinstance(node.op, ast.Mult): v = a*b
            elif isinstance(node.op, ast.Div): v = a/b
            elif isinstance(node.op, ast.Pow) and b.denominator == 1 and abs(b) <= 12: v = a**int(b)
            else: raise ValueError('unsupported operator')
            if abs(v.numerator) > 10**100 or v.denominator > 10**100:
                raise ValueError('large result')
            return v
        raise ValueError('unsupported expression')
    return calc(tree.body)


def solve_math(prompt):
    p = prompt.strip().removesuffix('\\n').strip()
    m = re.fullmatch(r'(?:Calculate|Evaluate|What is) (.+?)[.?]', p)
    if m:
        try: return str(arithmetic(m[1])), 'numeric'
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError): pass
    m = re.fullmatch(r'What is the (.+?) digit of (\d+)\?', p)
    places = ['units', 'tens', 'hundreds', 'thousands', 'ten thousands', 'hundred thousands',
              'millions', 'ten millions', 'hundred millions', 'billions']
    if m and m[1] in places:
        return str(int(m[2]) // 10**places.index(m[1]) % 10), 'numeric'
    m = re.fullmatch(r'(?:Sort|Put) (.+?)(?: in (increasing|ascending|decreasing|descending) order)?[.?]', p)
    if m:
        try:
            parts = [arithmetic(s.strip()) for s in m[1].split(',')]
            if 2 <= len(parts) <= 12:
                return ', '.join(map(str, sorted(parts, reverse=m[2] in ('decreasing','descending')))), 'numeric_list'
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError): pass
    m = re.fullmatch(r'(?:What is|Calculate|Find) the (greatest common (?:divisor|factor)|least common multiple) of (\d+) and (\d+)[.?]', p)
    if m:
        a,b=int(m[2]),int(m[3]); v=math.lcm(a,b) if m[1]=='least common multiple' else math.gcd(a,b)
        return str(v), 'numeric'
    m = re.fullmatch(r'Is (-?\d+) (even|odd)\?', p)
    if m: return ('Yes' if int(m[1])%2 == (m[2]=='odd') else 'No'), 'boolean'
    m = re.fullmatch(r'Is (-?\d+) a multiple of (-?\d+)\?', p)
    if m and int(m[2]): return ('Yes' if int(m[1])%int(m[2])==0 else 'No'), 'boolean'
    m = re.fullmatch(r'Two letters picked without replacement from ([a-z]+)\. Give prob of picking 2 ([a-z])\.', p)
    if m and len(m[1])>=2:
        return str(Fraction(math.comb(m[1].count(m[2]),2),math.comb(len(m[1]),2))), 'numeric'
    return None


# The sandbox has no home, project files, credentials or network. stdout from
# corpus code is discarded; process, CPU, address-space and file limits apply.
SANDBOX = r'''
import contextlib, io, json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
resource.setrlimit(resource.RLIMIT_AS, (384*1024*1024, 384*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024*1024, 1024*1024))
resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
class Sink:
    def write(self, value): return len(value)
    def flush(self): pass
payload=json.load(sys.stdin)
try:
    namespace={}
    with contextlib.redirect_stdout(Sink()), contextlib.redirect_stderr(Sink()):
        exec(compile(payload['code'], '<reference>', 'exec'), namespace)
        result=namespace['main_solution'](**payload['inputs'])
    encoded=json.dumps({'ok':True,'result':result},allow_nan=False)
    if len(encoded)>65536: raise ValueError('oversize output')
    print(encoded)
except BaseException as error:
    print(json.dumps({'ok':False,'error':type(error).__name__}))
'''


def run_reference(code, inputs):
    cmd=['bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv',
         '--setenv','PATH','/usr/bin:/bin','--setenv','LANG','C.UTF-8',
         '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64',
         '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--chdir','/tmp',
         '/usr/bin/python3','-I','-c',SANDBOX]
    try:
        # Regular files make RLIMIT_FSIZE effective even if corpus code bypasses
        # Python's redirected streams and writes directly to file descriptors.
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            r=subprocess.run(cmd,input=json.dumps({'code':code,'inputs':inputs}),
                             stdout=out,stderr=err,text=True,timeout=3)
            if r.returncode:return None
            out.seek(0);x=json.loads(out.read(65537))
        return x if x.get('ok') else None
    except (subprocess.TimeoutExpired, ValueError, OSError): return None


def equivalent(a,b):
    if isinstance(a,bool) or isinstance(b,bool): return type(a)==type(b) and a==b
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-9)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    return type(a)==type(b) and a==b


def literal(text):
    if len(text)>10000: raise ValueError('large literal')
    try: return json.loads(text)
    except ValueError: return ast.literal_eval(text)


def parse_code(text):
    parts=re.split(r'\n\n(User|Assistant):\s*', text)
    if not text.startswith('User: ') or len(parts)<3: return None
    prompt=parts[0][6:]
    turns=[(parts[i],parts[i+1]) for i in range(1,len(parts)-1,2)]
    if not turns or turns[-1][0]!='User' or not re.fullmatch(r'Correct (input|output)!',turns[-1][1].strip()):
        return None
    if len(turns)<2 or turns[-2][0]!='Assistant': return None
    answer=turns[-2][1].strip()
    # Parse a complete final JSON object, not a JSON-looking fragment in prose.
    candidates=[]
    for match in re.finditer(r'(?m)^\s*\{',answer):
        try:
            obj,end=json.JSONDecoder().raw_decode(answer[match.start():].lstrip())
            suffix=answer[match.start():].lstrip()[end:].strip().strip('`').strip()
            if not suffix and isinstance(obj,dict) and set(obj) in ({'input'},{'output'}): candidates.append(obj)
        except ValueError: pass
    if len(candidates)!=1: return None
    target=candidates[0]
    code_match=re.search(r'Tip: Here is a reference code snippet.*?\n\n(.*)\Z',prompt,re.S)
    given=re.search(r'Given the following (input|output):\s*\n(.*?)\n\nCan you predict',prompt,re.S)
    if not code_match or not given: return None
    code=code_match[1].strip()
    if code.startswith('```'): code=re.sub(r'^```\w*\n|\n```$', '',code)
    try:
        value=literal(given[2].strip())
        tree=ast.parse(code)
        group=digest(ast.dump(tree,include_attributes=False))
    except (ValueError,SyntaxError,RecursionError): return None
    if given[1]=='input' and set(target)=={'output'} and isinstance(value,dict):
        inputs, expected=value,target['output']
    elif given[1]=='output' and set(target)=={'input'} and isinstance(target['input'],dict):
        inputs,expected=target['input'],value
    else: return None
    # Replace the reasoning request so the training target is a complete direct
    # answer, not a truncated thought trace. Preserve the actual task and code.
    prompt=prompt.replace('Please reason and put your final answer in the following json format:',
                          'Return only your final answer in the following JSON format:')
    return {'prompt':prompt,'answer':json.dumps(target,ensure_ascii=False),
            'code':code,'inputs':inputs,'expected':expected,'group':group}


def instruction_pair(text):
    for names in [('### Instruction','### Reasoning','### Response'),('# Query','# Analysis','# Response')]:
        a,b,c=names
        if not text.startswith(a+'\n'): continue
        if text.count('\n'+b+'\n')!=1 or text.count('\n'+c+'\n')!=1: return None
        prompt,rest=text[len(a):].split('\n'+b+'\n')
        _,answer=rest.split('\n'+c+'\n')
        prompt,answer=prompt.strip(),answer.strip()
        if re.search(r'NAME_\d|<\|im_|</?think>|for (?:Engineer|Analysis|Decision Making) for',prompt+'\n'+answer): return None
        if not (10<=len(prompt)<=6000 and 20<=len(answer)<=9000): return None
        if generation_metrics(answer,[],None)['loop_flag']: return None
        return prompt,answer
    return None


def split_for(group):
    n=int(digest('horizon-split-v1:'+group)[:8],16)%100
    return 'test' if n<5 else 'validation' if n<10 else 'train'


def prepare(raw, out, tokenizer_path, max_length=2048):
    tok=load_tokenizer(tokenizer_path)
    rows=[]; counts=collections.Counter(); seen=set()
    def add(source,prompt,answer,group,verification,extra=None):
        key=digest(re.sub(r'\s+',' ',prompt).strip().casefold())
        if key in seen: counts['duplicate']+=1;return
        messages=[{'role':'user','content':prompt},{'role':'assistant','content':answer}]
        enc=encode_complete(tok,messages,max_length)
        if not enc: counts[source['kind']+':too_long']+=1;return
        seen.add(key)
        row={'id':digest(source['source_file']+':'+str(source['row'])),
             'kind':source['kind'],'source_file':source['source_file'],'source_row':source['row'],
             'source_text_sha256':digest(source['text']),'problem_group':group,'split':split_for(group),
             'verification':verification,'messages':messages,'input_tokens':len(enc['input_ids']),
             'assistant_tokens':len(enc['targets']), **(extra or {})}
        rows.append(row);counts[source['kind']+':accepted']+=1
    pending=[]
    for path in sorted(raw.glob('*.jsonl')):
        with path.open() as f:
            for line in f:
                s=json.loads(line); kind=s['kind'];counts[kind+':scanned']+=1
                if kind=='math':
                    prompt=s['text'].split('\n\n',1)[0].removesuffix('\\n').strip()
                    solved=solve_math(prompt)
                    if solved:
                        answer,check=solved
                        add(s,prompt+'\nGive only the final answer.',answer,digest(prompt.casefold()),
                            'independently_recomputed_numeric_grammar',{'check':{'kind':check,'expected':answer}})
                elif kind=='instruction':
                    pair=instruction_pair(s['text'])
                    if pair: add(s,*pair,digest(re.sub(r'\s+',' ',pair[0]).casefold()),'source_curated_not_independently_verified')
                else:
                    parsed=parse_code(s['text'])
                    if parsed:
                        # Token-filter before reference execution to avoid running
                        # programs that could never enter this bounded pilot.
                        enc=encode_complete(tok,[{'role':'user','content':parsed['prompt']},
                                                 {'role':'assistant','content':parsed['answer']}],max_length)
                        if enc: pending.append((s,parsed))
        print(path.name,dict(counts),'pending_code',len(pending),flush=True)
    def verify(pair):
        s,p=pair;result=run_reference(p['code'],p['inputs'])
        return s,p,bool(result and equivalent(result['result'],p['expected']))
    # Validate the isolation mechanism before trusting any reference result.
    check=run_reference('def main_solution(x):\n return x+1',{'x':3})
    if check!={'ok':True,'result':4}: raise RuntimeError('Local isolated verifier unavailable')
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i,(s,p,ok) in enumerate(pool.map(verify,pending)):
            if ok: add(s,p['prompt'],p['answer'],p['group'],'isolated_reference_execution',
                       {'reference_code':p['code'],'check':{'kind':'codeio','inputs':p['inputs'],
                        'expected_output':p['expected'],'target':json.loads(p['answer'])}})
            else: counts['code:reference_rejected']+=1
            if (i+1)%1000==0:print('verified',i+1,dict(counts),flush=True)
    random.Random(251025).shuffle(rows)
    out.mkdir(parents=True,exist_ok=True)
    for split in ['train','validation','test']:
        with (out/(split+'.jsonl')).open('w') as f:
            for row in rows:
                if row['split']==split:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest={'counts':dict(counts),'max_length':max_length,'tokenizer':str(tokenizer_path),
              'raw_manifest':json.loads((raw/'manifest.json').read_text()),'splits':{},
              'instruction_quality':'Source-curated final responses with explicit boundaries; not a factual correctness guarantee.',
              'code_quality':'Reference-program agreement in a networkless bwrap sandbox; not a proof the reference matches every natural-language requirement.',
              'math_quality':'Targets recomputed independently; original reasoning is excluded.'}
    for split in ['train','validation','test']:
        selected=[r for r in rows if r['split']==split]
        manifest['splits'][split]={'rows':len(selected),'by_kind':dict(collections.Counter(r['kind'] for r in selected)),
            'assistant_tokens':sum(r['assistant_tokens'] for r in selected),
            'input_tokens':sum(r['input_tokens'] for r in selected),
            'sha256':hashlib.sha256((out/(split+'.jsonl')).read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['splits'],indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw',type=Path,default=Path('target/manifold-healing/horizon-raw'))
    ap.add_argument('--out',type=Path,default=Path('target/manifold-healing/horizon-prepared'))
    ap.add_argument('--tokenizer',type=Path,default=Path('target/manifold-scale/qwen27'))
    args=ap.parse_args();prepare(args.raw,args.out,args.tokenizer)
