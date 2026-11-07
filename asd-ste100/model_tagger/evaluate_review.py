"""Keep the proposed software review set isolated, then score without tuning."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch
from tagger import ROOT, Config, Tagger, SteTokenizer, Lexicon
from safe_editor import rewrite, validate
from audit_splits import key
from prepare import records,input_paths
from copy_fidelity import preserved

BENCHMARK=ROOT/'model_tagger/benchmarks/software-review-v1.jsonl'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('isolation','score'))
    p.add_argument('--pairs',type=Path)
    p.add_argument('--benchmark',type=Path,default=BENCHMARK)
    p.add_argument('--vocab',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--policy',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cases=list(records(args.benchmark))
    base=dict(benchmark_sha256=sha(args.benchmark),code_sha256=sha(Path(__file__)),cases=len(cases),review_status='assistant_reviewed_owner_pending')
    if args.command=='isolation':
        keys={key(text):case['id'] for case in cases for text in [case['source'],*case['acceptable_outputs']]}
        overlaps=[];scanned=0
        for path,_ in input_paths(args.pairs):
            for row in records(path):
                scanned+=1
                for field in ('src','ste'):
                    code=key(row[field])
                    if code in keys:overlaps.append(dict(benchmark=keys[code],pair=row['id'],split=row['split'],field=field))
        report=dict(base,passed=not overlaps,pairs_scanned=scanned,overlaps=overlaps)
    else:
        torch.set_num_threads(4)
        tok=SteTokenizer(args.vocab);lex=Lexicon()
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        policy=json.loads(args.policy.read_text())
        if state['vocab_sha256']!=sha(args.vocab/'vocab.json') or policy['vocab_sha256']!=state['vocab_sha256'] or policy['checkpoint_sha256']!=sha(args.checkpoint):
            raise ValueError('policy/checkpoint/vocabulary mismatch')
        model=Tagger(tok,Config(**state['model_config']));model.load_state_dict(state['model'])
        results=[];counts=Counter()
        for case in cases:
            result=rewrite(model,tok,lex,case['source'],case['mode'],policy['max_passes'],policy['edit_margin'])
            exact=result['text'] in case['acceptable_outputs']
            copies=preserved(result['text'],[dict(surface=s,count=case['source'].count(s)) for s in case['must_preserve']])
            guard_issues=validate(case['source'],result['text'])
            counts[case['category']+'_cases']+=1;counts[case['category']+'_exact']+=exact
            counts['literal_failures']+=not copies;counts['guard_failures']+=bool(guard_issues);counts['unresolved']+=not result['resolved']
            results.append(dict(case,exact_accepted_text=exact,literals_preserved=copies,guard_issues=guard_issues,result=result))
        report=dict(base,checkpoint_sha256=sha(args.checkpoint),policy=policy,counts=dict(counts),predictions=results,
                    limits=['Exact references and deterministic guards do not establish semantic correctness.','Owner review pending; no training or threshold tuning on this set.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='predictions'},indent=2))

if __name__=='__main__':main()
