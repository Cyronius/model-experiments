"""Re-execute every selected reference and recompute every selected math target."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time

from correctness_prepare import OUT,pure_reference
from horizon_data import run_reference,equivalent,solve_math


def verify(source,out):
    path=out/'data.json';data=json.loads(path.read_text());prep=json.loads((out/'preparation.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=prep['data_sha256']:raise ValueError('Data changed')
    rows=[r for r in data['train'] if r['kind']=='code']
    evalids={c['id'] for c in data['dev']+data['test'] if c['family']=='horizon_code'}
    for split in ['validation','test']:
        content=(source/(split+'.jsonl')).read_bytes()
        if hashlib.sha256(content).hexdigest()!=prep['prepared_source_manifest']['splits'][split]['sha256']:
            raise ValueError('Prepared held-out source changed')
        rows += [r for r in map(json.loads,content.decode().splitlines()) if r['id'] in evalids]
    if len({r['id'] for r in rows}&evalids)!=len(evalids):raise ValueError('Missing held-out references')
    def check(row):
        if not pure_reference(row['reference_code']):return row['id'],False
        result=run_reference(row['reference_code'],row['check']['inputs'])
        return row['id'],bool(result and equivalent(result['result'],row['check']['expected_output']))
    start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(check,rows))
    failed=[i for i,ok in results if not ok]
    math_rows=[r for r in data['train'] if r['kind']=='math']
    for row in math_rows:
        solved=solve_math(row['messages'][0]['content'].removesuffix('\nGive only the final answer.'))
        if not solved or solved[0]!=row['messages'][-1]['content']:failed.append(row['id'])
    report={'data_sha256':prep['data_sha256'],
        'verifier_sha256':hashlib.sha256((Path(__file__).parent/'horizon_data.py').read_bytes()).hexdigest(),
        'checked_reference_examples':len(rows),'recomputed_math_examples':len(math_rows),
        'failed_ids':failed,'seconds':time.monotonic()-start,'isolated_network_filesystem_environment':True,
        'set_order_dependent_programs_excluded':True}
    (out/'verification-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if failed:raise ValueError('Selected target verification failed')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path('target/manifold-healing/horizon-prepared'))
    ap.add_argument('--out',type=Path,default=OUT)
    args=ap.parse_args();verify(args.source,args.out)
