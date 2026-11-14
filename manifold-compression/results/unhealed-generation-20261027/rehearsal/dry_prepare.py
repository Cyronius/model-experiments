import argparse, json, sys
from pathlib import Path
sys.path.insert(0, '/home/cyrus/code/model-experiments/manifold-compression')
import unhealed_generation as ug
sp = Path(sys.argv[1])
ug.OUT = sp / 'dry-suite'
science = sp / 'dummy-science.json'
science.write_text(json.dumps({'items': [{'id': 'dummy-1', 'family': 'g1', 'messages': [{'role': 'user', 'content': 'Why does ice float on water?'}],
    'reference_answer': 'Ice is less dense than liquid water because hydrogen bonds hold it in an open lattice.', 'source_url': 'x',
    'manual_rubric': {'essential': ['less dense'], 'optional': ['hydrogen bonds']}}]}))
ug.stage_prepare(argparse.Namespace(science=science))
suite = json.loads((ug.OUT / 'suite.json').read_text())
for name, cases in suite['parts'].items():
    print(name, len(cases), cases[0]['id'], cases[-1]['id'], sorted({c['max_new_tokens'] for c in cases}))
print(suite['humaneval_subsets'], suite['gsm8k_pool'])
