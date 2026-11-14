import argparse, json, sys
from pathlib import Path
sys.path.insert(0, '/home/cyrus/code/model-experiments/manifold-compression')
import unhealed_generation as ug
sp = Path(sys.argv[1])
suite = json.loads((sp / 'dry-suite/suite.json').read_text())
ug.OUT = sp / 'dry-run'
ug.OUT.mkdir()
ug.PEER_FILES = []
pick = {'track-a': ['rh-test-retain-arithmetic-0', 'rh-test-program-3'], 'humaneval': ['HumanEval-23', 'HumanEval-0'],
        'gsm8k': ['gsm8k-1', 'gsm8k-1199'], 'science-new': ['dummy-1']}
parts = {p: [dict(c, max_new_tokens=96) for c in suite['parts'][p] if c['id'] in ids] for p, ids in pick.items()}
(ug.OUT / 'suite.json').write_text(json.dumps(suite | {'parts': parts}))
rules = {k: {} for k in ug.RULE_KEYS} | {'arms': ['stock', 'P1']}
(ug.OUT / 'rules.json').write_text(json.dumps(rules))
ug.stage_freeze(argparse.Namespace(rules=ug.OUT / 'rules.json', reviewers=['claude', 'codex'],
                max_hours={'stock': 0.25, 'P1': 0.25}, max_total_hours=0.5, threads=16, batch=16))
for arm in ('P1', 'stock'):
    for part in ug.PART_ORDER:
        ug.stage_generate(argparse.Namespace(arm=arm, part=part, threads=16, batch=16))
    ug.stage_grade(argparse.Namespace(arm=arm))
    g = json.loads((ug.OUT / f'grades/{arm}.json').read_text())
    for id_, r in g['cases'].items():
        print(arm, id_, r['ended'], r['loop'], r['grade'] and {k: v for k, v in r['grade'].items() if k in ('correct', 'final_answer_method', 'bare_final_line', 'flexible_correct', 'syntax_pass')})
print(json.loads((ug.OUT / 'clock.json').read_text()))
