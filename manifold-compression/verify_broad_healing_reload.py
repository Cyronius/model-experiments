"""Independent reconstruction check, including the two legacy long answers."""
import json
from broad_healing import OUT
from load_broad_healing import load
from local_projection_probe import SOURCE, evaluate_generations, save
from scale_data import load_tokenizer


def main():
    destination=OUT/'independent-reload-generations.json'
    if destination.exists():raise ValueError('Refusing to reuse prior generation output for a fresh reload check')
    data=json.loads((OUT/'data.json').read_text())
    cases=data['test'][:4]+[c for c in data['legacy'] if c['family']=='explanation']
    # These are complete original batches: first four closed cases and both
    # legacy explanations. Different padding/batch composition can alter output.
    model=load();tok=load_tokenizer(SOURCE)
    evaluate_generations(model,tok,cases,destination)
    expected={c['id']:c for name in ['test-selected.json','legacy-selected.json']
              for c in json.loads((OUT/name).read_text())['cases']}
    actual=json.loads(destination.read_text())['cases']
    checks={c['id']:{k:c[k]==expected[c['id']][k] for k in ['answer','ended','generated_tokens']} for c in actual}
    passed=all(all(v.values()) for v in checks.values())
    save({'fresh_process':True,'source_hashes_verified':True,'cases':checks,'exact_match':passed},OUT/'independent-reload.json')
    if not passed:raise RuntimeError('Independent reconstruction changed generated answers')
    print('Independent reconstruction reproduces all six answers, including both legacy explanations.')

if __name__=='__main__':main()
