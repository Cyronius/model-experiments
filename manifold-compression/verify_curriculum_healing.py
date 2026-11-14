"""Independent reconstruction plus prespecified within-topic science probes."""
import json
from curriculum_healing import OUT, SOURCE, assess
from load_broad_healing import load
from correctness_heal import load_adapter
from local_projection_probe import save
from scale_data import load_tokenizer
from curriculum_numeric_audit import audit_file


def main():
    final=OUT/'independent-reload-generations.json'
    if final.exists():raise ValueError('Refusing to reuse output for an independent reload check')
    data=json.loads((OUT/'data.json').read_text());probes=json.loads((OUT/'science-transfer-cases.json').read_text())['cases']
    math_cases=json.loads((OUT/'math-form-cases.json').read_text())['cases']
    model=load();tok=load_tokenizer(SOURCE)
    assess(model,tok,probes,OUT/'science-transfer-parent.json')
    assess(model,tok,math_cases,OUT/'math-form-parent.json')
    save(audit_file(OUT/'math-form-parent.json',math_cases),OUT/'math-form-parent-content-audit.json')
    load_adapter(model,OUT/'selected')
    assess(model,tok,probes,OUT/'science-transfer-selected.json')
    assess(model,tok,math_cases,OUT/'math-form-selected.json')
    save(audit_file(OUT/'math-form-selected.json',math_cases),OUT/'math-form-selected-content-audit.json')
    # Exploratory learning diagnostic for the unselected final checkpoint;
    # never substitutes for the development-selected model's held-out score.
    load_adapter(model,OUT/'step-16')
    assess(model,tok,probes,OUT/'science-transfer-final32.json')
    assess(model,tok,math_cases,OUT/'math-form-final32.json')
    save(audit_file(OUT/'math-form-final32.json',math_cases),OUT/'math-form-final32-content-audit.json')
    load_adapter(model,OUT/'selected')
    cases=data['test'][:4]+[c for c in data['legacy'] if c['family']=='explanation']
    assess(model,tok,cases,final)
    expected={c['id']:c for name in ['test-selected.json','legacy-selected.json'] for c in json.loads((OUT/name).read_text())['cases']}
    checks={c['id']:{k:c[k]==expected[c['id']][k] for k in ['answer','ended','generated_tokens']} for c in json.loads(final.read_text())['cases']}
    passed=all(all(v.values()) for v in checks.values())
    save({'fresh_process':True,'pinned_source_verified':True,'cases':checks,'exact_match':passed},OUT/'independent-reload.json')
    if not passed:raise RuntimeError('Fresh reconstruction changed answers')
    print('Independent reconstruction matches six answers, including both legacy explanations.')

if __name__=='__main__':main()
