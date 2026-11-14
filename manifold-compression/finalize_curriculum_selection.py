"""Apply the declared content-grading correction before final reporting."""
import json
from pathlib import Path
import shutil
import torch
from curriculum_healing import OUT, PARENT, SOURCE, RECIPE, assess
from curriculum_numeric_audit import audit_file, select
from load_broad_healing import load
from correctness_heal import load_adapter, save_adapter
from local_projection_probe import save, grade_case
from broad_healing import sha, OLD
from scale_data import load_tokenizer
from audit_rotation import cache_check


def main():
    decision=select(OUT);data=json.loads((OUT/'data.json').read_text());result=json.loads((OUT/'result.json').read_text())
    winner=decision['selected_stage_step']
    if not decision['winner_unchanged']:
        archive=OUT/'provisional-selection';archive.mkdir(exist_ok=False)
        for name in ['selected','result.json','test-selected.json','legacy-selected.json','gates.json']:
            shutil.move(str(OUT/name),str(archive/name))
        model=load(OUT/f'step-{winner}');tok=load_tokenizer(SOURCE)
        meta=json.loads((OUT/f'step-{winner}/adapter.json').read_text())
        meta.update({'stage_step':winner,'total_step':16+winner,'parent_adapter_sha256':data['protocol']['parent_adapter_sha256'],'data_sha256':sha(OUT/'data.json'),'recipe_sha256':sha(RECIPE),'selection_metric':'numeric-content-audit-v1'})
        save_adapter(model,OUT/'selected',meta)
        ids=torch.tensor([data['train'][0]['input_ids'][:64]])
        with torch.no_grad():before=model(ids,use_cache=False).logits
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:p.zero_()
        load_adapter(model,OUT/'selected')
        with torch.no_grad():torch.testing.assert_close(before,model(ids,use_cache=False).logits,atol=0,rtol=0)
        gate=cache_check(model,ids);assert gate['numerical_pass']
        save({'adapter_reload_exact':True,'cache':gate},OUT/'gates.json')
        result['fresh_after']=assess(model,tok,data['test'],OUT/'test-selected.json')
        result['legacy_after']=assess(model,tok,data['legacy'],OUT/'legacy-selected.json')
        result['selected_stage_step']=winner;result['selected_total_step']=16+winner
        save(result,OUT/'result.json')
    else:
        meta=json.loads((OUT/'selected/adapter.json').read_text());meta['selection_metric']='numeric-content-audit-v1';save(meta,OUT/'selected/adapter.json')
    for name,split in [('test-0','test'),('test-selected','test'),('stock-test','test'),('legacy-selected','legacy')]:
        save(audit_file(OUT/(name+'.json'),data[split]),OUT/(name+'-content-audit.json'))
    save(audit_file(PARENT/'legacy-selected.json',data['legacy']),OUT/'legacy-parent-content-audit.json')
    stock=json.loads((OLD/'stock-generations.json').read_text());specs={c['id']:c for c in data['legacy']}
    # The original stored stock sorting grades predate the project's uniform
    # integer-string content convention. Apply that established convention.
    for c in stock['cases']:
        if c['family']=='sorting':c['grade']=grade_case(c['answer'],specs[c['id']])
    save(stock,OUT/'legacy-stock-normalized.json')
    save(audit_file(OUT/'legacy-stock-normalized.json',data['legacy']),OUT/'legacy-stock-content-audit.json')
    print(json.dumps(decision,indent=2))

if __name__=='__main__':main()
