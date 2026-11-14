"""Reconstruct a matched-study adapter with pinned source and recipe checks."""
import json
from pathlib import Path
import torch
from capacity_projection import OUT, SOURCE, RECIPE, ARMS
from broad_healing import sha
from correctness_heal import install_lora, load_adapter
from local_projection_models import from_recipe
from local_projection_probe import load_source

def load(arm,step=8):
    if arm not in ARMS or step not in [0,4,8]:raise ValueError('Unknown arm or checkpoint')
    p=json.loads((OUT/'protocol.json').read_text());metadata=json.loads((OUT/arm/f'step-{step}/adapter.json').read_text())
    if metadata['data_sha256']!=sha(OUT/'data.json'):raise ValueError('Study data changed')
    for name,record in p['source_manifest']['files'].items():
        if sha(Path(SOURCE)/name)!=record['sha256']:raise ValueError('Pinned source differs: '+name)
    torch.set_num_threads(4)
    if arm.startswith('projected'):
        if sha(RECIPE)!=p['recipe_sha256'] or metadata['recipe_sha256']!=p['recipe_sha256']:raise ValueError('Projection recipe changed')
        model=from_recipe(SOURCE,torch.load(RECIPE,weights_only=True,map_location='cpu'),load_source)
    else:model=load_source(SOURCE)
    info=install_lora(model,metadata['rank'],metadata['alpha'])
    if info['modules']!=metadata['modules']:raise ValueError('Adapter module inventory differs')
    load_adapter(model,OUT/arm/f'step-{step}')
    return model.eval()

if __name__=='__main__':
    import argparse,gc
    from local_projection_probe import save
    from curriculum_healing import assess
    from scale_data import load_tokenizer
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=ARMS);args=parser.parse_args()
    data=json.loads((OUT/'data.json').read_text());cases=[c for c in data['test'] if c['max_new_tokens']==256][:4]
    model=load(args.arm);tok=load_tokenizer(SOURCE);path=OUT/args.arm/'independent-reload-generations.json'
    from audit_rotation import cache_check
    ids=torch.tensor([data['train'][data['updates'][0][0]]['input_ids'][:32]])
    cache=cache_check(model,ids);assert cache['numerical_pass']
    assess(model,tok,cases,path)
    expected={c['id']:c for c in json.loads((OUT/args.arm/'test-8.json').read_text())['cases']}
    actual=json.loads(path.read_text())['cases'];checks={c['id']:{k:c[k]==expected[c['id']][k] for k in ['answer','ended','generated_tokens']} for c in actual}
    result={'fresh_reconstruction':True,'final_cache':cache,'cases':checks,'exact_match':all(all(v.values()) for v in checks.values())}
    save(result,OUT/args.arm/'independent-reload.json');assert result['exact_match'];print(json.dumps(result),flush=True)
