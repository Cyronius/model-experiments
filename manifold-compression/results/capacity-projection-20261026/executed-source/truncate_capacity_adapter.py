"""Post-hoc rank compression diagnostic with a full-rank SVD control."""
import argparse,json
import torch
from capacity_projection import OUT,SOURCE,evaluate
from load_capacity_projection import load
from correctness_heal import LowRankLinear,save_adapter,load_adapter
from local_projection_probe import save
from scale_data import load_tokenizer
from audit_rotation import cache_check
from broad_healing import sha

DEST=OUT/'rank-truncation'

def factorize(a,b,rank):
    """Best rank-k approximation of B@A via a small core SVD; preserve scale."""
    qb,rb=torch.linalg.qr(b.double(),mode='reduced')
    qa,ra=torch.linalg.qr(a.T.double(),mode='reduced')
    u,s,vh=torch.linalg.svd(rb@ra.T,full_matrices=False)
    k=min(rank,len(s));root=s[:k].sqrt()
    anew=torch.zeros(rank,a.shape[1],dtype=a.dtype);bnew=torch.zeros(b.shape[0],rank,dtype=b.dtype)
    anew[:k]=((root[:,None]*vh[:k])@qa.T).to(a.dtype)
    bnew[:,:k]=((qb@u[:,:k])*root).to(b.dtype)
    return anew,bnew,s


def install_factors(model,weights,rank):
    names=[]
    for name,m in list(model.named_modules()):
        if not isinstance(m,LowRankLinear):continue
        new=LowRankLinear(m.base,rank,2*rank)
        with torch.no_grad():new.a.copy_(weights[name+'.a']);new.b.copy_(weights[name+'.b'])
        parent,leaf=name.rsplit('.',1);setattr(model.get_submodule(parent),leaf,new);names.append(name)
    assert names
    return {'rank':rank,'alpha':2*rank,'modules':names,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}


def load_variant(name):
    if name not in ['full-rank-control','rank16-truncated']:raise ValueError('Unknown diagnostic variant')
    from safetensors.torch import load_file
    protocol=json.loads((DEST/'protocol.json').read_text())
    assert sha(OUT/'projected-r64/step-8/adapter.safetensors')==protocol['source_adapter_sha256']
    metadata=json.loads((DEST/name/'adapter.json').read_text());model=load('projected-r64')
    weights=load_file(str(DEST/name/'adapter.safetensors'))
    expected={n+suffix for n in metadata['modules'] for suffix in ['.a','.b']}
    assert set(weights)==expected
    info=install_factors(model,weights,metadata['rank']);assert info['modules']==metadata['modules']
    return model.eval()


def verify():
    data=json.loads((OUT/'data.json').read_text());model=load_variant('rank16-truncated');tok=load_tokenizer(SOURCE)
    cases=[c for c in data['test'] if c['max_new_tokens']==256][:4]
    path=DEST/'rank16-truncated/reload-generations.json';evaluate(model,tok,cases,path)
    prior={c['id']:c for c in json.loads((DEST/'rank16-truncated/test.json').read_text())['cases']}
    actual=json.loads(path.read_text())['cases']
    same=all(all(c[k]==prior[c['id']][k] for k in ['answer','ended','generated_tokens']) for c in actual)
    save({'exact_match':same,'cases':[c['id'] for c in actual]},DEST/'rank16-truncated/reload.json');assert same
    print('Truncated adapter independently reproduces four answers.',flush=True)


def prepare():
    if (DEST/'protocol.json').exists():raise ValueError('Diagnostic already declared')
    save({'scope':'Exploratory post-hoc diagnostic, outside primary fixed 2x2 endpoints.',
      'declared_after':'Projected rank64 scores 10/39 versus rank16 5/39; stock-r16 complete. Before stock-r64 trained evaluation.',
      'source_arm':'projected-r64','source_step':8,'source_adapter_sha256':sha(OUT/'projected-r64/step-8/adapter.safetensors'),
      'variants':['full-rank-control','rank16-truncated'],'method':'Double-precision thin QR of B and A.T; SVD of their small core. Balance factors with square-root singular values, store FP32. Keep alpha/rank=2. No retraining or evaluation-dependent tuning.',
      'evaluation':'Same 47 cases, decoding and numeric audit as primary study. Full-rank control must be inspected for generation equivalence. Separate post-hoc program edge audit applies.',
      'interpretation':'If rank16 truncation preserves gains, a rank16 final update can represent those gains on this set. This implicates training trajectory/optimization rather than a necessary rank>16 final update. A failed truncation does not prove every rank16 solution is inferior.',
      'cloud_spend':0},DEST/'protocol.json')


def run():
    if (DEST/'result.json').exists() or (DEST/'full-rank-control').exists():raise ValueError('Existing diagnostic; explicit recovery required')
    p=json.loads((DEST/'protocol.json').read_text());assert sha(OUT/'projected-r64/step-8/adapter.safetensors')==p['source_adapter_sha256']
    torch.set_num_threads(4);model=load('projected-r64');tok=load_tokenizer(SOURCE);data=json.loads((OUT/'data.json').read_text())
    full={};cut={};spectra={};total=0.;kept=0.
    for name,m in model.named_modules():
        if not isinstance(m,LowRankLinear):continue
        a,b,s=factorize(m.a.detach(),m.b.detach(),64);aa,bb,ss=factorize(m.a.detach(),m.b.detach(),16)
        full[name+'.a']=a;full[name+'.b']=b;cut[name+'.a']=aa;cut[name+'.b']=bb
        energy=float(s.square().sum());retained=float(s[:16].square().sum());total+=energy;kept+=retained
        spectra[name]={'singular_values':s.tolist(),'rank16_energy_fraction':retained/energy if energy else 1.}
    save({'modules':spectra,'global_rank16_update_energy_fraction':kept/total,'scope':'Frobenius energy of LoRA weight updates, not activation-weighted importance or correctness.'},DEST/'spectra.json')
    ids=torch.tensor([data['train'][data['updates'][0][0]]['input_ids'][:32]])
    model.eval()
    with torch.no_grad():before=model(ids,use_cache=False).logits
    result={}
    for name,rank,weights in [('full-rank-control',64,full),('rank16-truncated',16,cut)]:
        info=install_factors(model,weights,rank);model.eval();out=DEST/name
        metadata=json.loads((OUT/'projected-r64/step-8/adapter.json').read_text());metadata.update(info);metadata.update({'diagnostic':name,'source_adapter_sha256':p['source_adapter_sha256']})
        save_adapter(model,out,metadata)
        with torch.no_grad():after=model(ids,use_cache=False).logits
        numerical={'max_abs_logit_difference':float((before-after).abs().max()),'top1_agreement':float((before.argmax(-1)==after.argmax(-1)).float().mean())}
        if rank==64:torch.testing.assert_close(before,after,atol=1e-4,rtol=1e-4)
        cache=cache_check(model,ids);assert cache['numerical_pass'];save({'original_vs_variant':numerical,'cache':cache},out/'gates.json')
        result[name]=evaluate(model,tok,data['test'],out/'test.json')
        print(name,json.dumps(result[name]),flush=True)
    original=json.loads((OUT/'projected-r64/test-8-content-audit.json').read_text());control=json.loads((DEST/'full-rank-control/test-content-audit.json').read_text())
    old={c['id']:c for c in original['cases']}
    differences=[c['id'] for c in control['cases'] if any(c[k]!=old[c['id']][k] for k in ['answer','ended','generated_tokens','grade'])]
    result['full_rank_control_changed_cases']=differences;result['global_rank16_update_energy_fraction']=kept/total
    save(result,DEST/'result.json');print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','verify']);a=parser.parse_args()
    {'prepare':prepare,'run':run,'verify':verify}[a.action]()
