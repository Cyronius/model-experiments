"""Held-out geometry diagnostics for the local projection screen."""
import torch

from local_projection_probe import DEFAULT_OUT,save


@torch.no_grad()
def main(out=DEFAULT_OUT):
    torch.set_num_threads(4)
    small=torch.load(out/'small-pca.pt',weights_only=True)['q']
    large=torch.load(out/'large-pca.pt',weights_only=True)['q']
    recipe=torch.load(out/'shared-large-recipe.pt',weights_only=True)
    k=recipe['width']
    overlap=(small[:,:k].T@large[:,:k]).square().sum().item()/k
    result={'small_large_mean_squared_subspace_overlap':overlap,'width':k,
            'scope':'Representation diagnostics on sampled states; not answer-quality scores. Overlap is mean squared cosine of principal angles.'}
    xs=torch.load(out/'validation-states.pt',weights_only=True)['samples']
    result['heldout_sampled_retained_energy']={}
    arms=['shared-small','shared-large','shared-broad','layer-large-matched','layer-large-samewidth']
    if (out/'shared-extended-recipe.pt').exists():
        arms.append('shared-extended')
        extended=torch.load(out/'extended-pca.pt',weights_only=True)['q']
        result['large_extended_mean_squared_subspace_overlap']=(large[:,:k].T@extended[:,:k]).square().sum().item()/k
    for arm in arms:
        r=torch.load(out/f'{arm}-recipe.pt',weights_only=True)
        ratios=[]
        for i,x in enumerate(xs):
            p=r['basis'][min(i//2,len(r['basis'])-1),:,:r['width']]
            ratios.append((x@p).square().sum().item()/x.square().sum().item())
        result['heldout_sampled_retained_energy'][arm]={'per_site':ratios,'mean':sum(ratios)/len(ratios),'min':min(ratios)}
    save(result,out/'geometry-diagnostics.json')
    print({k:v for k,v in result.items() if k!='heldout_sampled_retained_energy'})
    print({k:{a:b for a,b in v.items() if a!='per_site'} for k,v in result['heldout_sampled_retained_energy'].items()})


if __name__=='__main__':main()
