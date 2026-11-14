"""Fit per-site linear decoders using full calibration covariance, then fold.

The encoder and compressed residual coordinates remain fixed. Each decoder is
D = P + R A.T, with P.T D = I; only the reader matrices and final mixing matrix
change. No deployed parameter is added. No answer labels or model-loss training
are used. Compare with shared-extended, which uses the same encoder and scales.
"""
import gc
import json

import torch
from transformers.utils.logging import disable_progress_bar

from local_projection_probe import DEFAULT_OUT,save,log,evaluate


@torch.no_grad()
def fit(out=DEFAULT_OUT):
    recipe=torch.load(out/'shared-extended-recipe.pt',weights_only=True)
    q=recipe['basis'][0].double();k=recipe['width'];p,r=q[:,:k],q[:,k:]
    packet=torch.load(out/'extended-states.pt',weights_only=True)
    covs=packet['covs'];tokens=packet['tokens']
    del packet;gc.collect()
    validation=torch.load(out/'validation-states.pt',weights_only=True)['samples']
    decoders=[];audit=[]
    for site,cov in enumerate(covs):
        c=cov.double()/tokens;c=(c+c.T)*.5
        gram=p.T@c@p;cross=p.T@c@r
        std=gram.diagonal().sqrt().clamp_min(1e-6)
        # Fixed ridge strength, matching the decoder probe; no validation tuning.
        coefficient=torch.linalg.solve(gram/std[:,None]/std[None,:]+.1*torch.eye(k),cross/std[:,None])/std[:,None]
        decoder=(p+r@coefficient.T).float()
        identity_error=(p.T@decoder.double()-torch.eye(k)).abs().max().item()
        if identity_error>1e-4:raise RuntimeError('Decoder changes retained coordinates')
        x=validation[site].float();z=x@p.float()
        before=(x-z@p.float().T).square().mean().item()
        after=(x-z@decoder.T).square().mean().item()
        audit.append({'site':site,'orthogonal_mse':before,'fitted_mse':after,
                      'recovered_discarded_fraction':1-after/before,'retained_coordinate_error':identity_error})
        decoders.append(decoder)
        if (site+1)%8==0:log(f'Full-covariance linear decoder {site+1}/{len(covs)}')
    recipe=dict(recipe,decoders=decoders,decoder_method='Full 262k-token covariance, standardized zero-intercept ridge 0.1; readers only; same encoder/scales/parameter count')
    torch.save(recipe,out/'shared-extended-linear-recipe.pt')
    save({'tokens':tokens,'ridge':.1,'heldout':audit,'deployment_parameter_increase':0,
          'scope':'Analytic activation reconstruction folded into weights, without whole-model training. Held-out answer tests determine usefulness.'},out/'linear-decoder-audit.json')


def main(out=DEFAULT_OUT):
    torch.set_num_threads(8);torch.set_num_interop_threads(1);disable_progress_bar()
    if not (out/'shared-extended-linear-recipe.pt').exists():fit(out)
    gc.collect();torch.set_num_threads(4)
    evaluate(out,'shared-extended-linear')
    baseline=json.loads((out/'shared-extended-result.json').read_text())
    current=json.loads((out/'shared-extended-linear-result.json').read_text())
    assert baseline['parameters']==current['parameters']
    assert baseline['stored_scalars']==current['stored_scalars']
    log('Fitted linear reconstruction evaluated at exactly matched model storage')


if __name__=='__main__':main()
