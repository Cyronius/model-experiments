"""Extend the exact original calibration sampler to 256k tokens, locally."""
import gc
import json

import torch
from transformers import AutoTokenizer
from transformers.utils.logging import disable_progress_bar

from local_projection_probe import (ROOT,SOURCE,DEFAULT_OUT,collect_packet,load_source,
                                    eig,evaluate,save,log)
from local_projection_models import fit_rms,rms_error
from healing_data import encode_complete
from scale_data import load_tokenizer,prepare


def main(out=DEFAULT_OUT):
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(251025)
    disable_progress_bar()
    source_file=out/'extended-calibration-source.json'
    if not source_file.exists():
        toks={'qwen2':load_tokenizer(SOURCE),
              'qwen27':load_tokenizer(ROOT/'target/manifold-scale/qwen27'),
              'mistral24':load_tokenizer(ROOT/'target/manifold-scale/mistral24')}
        prepare(ROOT/'target/manifold-audit/ultrachat',toks,source_file,
                calibration_tokens=262144,validation_rows=32,max_length=768)
        del toks;gc.collect()
    tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
    original=json.loads((out/'data.json').read_text())
    rows=[]
    for row in json.loads(source_file.read_text())['calibration']:
        encoded=encode_complete(tok,row['messages'],768)
        if encoded is None:raise ValueError('Unexpected rejected original-sampler row')
        rows.append({'id':row['id'],'messages':row['messages'],**encoded})
    assert rows[:len(original['large'])]==original['large'], '256k corpus must exactly extend the original 64k corpus'
    prompts={r['messages'][0]['content'].strip().casefold() for r in rows}
    assert not prompts.intersection(r['messages'][0]['content'].strip().casefold()
                                    for r in original['validation']+original['cases'])
    save({'tokens':sum(len(r['input_ids']) for r in rows),'rows':len(rows),
          'exact_64k_prefix':True,'heldout_prompt_overlap':0,'sampler':'scale_data.prepare, same three tokenizers, seed 24027, complete responses, max_length 768',
          'whole_model_healing':False,'cloud_spend_usd':0},out/'extended-calibration-protocol.json')
    packet_path=out/'extended-states.pt'
    if not packet_path.exists():
        model=load_source();collect_packet(model,rows,packet_path)
        del model;gc.collect()
    recipe_path=out/'shared-extended-recipe.pt'
    if not recipe_path.exists():
        packet=torch.load(packet_path,weights_only=True)
        xs=packet['samples'];covs=packet['covs'];layers=(len(covs)-1)//2
        q,values=eig(sum(c.double()/c.double().trace() for c in covs))
        torch.save({'q':q,'values':values},out/'extended-pca.pt')
        width=json.loads((out/'protocol.json').read_text())['global_width']
        p=q[:,:width].contiguous()
        val=torch.load(out/'validation-states.pt',weights_only=True)['samples']
        weights=torch.tensor([len(rows[g]['input_ids'])/min(32,len(rows[g]['input_ids']))
                              for g in packet['groups'].tolist()])
        scales=[];errors=[]
        for i,x in enumerate(xs):
            scale,predictor=fit_rms(x,p,weights=weights);scales.append(scale)
            errors.append(rms_error(val[i],p,scale,predictor))
        recipe={'basis':q.unsqueeze(0).expand(layers,-1,-1),'width':width,'local':False,
                'scales':scales,'predictors':None,'source':'Qwen/Qwen3.5-2B','calibration':'extended'}
        torch.save(recipe,recipe_path)
        save({'heldout_rms':errors},out/'extended-fit-summary.json')
        del packet,xs,covs,q,val,p,weights,x;gc.collect()
    # All answer comparisons use the same four-thread setting as the other arms.
    torch.set_num_threads(4)
    evaluate(out,'shared-extended')
    log('256k-token calibration comparison complete')


if __name__=='__main__':main()
