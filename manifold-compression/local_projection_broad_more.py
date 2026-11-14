"""Extend the frozen broad calibration mixture to 256k tokens, without healing."""
import argparse
import gc
import hashlib
import json
import random
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
from transformers.utils.logging import disable_progress_bar

from healing_data import encode_complete
from local_projection_models import fit_rms, rms_error
from local_projection_probe import (ROOT, SOURCE, DEFAULT_OUT as BASE, collect_packet,
                                    load_source, eig, evaluate, save, log)

OUT = ROOT/'manifold-compression/results/broad-more-20261025'
ARM = 'shared-broad-extended'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(out=OUT, base=BASE):
    if (out/'protocol.json').exists():
        protocol=json.loads((out/'protocol.json').read_text())
        assert sha(out/'data.json')==protocol['data_sha256']
        return
    original=json.loads((base/'data.json').read_text())
    old_protocol=json.loads((base/'protocol.json').read_text())
    assert sha(base/'data.json')==old_protocol['data_sha256']
    manifest=json.loads((base/'artifact-manifest.json').read_text())
    for name in ('local_projection_models.py','local_projection_probe.py'):
        assert sha(Path(__file__).parent/name)==manifest['../../'+name]['sha256']
    horizon=json.loads((ROOT/'manifold-compression/results/correctness-healing-20261025/data.json').read_text())['train']
    kinds={r['id']:r['kind'] for r in horizon}
    prefix=original['broad']
    targets={'chat':131072,'math':65536,'code':65536}
    totals={k:0 for k in targets};counts={k:0 for k in targets}
    def kind(row):
        return 'chat' if row['id'].startswith('long-') else kinds[row['id']]
    for row in prefix:
        k=kind(row);totals[k]+=len(row['input_ids']);counts[k]+=1
    tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
    seen={r['messages'][0]['content'].strip().casefold()
          for r in prefix+original['large']+original['validation']+original['cases']}
    extra=[]
    def append(raw,k):
        prompt=raw['messages'][0]['content'].strip().casefold()
        if prompt in seen:return
        encoded=encode_complete(tok,raw['messages'],2048)
        if encoded is None or (k=='chat' and len(encoded['input_ids'])<=768):return
        row={'id':raw['id'],'messages':raw['messages'],**encoded}
        extra.append(row);seen.add(prompt)
        totals[k]+=len(row['input_ids']);counts[k]+=1
    source=next((ROOT/'target/manifold-audit/ultrachat/data').glob('train_sft-00000-*.parquet'))
    table=pq.read_table(source,columns=['prompt_id','messages'])
    order=list(range(len(table)));random.Random(25102581).shuffle(order)
    for idx in order:
        raw=table.slice(idx,1).to_pylist()[0];messages=raw['messages'][:2]
        if len(messages)==2 and [m['role'] for m in messages]==['user','assistant']:
            append({'id':'long-'+raw['prompt_id'],'messages':messages},'chat')
        if totals['chat']>=targets['chat']:break
    for k in ('math','code'):
        for raw in horizon:
            if raw['kind']==k:append(raw,k)
            if totals[k]>=targets[k]:break
        if totals[k]<targets[k]:raise ValueError(f'Insufficient independent {k} examples: {totals[k]}/{targets[k]}')
    assert all(totals[k]>=targets[k] for k in targets)
    random.Random(25102583).shuffle(extra)
    rows=prefix+extra
    assert rows[:len(prefix)]==prefix
    ids=[r['id'] for r in rows];prompts=[r['messages'][0]['content'].strip().casefold() for r in rows]
    assert len(ids)==len(set(ids)) and len(prompts)==len(set(prompts))
    forbidden=['\n'.join(m['content'] for m in c['messages']).strip().casefold()
               for c in original['cases']]
    for row in rows:
        text='\n'.join(m['content'] for m in row['messages']).casefold()
        assert not any(p in text for p in forbidden), f'Held-out overlap: {row["id"]}'
    data={'calibration':rows,'prefix_rows':len(prefix),'validation':original['validation'],'cases':original['cases']}
    out.mkdir(parents=True,exist_ok=True)
    save(data,out/'data.json')
    protocol={'model':old_protocol['model'],'source_manifest':old_protocol['source_manifest'],
        'data_sha256':sha(out/'data.json'),'base_directory':str(base),'base_data_sha256':sha(base/'data.json'),
        'plan':old_protocol['plan'],'arm':ARM,'width':old_protocol['global_width'],
        'requested_tokens':262144,'actual_tokens':sum(totals.values()),'rows':len(rows),
        'by_kind':{k:{'tokens':totals[k],'rows':counts[k]} for k in targets},
        'exact_64k_prefix':True,'prefix_rows':len(prefix),'additional_rows':len(extra),
        'heldout_exact_prompt_overlap':0,'max_length':2048,
        'sampling':'Original broad corpus unchanged, then new examples from the same seeded long-chat ordering and verified Horizon training ordering; same 50/25/25 token targets. No repeated examples.',
        'activation_reuse':'Original broad activation packet plus a new packet for only added conversations; uncentered covariance sums combine additively. Per-conversation sampled states retain token weighting.',
        'precision':old_protocol['calibration_precision'],'generation':old_protocol['generation'],
        'comparison':'Complete 2x2: chat-only/broad x 64k/256k. Same 74.35% shared projection, constant RMS correction, frozen 62 prompts and original four-thread FP32 decoding.',
        'whole_model_healing':False,'cloud_spend_usd':0,
        'limits':'One fixed corpus and small repeated evaluation screen; transfer and general intelligence unproven.'}
    save(protocol,out/'protocol.json')
    for name in ('fullwidth-control.json','roundtrip.json','environment.json','tests.txt'):
        shutil.copy2(base/name,out/name)
    for name in ('reference.pt','validation-states.pt'):
        link=out/name
        if not link.exists():link.symlink_to((base/name).resolve())
    log(json.dumps({k:protocol[k] for k in ('actual_tokens','rows','by_kind','additional_rows')}))


def packet_weights(packet,rows):
    assert packet['rows']==len(rows)
    assert packet['tokens']==sum(len(r['input_ids']) for r in rows)
    expected=torch.tensor([min(32,len(r['input_ids'])) for r in rows])
    assert torch.equal(torch.bincount(packet['groups'],minlength=len(rows)),expected)
    assert packet['samples'].shape[1]==len(packet['groups'])
    per_row=torch.tensor([len(r['input_ids'])/min(32,len(r['input_ids'])) for r in rows])
    return per_row[packet['groups']]


def run(out=OUT,base=BASE):
    prepare(out,base)
    data=json.loads((out/'data.json').read_text());n=data['prefix_rows']
    original=data['calibration'][:n];extra=data['calibration'][n:]
    path=out/'additional-states.pt'
    if not path.exists():
        model=load_source();collect_packet(model,extra,path)
        del model;gc.collect()
    recipe_path=out/f'{ARM}-recipe.pt'
    if not recipe_path.exists():
        packets=[torch.load(base/'broad-states.pt',weights_only=True),torch.load(path,weights_only=True)]
        weights=torch.cat([packet_weights(p,r) for p,r in zip(packets,[original,extra],strict=True)])
        ns=packets[0]['samples'].shape[0]
        assert ns==packets[1]['samples'].shape[0]==49
        covs=[a.double()+b.double() for a,b in zip(packets[0]['covs'],packets[1]['covs'],strict=True)]
        q,values=eig(sum(c/c.trace() for c in covs))
        torch.save({'q':q,'values':values},out/'broad-extended-pca.pt')
        width=json.loads((out/'protocol.json').read_text())['width'];p=q[:,:width].contiguous()
        validation=torch.load(base/'validation-states.pt',weights_only=True)['samples']
        scales=[];errors=[]
        for site in range(ns):
            x=torch.cat([packet['samples'][site] for packet in packets])
            scale,predictor=fit_rms(x,p,weights=weights);scales.append(scale)
            errors.append(rms_error(validation[site],p,scale,predictor))
            if (site+1)%8==0:log(f'Fitted shared broad 256k norm {site+1}/{ns}')
        recipe={'basis':q.unsqueeze(0).expand(24,-1,-1),'width':width,'local':False,
                'scales':scales,'predictors':None,'source':'Qwen/Qwen3.5-2B','calibration':'broad-extended'}
        torch.save(recipe,recipe_path)
        old_q=torch.load(base/'shared-broad-recipe.pt',weights_only=True)['basis'][0,:,:width]
        energies=[(x@p).square().sum().item()/x.square().sum().item() for x in validation]
        save({'heldout_rms':errors,'heldout_retained_energy':energies,
              'mean_heldout_retained_energy':sum(energies)/len(energies),
              'broad_64k_256k_mean_squared_subspace_overlap':(old_q.T@p).square().sum().item()/width,
              'calibration_sample_states':len(weights),'weighted_tokens':weights.double().sum().item(),
              'scope':'Representation diagnostics only; answer correctness is the selection criterion.'},out/'fit-summary.json')
        del packets,covs,weights,validation,q,p,x;gc.collect()
    torch.set_num_threads(4)
    evaluate(out,ARM)
    current=json.loads((out/f'{ARM}-result.json').read_text())
    reference=json.loads((base/'shared-broad-result.json').read_text())
    assert current['parameters']==reference['parameters']
    assert current['stored_scalars']==reference['stored_scalars']
    log('Broader + more comparison complete, at exactly matched model storage')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['prepare','run'])
    ap.add_argument('--out',type=Path,default=OUT)
    args=ap.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(251025)
    disable_progress_bar()
    globals()[args.stage](args.out)
