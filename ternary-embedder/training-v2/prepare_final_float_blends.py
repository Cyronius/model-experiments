"""Final free two-point retention check between a float parent and descendant."""
import datetime,json
from pathlib import Path
import torch
from pair_training_state import atomic_json,atomic_torch_save,digest
ROOT=Path(__file__).resolve().parent
def main():
 torch.set_num_threads(2);output=ROOT/'final-float-blends/results/frozen.json'
 if output.exists():raise RuntimeError('Already frozen')
 parent=json.loads((ROOT/'float-blends/results/frozen.json').read_text())['models']['float-25']
 child=json.loads((ROOT/'human-mix/results/frozen.json').read_text())['models']['float']
 assert child['initial_checkpoint_sha256']==parent['sha256']
 for source in (parent,child):assert digest(Path(source['path']))==source['sha256']
 a,b=[torch.load(source['path'],map_location='cpu',weights_only=False) for source in (parent,child)]
 assert not a['quant'] and not b['quant'] and a['dims']==b['dims']
 frozen=dict(local_only=True,selected_before_external_scores=True,frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models={})
 for fraction in (.25,.5):
  name=f'float-{int(fraction*100)}';path=ROOT/'work/final-float-blends'/(name+'.pt');path.parent.mkdir(parents=True,exist_ok=True)
  state={k:v.lerp(b['state'][k],fraction) for k,v in a['state'].items()}
  atomic_torch_save(path,dict(dims=a['dims'],state=state,quant=False,meta=dict(parent_sha256=parent['sha256'],child_sha256=child['sha256'],child_fraction=fraction)))
  frozen['models'][name]=dict(path=str(path.resolve()),sha256=digest(path),precision='float',layers=3,eligible_for_promotion=False,child_fraction=fraction,note='Development blend; no new training or paid resource.')
 output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,frozen);print('Both final float blends frozen before scoring.',flush=True)
if __name__=='__main__':main()
