"""Keep selected checkpoints in the artifact bundle; inventory redundant snapshots."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path('/workspace/ternary-pilot/ternary-embedder/training-v2')
if (ROOT/'pilot.status').read_text().strip()!='done':raise RuntimeError('Training must finish before preparing recovery')
keep={f'{name}/best.pt' for name in ['adapted-0.6B-gradual','adapted-8B-gradual','adapted-8B-abrupt','continued-8B-qat']}
keep.add('adapted-8B-gradual/float.pt')
for name in keep:
 if not (ROOT/'runs'/name).is_file():raise RuntimeError('Missing retained checkpoint: '+name)
extra=ROOT.parent.parent/'checkpoint-extras';inventory=[]
for path in list((ROOT/'runs').rglob('*.pt')):
 name=str(path.relative_to(ROOT/'runs'));retain=name in keep
 inventory.append({'path':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'retained':retain})
 if not retain:
  dest=extra/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(path),dest)
(ROOT/'runs/checkpoint-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
print('Retained',len(keep),'checkpoints; inventoried',len(inventory),'total. Redundant snapshots excluded from transfer.')
