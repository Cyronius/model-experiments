"""Resumable source-batch epochs and atomic, independently verifiable checkpoints."""
import copy,hashlib,json,os,time
from pathlib import Path
import numpy as np,torch

class SourceSampler:
 def __init__(self,by_source,batch,seed):
  self.by_source=by_source;self.sources=sorted(by_source);self.batch=batch;self.rng=np.random.default_rng(seed);self.epoch=0;self.position=0
  self.batches_per_epoch=sum(max(1,(len(v)+batch-1)//batch-(len(v)%batch==1)) for v in by_source.values());self._start_epoch()
 def _start_epoch(self):
  self.permutations={s:self.rng.permutation(self.by_source[s]).astype(np.int32) for s in self.sources};self.offsets={s:0 for s in self.sources}
  schedule=[]
  for i,s in enumerate(self.sources):schedule.extend([i]*max(1,(len(self.by_source[s])+self.batch-1)//self.batch-(len(self.by_source[s])%self.batch==1)))
  self.schedule=self.rng.permutation(np.array(schedule,dtype=np.int16));self.position=0
 def next(self):
  if self.position==len(self.schedule):self.epoch+=1;self._start_epoch()
  source=self.sources[int(self.schedule[self.position])];self.position+=1;start=self.offsets[source];n=len(self.permutations[source]);end=n if n-start==self.batch+1 else min(n,start+self.batch);self.offsets[source]=end
  return source,self.permutations[source][start:end].tolist()
 def state_dict(self):return copy.deepcopy({'epoch':self.epoch,'position':self.position,'permutations':self.permutations,'offsets':self.offsets,'schedule':self.schedule,'rng':self.rng.bit_generator.state})
 def load_state_dict(self,state):
  for k in ('epoch','position','permutations','offsets','schedule'):setattr(self,k,copy.deepcopy(state[k]))
  self.rng.bit_generator.state=copy.deepcopy(state['rng'])

def atomic_json(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(path)
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def atomic_torch_save(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('wb') as f:torch.save(data,f);f.flush();os.fsync(f.fileno())
 tmp.replace(path)
def checkpoint(model,opt,sampler,step,config,progress,out,extra_resume=None):
 out=Path(out);path=out/f'checkpoint-{step:07d}.pt'
 training_state=model.state_dict();inference=training_state
 if config.get('learn_row_scales'):
  from learned_row_scales import inference_state
  inference=inference_state(model)
 payload={'dims':model.dims,'state':inference,'quant':model.quantized,'meta':{'step':step,'config':config,'validation':progress.get('last_validation'),'growth_alpha':next((layer.alpha for layer in model.layers if hasattr(layer,'alpha')),1.)},
  'resume':{'optimizer':opt.state_dict(),'sampler':sampler.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,'progress':progress,'format':1}}
 if config.get('learn_row_scales'):payload['resume']['training_state']=training_state
 if extra_resume:
  if set(extra_resume)&set(payload['resume']):raise ValueError('Extra resume state collides with core state')
  payload['resume'].update(extra_resume)
 atomic_torch_save(path,payload);info={'file':path.name,'step':step,'sha256':digest(path),'bytes':path.stat().st_size,'saved_unix':time.time()};atomic_json(path.with_suffix('.json'),info);atomic_json(out/'latest.json',info)
 # Never remove the latest or its predecessor. Off-pod pulls run independently.
 previous=sorted(out.glob('checkpoint-*.pt'))[:-2]
 for old in previous:old.unlink(missing_ok=True);old.with_suffix('.json').unlink(missing_ok=True)
 return info
