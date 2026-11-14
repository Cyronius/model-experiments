"""Exercise actual-width ternary cached training through a saved interruption."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
import torch
import numpy as np
ROOT=Path(__file__).resolve().parent

def assert_tree(test,a,b):
 if isinstance(a,torch.Tensor):test.assertTrue(torch.equal(a,b))
 elif isinstance(a,np.ndarray):test.assertTrue(np.array_equal(a,b))
 elif isinstance(a,dict):
  test.assertEqual(a.keys(),b.keys())
  for k in a:assert_tree(test,a[k],b[k])
 elif isinstance(a,(list,tuple)):
  test.assertEqual(len(a),len(b))
  for x,y in zip(a,b):assert_tree(test,x,y)
 else:test.assertEqual(a,b)

class ResumeTests(unittest.TestCase):
 def test_replay_resume_is_bit_exact(self):self.run_case(False)
 def test_learned_scales_replay_resume_is_bit_exact(self):self.run_case(True)
 def test_auxiliary_head_and_sampler_resume_is_bit_exact(self):self.run_case(False,True)
 def test_learned_scales_with_auxiliary_resume_is_bit_exact(self):self.run_case(True,True)
 def test_scale_only_with_auxiliary_resume_is_bit_exact(self):self.run_case(True,True,scale_only=True)
 def test_explicit_negative_training_and_validation_resume(self):self.run_case(False,False,True)
 def run_case(self,learn_scales,auxiliary=False,hard_negatives=False,scale_only=False):
  torch.set_num_threads(2)
  with tempfile.TemporaryDirectory() as directory:
   base=Path(directory);data=base/'data.jsonl'
   source=ROOT/('work/hardneg/data/train.jsonl' if hard_negatives else 'work/data-first/capfit/train.jsonl')
   with source.open() as f:data.write_text(''.join(next(f) for _ in range(32)))
   common=[sys.executable,str(ROOT/'pair_train.py'),'--data',str(data),'--checkpoint',str(ROOT/'work/depth3/starts/2-ternary.pt'),'--steps','6','--batch','8','--micro-batch','4','--save-every','3','--save-seconds','120','--expected-layers','2']
   if learn_scales:common+=['--learn-row-scales']
   if scale_only:common+=['--train-row-scales-only']
   if hard_negatives:common+=['--use-negatives','--validation-negatives','--dev',str(data),'--eval-every','3']
   if auxiliary:
    examples=[json.loads(l) for l in data.read_text().splitlines()];aux=base/'aux.jsonl';aux.write_text(''.join(json.dumps({'sentence1':r['query'],'sentence2':r['positive'],'label':i%3})+'\n' for i,r in enumerate(examples)))
    common+=['--aux-data',str(aux),'--aux-dev',str(aux),'--aux-batch','4','--aux-encoder-weight','.25','--aux-warmup-steps','2','--dev',str(data),'--eval-every','3']
   def run(out,*extra):
    p=subprocess.run(common+['--out',str(base/out)]+list(extra),capture_output=True,text=True)
    self.assertEqual(p.returncode,0,p.stdout[-1500:]+p.stderr[-1500:])
   run('full');run('resume','--stop-after','3');pointer=json.loads((base/'resume/latest.json').read_text())
   if auxiliary:
    warmed=torch.load(base/'resume/checkpoint-0000000.pt',map_location='cpu',weights_only=False);parent=torch.load(ROOT/'work/depth3/starts/2-ternary.pt',map_location='cpu',weights_only=False)
    master={k:v for k,v in warmed['resume']['training_state'].items() if not k.endswith('.log_scale')} if learn_scales else warmed['state']
    assert_tree(self,master,parent['state'])
   run('resume','--resume',str(base/'resume'/pointer['file']))
   a=torch.load(base/'full/checkpoint-0000006.pt',weights_only=False);b=torch.load(base/'resume/checkpoint-0000006.pt',weights_only=False)
   assert_tree(self,a['state'],b['state']);assert_tree(self,a['resume'],b['resume'])
   if scale_only:
    master={k:v for k,v in a['resume']['training_state'].items() if not k.endswith('.log_scale')}
    assert_tree(self,master,parent['state'])
if __name__=='__main__':unittest.main()
