import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pair_training_state import SourceSampler,atomic_torch_save,atomic_json,digest

class RecoveryTests(unittest.TestCase):
 def test_epoch_covers_every_row_with_tail_one(self):
  sampler=SourceSampler({'a':list(range(9)),'b':list(range(9,16))},4,5);seen=[]
  for _ in range(sampler.batches_per_epoch):seen.extend(sampler.next()[1])
  self.assertEqual(sorted(seen),list(range(16)));self.assertEqual(len(seen),len(set(seen)))
 def test_resume_repeats_next_batches_across_epoch_boundary(self):
  a=SourceSampler({'a':list(range(13)),'b':list(range(13,30))},4,5)
  for _ in range(5):a.next()
  state=a.state_dict();expected=[a.next() for _ in range(20)]
  b=SourceSampler({'a':list(range(13)),'b':list(range(13,30))},4,999);b.load_state_dict(state)
  self.assertEqual(expected,[b.next() for _ in range(20)])
 def test_published_checkpoint_is_whole_and_verifiable(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'model.pt';atomic_torch_save(path,{'state':torch.arange(50)})
   atomic_json(path.with_suffix('.json'),{'sha256':digest(path)})
   self.assertEqual(digest(path),json.loads(path.with_suffix('.json').read_text())['sha256'])
   self.assertTrue(torch.equal(torch.load(path,weights_only=False)['state'],torch.arange(50)))
   self.assertFalse(path.with_suffix('.pt.tmp').exists())
if __name__=='__main__':unittest.main()
