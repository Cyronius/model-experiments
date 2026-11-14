import unittest
import torch
from pair_train import batch_positive_mask


class PositiveMaskTests(unittest.TestCase):
 def check_device(self,device):
  batch=[{'query_visible_key':q} for q in ('a','b','a')]
  known={'a':{'d1','d3','outside'},'b':{'d2','d4'}}
  # d4 can enter as a negative for a different query, but is positive for b.
  result=batch_positive_mask(batch,['d1','d2','d3','d4'],known,device)
  expected=torch.tensor([[1,0,1,0],[0,1,0,1],[1,0,1,0]],dtype=torch.bool)
  self.assertTrue(torch.equal(result.cpu(),expected))
  self.assertEqual(result.device.type,torch.device(device).type)
  self.assertEqual(known['a'],{'d1','d3','outside'})
 def test_multiple_positives_duplicate_queries_and_negative_aliases(self):self.check_device('cpu')
 @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
 def test_cuda_mask_matches_labels(self):self.check_device('cuda')
 def test_no_current_positive_stays_empty_for_loss_validation(self):
  m=batch_positive_mask([{'query_visible_key':'a'}],['b'],{'a':{'c'}},'cpu')
  self.assertEqual(m.tolist(),[[False]])


if __name__=='__main__':unittest.main()
