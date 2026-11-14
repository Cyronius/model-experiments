import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data_first_fetch import clean,key,valid_pair
from data_first_build import Groups,visible_key
from pair_train import positive_pair_loss

class DataTests(unittest.TestCase):
 def test_canonical_duplicate(self):
  self.assertEqual(key('A&nbsp; <b>test</b>'),key('a test'))
  self.assertEqual(valid_pair('A useful test question','a useful TEST question','search relevance'),'identical_text')
 def test_transitive_split_has_no_role_leakage(self):
  g=Groups();g.union(b'query1',b'document');g.union(b'query2',b'document');g.union(b'query2',b'answer2')
  self.assertEqual(len({g.split(x,23,.5) for x in (b'query1',b'query2',b'document',b'answer2')}),1)
 def test_split_is_insertion_order_independent(self):
  a,b=Groups(),Groups();edges=[(b'a',b'b'),(b'c',b'd'),(b'b',b'd')]
  for x,y in edges:a.union(x,y)
  for x,y in reversed(edges):b.union(y,x)
  for seed in range(20):self.assertEqual(a.split(b'a',seed,.4),b.split(b'd',seed,.4))
 def test_visible_identity_is_token_identity(self):
  self.assertEqual(visible_key([101,23,102]),visible_key([101,23,102]))
  self.assertNotEqual(visible_key([101,23,102]),visible_key([101,24,102]))
 def test_multiple_positives_not_penalized_as_negatives(self):
  q=torch.tensor([[1.,0.]]);p=torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
  both=positive_pair_loss(q,p,torch.tensor([[True,True,False]]),1.)
  single=positive_pair_loss(q,p,torch.tensor([[True,False,False]]),1.)
  self.assertLess(float(both),float(single))
 def test_pair_loss_gradient_finite_and_improves_match(self):
  q=torch.tensor([[.1,.9],[.9,.1]],requires_grad=True);p=torch.eye(2);mask=torch.eye(2,dtype=torch.bool)
  before=positive_pair_loss(q,p,mask,.5);before.backward();self.assertTrue(torch.isfinite(q.grad).all())
  self.assertLess(float(positive_pair_loss(q.detach()-.1*q.grad,p,mask,.5)),float(before.detach()))
if __name__=='__main__':unittest.main()
