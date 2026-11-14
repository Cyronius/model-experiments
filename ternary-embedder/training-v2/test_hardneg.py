import unittest,torch
from pair_train import batch_documents
from pair_validation import PairValidation
from restore_pair_negatives import near_duplicate

class Tokens:
 def __call__(self,texts,**kwargs):return {'input_ids':[[int(s)] for s in texts]}
 def pad(self,encoded,**kwargs):
  ids=torch.tensor(encoded['input_ids']);return Batch(input_ids=ids,attention_mask=torch.ones_like(ids))
class Batch(dict):
 def to(self,device):return Batch({k:v.to(device) for k,v in self.items()})
class Vectors(torch.nn.Module):
 dims={'out_dim':2}
 def forward(self,ids,mask):
  values=torch.tensor([[1.,0.],[.8,.6],[1.,0.]],device=ids.device);return None,None,values[ids[:,0]]

class NegativeTests(unittest.TestCase):
 def test_positive_precedes_duplicate_negative(self):
  rows=[{'positive_visible_key':'a','positive':'support','negative_visible_key':'b','negative':'other'}, {'positive_visible_key':'b','positive':'other support','negative_visible_key':'a','negative':'support alias'}]
  self.assertEqual(batch_documents(rows,True),{'a':'support','b':'other support'})
 def test_validation_negatives_actually_compete(self):
  rows=[{'query':'0','positive':'1','negative':'2','query_visible_key':'q','positive_visible_key':'p','negative_visible_key':'n','source':'test'}]
  m=Vectors();simple=PairValidation(rows,Tokens())(m,'cpu');hard=PairValidation(rows,Tokens(),True)(m,'cpu')
  self.assertAlmostEqual(simple['macro_ndcg10'],100.);self.assertAlmostEqual(hard['macro_ndcg10'],63.092975,places=4);self.assertTrue(m.training)
  self.assertEqual(hard['by_source']['test']['documents'],2)
 def test_duplicate_filter(self):
  self.assertTrue(near_duplicate('a b c d e f','f e d c b a'));self.assertFalse(near_duplicate('cats chase mice','atoms contain protons'))
if __name__=='__main__':unittest.main()
