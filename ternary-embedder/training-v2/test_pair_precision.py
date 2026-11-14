import unittest,torch
from export_pair_precision import pack_weight,unpack_weight
class PrecisionTests(unittest.TestCase):
 def test_signed_nibbles_and_scales_match_reference(self):
  torch.manual_seed(12);w=torch.randn(12,128)
  for bits in (4,8):
   value=pack_weight(w,bits);limit=2**(bits-1)-1;blocks=w.reshape(12,-1,64);s=(blocks.abs().amax(-1,keepdim=True)/limit).clamp_min(1e-8).to(torch.bfloat16).float();expected=((blocks/s).round().clamp(-limit,limit)*s).reshape_as(w)
   self.assertTrue(torch.equal(unpack_weight(value),expected));self.assertEqual(value['packed'].numel(),w.numel()*bits//8)
if __name__=='__main__':unittest.main()
