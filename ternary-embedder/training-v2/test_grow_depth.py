import copy,tempfile,unittest
from pathlib import Path
import torch
from grow_depth import Student,install_growth,set_growth,grow
class GrowthTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2);torch.manual_seed(19)
 def model(self,quant):
  m=Student(vocab=80,emb=24,d=24,heads=3,ffn=48,layers=2,max_pos=16,out_dim=24);m.set_quant(quant);return m.eval()
 def test_identity_start_and_ordinary_export_for_both_precisions(self):
  for quant in (False,True):
   with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);old=self.model(quant);old.save(root/'old.pt');grow(root/'old.pt',root/'init.pt',5);new=Student.load(root/'init.pt');install_growth(new);new.eval();ids=torch.randint(1,80,(3,11));mask=torch.ones_like(ids,dtype=torch.bool)
    with torch.inference_mode():
     set_growth(new,0);self.assertTrue(torch.equal(old(ids,mask)[2],new(ids,mask)[2]));set_growth(new,1);new.save(root/'new.pt');loaded=Student.load(root/'new.pt').eval();self.assertTrue(torch.equal(new(ids,mask)[2],loaded(ids,mask)[2]))
    self.assertNotEqual(new.layers[1].q.weight.data_ptr(),new.layers[2].q.weight.data_ptr())
 def test_new_layer_has_finite_nonzero_gradients_during_ramp(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);old=self.model(True);old.save(root/'old.pt');grow(root/'old.pt',root/'init.pt',5);m=Student.load(root/'init.pt');install_growth(m);set_growth(m,.4)
   ids=torch.randint(1,80,(3,9));v=m(ids,torch.ones_like(ids,dtype=torch.bool))[2];(v[:,0].sum()).backward();grads=[p.grad for p in m.layers[2].parameters()];self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads));self.assertGreater(sum(float(g.abs().sum()) for g in grads),0.)
if __name__=='__main__':unittest.main()
