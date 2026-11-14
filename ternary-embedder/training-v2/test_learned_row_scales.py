import copy,unittest,torch
from grow_depth import Student
from model import tern_codes
from learned_row_scales import install,inference_state,LearnedScaleLinear,scales_only,optimizer

class ScaleTests(unittest.TestCase):
 @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
 def test_actual384_three_layer_bf16_training_and_canonical_export(self):
  from contrastive_cache import cached_backward
  torch.manual_seed(19)
  for only in (False,True):
   with self.subTest(scales_only=only):
    m=Student(vocab=30522,emb=384,d=384,heads=12,ffn=1536,layers=3,max_pos=128,out_dim=384).cuda();m.set_quant(True);install(m)
    if only:scales_only(m)
    opt=optimizer(m,1e-5,100.)
    for group in opt.param_groups:group['lr']*=group.get('lr_multiplier',1.)
    ids=torch.randint(1,30522,(8,16),device='cuda');enc={'input_ids':ids,'attention_mask':torch.ones_like(ids)};positive=torch.eye(4,dtype=torch.bool,device='cuda')
    for _ in range(2):
     opt.zero_grad(set_to_none=True);cached_backward(m,enc,4,positive,.05,4,amp=True)
     gradients=[p.grad for n,p in m.named_parameters() if n.endswith('.log_scale')]
     self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients));opt.step()
    ordinary=Student(**m.dims).cuda();ordinary.set_quant(True);ordinary.load_state_dict(inference_state(m))
    with torch.no_grad():
     expected=m(ids,enc['attention_mask'].bool())[2];actual=ordinary(ids,enc['attention_mask'].bool())[2]
    self.assertTrue(torch.allclose(expected,actual,atol=2e-6,rtol=2e-6))
 def test_initial_function_and_export_after_updates(self):
  torch.set_num_threads(2);torch.manual_seed(5)
  old=Student(vocab=100,emb=24,d=24,heads=3,ffn=48,layers=2,max_pos=16,out_dim=24);old.set_quant(True)
  rng=torch.get_rng_state().clone();m=install(copy.deepcopy(old));self.assertTrue(torch.equal(rng,torch.get_rng_state()));ids=torch.randint(1,100,(6,11));mask=torch.ones_like(ids,dtype=torch.bool)
  self.assertTrue(torch.equal(old(ids,mask)[2],m(ids,mask)[2]))
  opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
  for _ in range(4):
   opt.zero_grad();v=m(ids,mask)[2];loss=(v[0]-v[1]).square().sum();loss.backward()
   scales=[layer for layer in m.modules() if isinstance(layer,LearnedScaleLinear)]
   self.assertTrue(all(torch.isfinite(x.log_scale.grad).all() for x in scales));self.assertGreater(sum(float(x.log_scale.grad.abs().sum()) for x in scales),0.)
   opt.step()
  ordinary=Student(**m.dims);ordinary.set_quant(True);ordinary.load_state_dict(inference_state(m))
  self.assertTrue(torch.allclose(m(ids,mask)[2],ordinary(ids,mask)[2],atol=2e-6,rtol=2e-6))
  self.assertFalse(any('log_scale' in k for k in ordinary.state_dict()))
 def test_zero_row_export(self):
  m=Student(vocab=100,emb=24,d=24,heads=3,ffn=48,layers=2,max_pos=16,out_dim=24);m.set_quant(True);install(m)
  with torch.no_grad():m.layers[0].q.weight[0].zero_()
  out=inference_state(m);codes,_=tern_codes(out['layers.0.q.weight']);self.assertTrue((codes[0]==0).all())
 def test_float_install_rejected(self):
  with self.assertRaises(ValueError):install(Student(vocab=100,emb=24,d=24,heads=3,ffn=48,layers=2,max_pos=16,out_dim=24))
 def test_scale_only_updates_do_not_change_master_weights(self):
  torch.set_num_threads(2);torch.manual_seed(17);m=Student(vocab=100,emb=24,d=24,heads=3,ffn=48,layers=2,max_pos=16,out_dim=24);m.set_quant(True);install(m);self.assertGreater(scales_only(m),0)
  before={k:v.clone() for k,v in m.state_dict().items()};ids=torch.randint(1,100,(4,9));mask=torch.ones_like(ids,dtype=torch.bool);opt=optimizer(m,1e-5,100.)
  for group in opt.param_groups:group['lr']*=group.get('lr_multiplier',1.)
  v=m(ids,mask)[2];(v[0]-v[1]).square().sum().backward();opt.step()
  self.assertTrue(all(torch.equal(v,before[k]) for k,v in m.state_dict().items() if not k.endswith('.log_scale')))
  self.assertTrue(any(not torch.equal(v,before[k]) for k,v in m.state_dict().items() if k.endswith('.log_scale')))
if __name__=='__main__':unittest.main()
