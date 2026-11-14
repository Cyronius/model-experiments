import copy,unittest,torch
from grow_depth import Student
from pair_train import positive_pair_loss
from contrastive_cache import cached_backward
class CacheTests(unittest.TestCase):
 def test_fp32_gradients_match_one_full_batch_graph(self):
  torch.set_num_threads(2);torch.manual_seed(13)
  for quant in (False,True):
   a=Student(vocab=100,emb=24,d=24,heads=3,ffn=48,layers=2,max_pos=16,out_dim=24);a.set_quant(quant);b=copy.deepcopy(a)
   ids=torch.randint(1,100,(12,9));enc={'input_ids':ids,'attention_mask':torch.ones_like(ids)};mask=torch.eye(6,dtype=torch.bool)
   v=a(ids,enc['attention_mask'].bool())[2].float();loss=positive_pair_loss(v[:6],v[6:],mask,.05);loss.backward()
   actual=cached_backward(b,enc,6,mask,.05,3)
   self.assertTrue(torch.allclose(loss.detach(),actual,atol=1e-6,rtol=1e-6))
   for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.allclose(x.grad,y.grad,atol=2e-5,rtol=2e-5))
 @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
 def test_cuda_bf16_replay_gradients(self):
  torch.manual_seed(13)
  for quant in (False,True):
   a=Student(vocab=100,emb=24,d=24,heads=3,ffn=48,layers=2,max_pos=16,out_dim=24).cuda();a.set_quant(quant);b=copy.deepcopy(a)
   ids=torch.randint(1,100,(12,9),device='cuda');enc={'input_ids':ids,'attention_mask':torch.ones_like(ids)};mask=torch.eye(6,dtype=torch.bool,device='cuda');mask[0,1]=True
   pieces=[]
   # Match replay's autocast boundaries. Sharing one autocast context caches a
   # shared BF16 parameter cast and accumulates its gradients in BF16 instead.
   for i in range(0,12,3):
    with torch.autocast('cuda',dtype=torch.bfloat16):pieces.append(a(ids[i:i+3],enc['attention_mask'][i:i+3].bool())[2].float())
   v=torch.cat(pieces);loss=positive_pair_loss(v[:6],v[6:],mask,.05);loss.backward();actual=cached_backward(b,enc,6,mask,.05,3,True)
   self.assertTrue(torch.allclose(loss.detach(),actual,atol=1e-6,rtol=1e-6))
   for (name,x),y in zip(a.named_parameters(),b.parameters()):self.assertTrue(torch.allclose(x.grad,y.grad,atol=2e-4,rtol=2e-4),f'{quant=} {name}: max error {float((x.grad-y.grad).abs().max())}')
if __name__=='__main__':unittest.main()
