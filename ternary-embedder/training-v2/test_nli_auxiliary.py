import copy,json,tempfile,unittest,torch
from pathlib import Path
from nli_auxiliary import features,NliAuxiliary
class Batch(dict):
 def to(self,device):return Batch({k:v.to(device) for k,v in self.items()})
class Tokens:
 def __call__(self,texts,return_tensors=None,**kwargs):
  ids=[[1,int(s)+3,2] for s in texts]
  return Batch(input_ids=torch.tensor(ids),attention_mask=torch.ones((len(ids),3),dtype=torch.long)) if return_tensors else {'input_ids':ids}
 def pad(self,encoded,**kwargs):
  ids=torch.tensor(encoded['input_ids']);return Batch(input_ids=ids,attention_mask=torch.ones_like(ids))
class AuxiliaryTests(unittest.TestCase):
 def test_encoder_gradient_weight_preserves_forward_and_head_gradient(self):
  torch.manual_seed(7);base=torch.randn(8,12);head=torch.nn.Linear(36,3,bias=False);labels=torch.tensor([0,1,2,1]);results=[]
  for weight in (0.,.25,1.):
   v=base.clone().requires_grad_();h=copy.deepcopy(head);logits=h(features(v,4,weight));torch.nn.functional.cross_entropy(logits,labels).backward();results.append((logits.detach(),v.grad,h.weight.grad))
  self.assertTrue(torch.equal(results[0][0],results[2][0]));self.assertTrue(torch.equal(results[1][0],results[2][0]));self.assertIsNone(results[0][1])
  self.assertTrue(torch.allclose(results[1][1],results[2][1]*.25,atol=1e-7));self.assertTrue(torch.equal(results[0][2],results[2][2]));self.assertTrue(torch.equal(results[1][2],results[2][2]))
 @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
 def test_cuda_bf16_head_warmup_joint_gradients_and_state(self):
  from grow_depth import Student
  torch.manual_seed(19)
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'rows.jsonl';path.write_text(''.join(json.dumps({'sentence1':str(i+1),'sentence2':str(i+11),'label':i%3})+'\n' for i in range(12)))
   for quant in (False,True):
    m=Student(vocab=100,emb=24,d=24,heads=3,ffn=48,layers=2,max_pos=16,out_dim=24).cuda();m.set_quant(quant);aux=NliAuxiliary(path,path,Tokens(),24,'cuda',batch=4);before={k:v.clone() for k,v in m.state_dict().items()};opt=torch.optim.AdamW([{'params':list(m.parameters())},{'params':list(aux.head.parameters())}],lr=1e-3)
    opt.zero_grad(set_to_none=True);aux.backward(m,0.,True);opt.step();self.assertTrue(all(torch.equal(before[k],v) for k,v in m.state_dict().items()))
    opt.zero_grad(set_to_none=True);aux.backward(m,.25,True);self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters()));opt.step()
    state=copy.deepcopy(aux.state_dict());recovered=NliAuxiliary(path,path,Tokens(),24,'cuda',batch=4);recovered.load_state_dict(state);self.assertEqual(aux.sampler.next(),recovered.sampler.next());self.assertTrue(torch.equal(aux.head.weight,recovered.head.weight));self.assertTrue(torch.isfinite(torch.tensor(aux.evaluate(m)['accuracy'])))
if __name__=='__main__':unittest.main()
