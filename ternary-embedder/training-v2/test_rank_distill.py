import tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
import torch
import numpy as np
from rank_distill import GradualLinear,Student,TernLinear,install,ranking_loss,schedule,set_alpha,save_checkpoint,split_queries,positive_loss

class DistillationTests(unittest.TestCase):
    def test_ranking_targets_need_not_match_student_dimensions(self):
        q=torch.randn(3,4,requires_grad=True);d=torch.randn(5,4,requires_grad=True)
        tq=torch.randn(3,11);td=torch.randn(5,11)
        loss=ranking_loss(q,d,tq,td);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(q.grad).all())
        identical=ranking_loss(q.detach(),d.detach(),q.detach(),d.detach())
        self.assertLess(abs(float(identical)),1e-5)
    def test_float_and_ternary_endpoints_are_exact(self):
        torch.manual_seed(7);reference=TernLinear(32,16);gradual=GradualLinear(32,16)
        gradual.load_state_dict(reference.state_dict());gradual.quant=True;x=torch.randn(4,32)
        for alpha in (0.,1.):
            reference.quant=bool(alpha);gradual.alpha=alpha
            self.assertTrue(torch.equal(reference(x),gradual(x)))
        gradual.alpha=.5;gradual(x).sum().backward()
        self.assertTrue(torch.isfinite(gradual.weight.grad).all())
    def test_final_checkpoint_preserves_standard_student_format(self):
        model=Student(vocab=100,emb=32,d=32,heads=4,ffn=128,max_pos=16,out_dim=32)
        install(model);set_alpha(model,1.);model.eval()
        ids=torch.randint(0,100,(2,8));mask=torch.ones_like(ids,dtype=torch.bool)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.pt';model.save(path);loaded=Student.load(path).eval()
            self.assertTrue(loaded.quantized)
            self.assertTrue(torch.equal(model(ids,mask)[2],loaded(ids,mask)[2]))
    def test_interrupted_save_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'best.pt';path.write_bytes(b'previous-valid-checkpoint')
            def fail(target,**metadata):
                Path(target).write_bytes(b'partial')
                raise OSError('simulated interrupted write')
            with self.assertRaises(OSError):save_checkpoint(Mock(save=fail),path)
            self.assertEqual(path.read_bytes(),b'previous-valid-checkpoint')
            self.assertFalse(path.with_name('best.pt.tmp').exists())
    def test_holdout_query_cannot_leak_as_training_passage(self):
        rows=[{'query':f'q{i}','positive':f'p{i}','negative':f'n{i}'} for i in range(40)]
        train,valid,held=split_queries(rows,np.random.default_rng(0))
        leaking=train[0];rows[leaking]['positive']=next(iter(held))
        train2,valid2,held2=split_queries(rows,np.random.default_rng(0))
        self.assertEqual(valid,valid2);self.assertEqual(held,held2)
        self.assertNotIn(leaking,train2)
    def test_verified_positives_allow_multiple_correct_passages(self):
        q=torch.randn(2,4,requires_grad=True);d=torch.randn(3,4,requires_grad=True)
        mask=torch.tensor([[True,False,False],[False,True,False]])
        expected=torch.nn.functional.cross_entropy(q@d.T/.05,torch.tensor([0,1]))
        self.assertTrue(torch.allclose(positive_loss(q,d,mask),expected))
        multi=torch.ones((2,3),dtype=torch.bool)
        self.assertEqual(float(positive_loss(q,d,multi).detach()),0.)
        positive_loss(q,d,mask).backward();self.assertTrue(torch.isfinite(q.grad).all())
    def test_schedule_has_float_ramp_and_full_qat_phases(self):
        self.assertEqual(schedule(4,16,.5,.25),0.)
        self.assertEqual(schedule(10,16,.5,.25),.5)
        self.assertEqual(schedule(16,16,.5,.25),1.)
if __name__=='__main__':unittest.main()
