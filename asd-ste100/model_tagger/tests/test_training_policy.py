"""Training-policy checks: deterministic stratified replay and train-only weighting."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from packed import Batches
from train import objective,batch_sampler
from tagger import loss


class TrainingPolicyTests(unittest.TestCase):
    def test_capped_replay_keeps_every_real_example_and_rotates_local_examples(self):
        index=np.array([[0,8+i%50,1,0,i%2,3 if i>=12 else i%3] for i in range(1012)])
        def selected(epoch,start=0):
            return list(Batches(index,16,42,epoch,start=start,local_cap=100))
        batches=selected(0);flat=[i for b in batches for i in b]
        self.assertEqual(len(flat),112)
        self.assertEqual(len(set(flat)),112)
        self.assertTrue(set(range(12)).issubset(flat))
        self.assertEqual(batches,selected(0))
        self.assertEqual(batches[2:],selected(0,2))
        self.assertNotEqual(set(flat),{i for b in selected(1) for i in b})

    def test_kind_repeats_batch_rows_of_that_kind_again(self):
        index=np.array([[0,8+i%50,1,0,i%2,5 if i<20 else 3] for i in range(220)])
        def flat(epoch,start=0,repeats=(1,1,1,1,1,3)):
            return [i for b in batch_sampler(index,16,7,epoch,start=start,local_cap=100,repeats=list(repeats)) for i in b]
        rows=flat(0);counts=np.bincount(rows,minlength=220)
        self.assertTrue((counts[:20]==3).all())
        self.assertEqual(len(rows),60+100)
        self.assertEqual(rows,flat(0))
        self.assertEqual(len(batch_sampler(index,16,7,0,start=2,local_cap=100,repeats=[1,1,1,1,1,3])),
                         len(batch_sampler(index,16,7,0,local_cap=100,repeats=[1,1,1,1,1,3]))-2)
        self.assertEqual(flat(0,repeats=(1,)*6),[i for b in Batches(index,16,7,0,local_cap=100) for i in b])

    def test_weighting_changes_train_gradient_but_not_dev_objective(self):
        class Fixed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scores=torch.nn.Parameter(torch.tensor([[[0.,2.]],[[2.,0.]]]))
            def forward(self,batch):
                return self.scores, self.scores.new_zeros((2,1,3))
        model=Fixed()
        batch=dict(labels=torch.tensor([[0],[0]]),label_cases=torch.zeros((2,1),dtype=torch.long),
                   actions=torch.ones((2,1),dtype=torch.bool),terminal=torch.tensor([False,True]),
                   kind=torch.tensor([0,3]))
        tags=SimpleNamespace(KEEP=0,transform_offset=2)
        args=SimpleNamespace(precision='fp32',terminal_weight=1.,keep_weight=1.,kind_weights=[8.,12.,3.,1.])
        value,_=objective(model,batch,tags,args)
        value.backward();weighted=model.scores.grad.clone()
        model.zero_grad();model.eval()
        dev,_=objective(model,batch,tags,args)
        expected=loss(model.scores,model.scores.new_zeros((2,1,3)),batch['labels'],batch['label_cases'],batch['actions'],tags)
        self.assertTrue(torch.equal(dev,expected))
        self.assertGreater(value.item(),dev.item())
        dev.backward()
        self.assertGreater(weighted[0].abs().sum(),model.scores.grad[0].abs().sum())
        self.assertLess(weighted[1].abs().sum(),model.scores.grad[1].abs().sum())


if __name__=='__main__':unittest.main()
