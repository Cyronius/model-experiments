import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import test_tagger as fixture
from tagger import Tagger,Config,collate,Tags,loss
from action_diffusion import ActionDiffusion,corrupt,reveal_count,rewrite


class DiffusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.TaggerTests.setUpClass();cls.tok=fixture.TaggerTests.tok;cls.lex=fixture.TaggerTests.lex
    @classmethod
    def tearDownClass(cls):fixture.TaggerTests.tearDownClass()
    def setup_model(self):
        config=Config(width=32,layers=1,heads=4,ffn=64,dropout=0)
        base=Tagger(self.tok,config);model=ActionDiffusion(self.tok,config)
        missing,unexpected=model.load_state_dict(base.state_dict(),strict=False)
        self.assertEqual(set(missing),{'draft_action.weight','noise_level.weight','noise_level.bias'});self.assertFalse(unexpected)
        row=dict(self.tok.encode_source('Open the file.','PROC'),src='Open the file.',mode='PROC')
        return base,model,collate([row],self.tok,self.lex)
    def test_zero_conditioning_preserves_warm_start_and_new_inputs_receive_gradients(self):
        base,model,batch=self.setup_model();base.eval();model.eval()
        draft=torch.full(batch['actions'].shape,model.mask_action)
        scores,cases=model(batch,draft,torch.ones(1));old,_=base(batch)
        self.assertTrue(torch.equal(scores,old))
        labels=scores.argmax(-1);value=loss(scores,cases,labels,torch.zeros_like(labels),batch['actions'],Tags(self.tok))
        value.backward()
        self.assertGreater(model.draft_action.weight.grad.abs().sum(),0)
        self.assertGreater(model.noise_level.weight.grad.abs().sum(),0)
    def test_forward_noise_is_repeatable_preserves_visible_labels_and_never_masks_padding(self):
        labels=torch.tensor([[0,1,2,-100],[2,1,0,3]])
        valid=labels.ne(-100)
        a=corrupt(labels,valid,99,torch.Generator().manual_seed(4))
        b=corrupt(labels,valid,99,torch.Generator().manual_seed(4))
        for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
        noisy,noise,masked=a
        self.assertFalse((masked&~valid).any());self.assertTrue(masked.any(-1).all())
        self.assertTrue(torch.equal(noisy[~masked&valid],labels[~masked&valid]))
        self.assertTrue(noisy[masked].eq(99).all())
    def test_reveal_schedule_finishes_in_five_steps(self):
        for count in range(1,258):
            remaining=count
            for left in range(5,0,-1):remaining-=reveal_count(remaining,left)
            self.assertEqual(remaining,0)
    def test_sampler_never_returns_corrupted_literals_and_is_bounded(self):
        _,model,_=self.setup_model()
        result=rewrite(model,self.tok,self.lex,'Use `x = 10`.',max_passes=5)
        self.assertIn('`x = 10`',result['text']);self.assertEqual(result['passes'],5)
        self.assertEqual(result['trace'][-1]['masked_remaining'],0)
        for cap in (0,6):
            with self.assertRaises(ValueError):rewrite(model,self.tok,self.lex,'Open file.',max_passes=cap)

if __name__=='__main__':unittest.main()
