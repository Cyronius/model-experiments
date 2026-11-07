"""Rollout/packed-batch invariants using a synthetic vocabulary only."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import test_tagger as fixtures
from tagger import Tags,source_units,collate,Tagger,Config,rewrite
import rollouts
from packed import pack,unpack,Collator,Batches


class RolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.TaggerTests.setUpClass()
        cls.tok,cls.lex,cls.tags=fixtures.TaggerTests.tok,fixtures.TaggerTests.lex,fixtures.TaggerTests.tags
        rollouts.TOK,rollouts.LEX,rollouts.TAGS=cls.tok,cls.lex,cls.tags

    @classmethod
    def tearDownClass(cls):fixtures.TaggerTests.tearDownClass()

    def pair(self,src,ste):
        return dict(self.tok.encode_pair({'src':src,'ste':ste,'mode':'PROC'},0),
                    src=src,ste=ste,mode='PROC',kind='rule',id='fixture',origin_id='origin',group='group',split='train',noise='drop_article',meaning_review='unjudged_rule')

    def test_roundtrip_and_vectorized_batch_match_reference(self):
        rows=[self.pair('Utilize fooBar.','Use fooBar.'),self.pair('Open file.','Open the file.')]
        states=[rollouts.trajectory(r)[0] for r in rows]
        packed=[]
        for row,labels,cases,terminal in states:
            array,header=pack(row,labels,cases,terminal,self.tags,self.lex)
            packed.append(unpack(array,(0,*header)))
        fast=Collator(self.tok)(packed);slow=collate([s[0] for s in states],self.tok,self.lex)
        for key in slow:self.assertTrue(torch.equal(fast[key],slow[key]),key)
        self.assertTrue(fast['refs'].gt(0).any())

    def test_every_selected_action_matches_reference_mask(self):
        row=self.pair('Utilize fooBar.','Use fooBar.')
        units=source_units(row);mask=self.tags.mask(units,row['slots'])
        for i in range(len(units)+1):
            actual=torch.tensor([self.tags.allowed(units,i,t,row['slots']) for t in range(self.tags.size)])
            self.assertTrue(torch.equal(actual,mask[i]))

    def test_early_checker_exit_is_rejected_not_partial_training(self):
        row=self.pair('File.','Use the file.')
        with self.assertRaisesRegex(Exception,'checker_exit_before_target'):
            rollouts.trajectory(row)

    def test_terminal_state_and_slot_fidelity(self):
        row=self.pair('Utilize fooBar.','Use fooBar.')
        states=rollouts.trajectory(row)
        self.assertEqual(len(states),2)
        self.assertTrue(states[-1][-1])
        self.assertTrue(all(x==Tags.KEEP for x in states[-1][1]))

    def test_checker_clean_input_still_runs_one_model_pass(self):
        tok,tags=self.tok,self.tags
        class Append(torch.nn.Module):
            def __init__(self):super().__init__();self.placeholder=torch.nn.Parameter(torch.zeros(1))
            def forward(self,batch):
                scores=torch.full(batch['legal'].shape,-100.)
                scores[:,:,0]=0;scores[:,1,tags.append(tok.output_ids['the'])]=100
                return scores.masked_fill(~batch['legal'],float('-inf')),torch.zeros((*scores.shape[:2],3))
        result=rewrite(Append(),tok,self.lex,'Open file.')
        self.assertEqual(result['text'],'Open the file.')
        self.assertEqual(result['passes'],1)

    def test_length_buckets_and_resume_are_deterministic(self):
        index=np.array([[0,n,1,0,0,0] for n in (8,9,16,17,40,60,100)])
        a=list(Batches(index,2,42,0));b=list(Batches(index,2,42,0,start=2))
        self.assertEqual(b,a[2:])
        self.assertEqual(sorted(i for batch in a for i in batch),list(range(7)))


if __name__=='__main__':unittest.main()
