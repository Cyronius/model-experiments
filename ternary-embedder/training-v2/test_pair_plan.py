import copy,unittest
from run_pair_plan import command
class PlanTests(unittest.TestCase):
 def test_continuation_namespace_preserves_the_training_command(self):
  plan={'profile':'scales','data_inputs':{'train':{'path':'training-v2/train.jsonl'},'dev':{'path':'training-v2/dev.jsonl'}},'training':{'teacher_targets':False,'epochs_each':1,'seed':43}}
  arm={'id':'learned','checkpoint':'training-v2/model.pt','layers':2,'batch':1024,'micro_batch':128,'lr':1e-5,'learn_row_scales':True}
  original=command(plan,arm)
  continuation=dict(plan,profile='scales-rest',run_namespace='scales')
  self.assertEqual(original,command(continuation,arm))
 def test_hard_negative_comparison_changes_only_expected_flags(self):
  plan={'profile':'hardneg','data_inputs':{'train':{'path':'training-v2/work/hardneg/data/train.jsonl'},'dev':{'path':'training-v2/work/hardneg/data/dev.jsonl'}},'training':{'teacher_targets':False,'epochs_each':2,'validation_negatives':True}}
  a={'id':'float-pairs','checkpoint':'training-v2/work/hardneg/starts/float.pt','layers':2,'batch':1024,'micro_batch':128,'lr':4e-5};b=dict(a,id='float-hard',use_negatives=True)
  out1,c1=command(plan,a);out2,c2=command(plan,b);self.assertNotEqual(out1,out2);self.assertNotIn('--use-negatives',c1);self.assertIn('--use-negatives',c2);self.assertIn('--validation-negatives',c1)
  c1[c1.index('--out')+1]='OUTPUT';c2[c2.index('--out')+1]='OUTPUT';c2.remove('--use-negatives');self.assertEqual(c1,c2)
 def test_rejects_input_outside_project(self):
  plan={'profile':'hardneg','data_inputs':{'train':{'path':'/etc/passwd'},'dev':{'path':'training-v2/dev.jsonl'}},'training':{'teacher_targets':False,'epochs_each':2}}
  arm={'id':'a','checkpoint':'training-v2/model.pt','layers':2,'batch':128,'micro_batch':128,'lr':1e-5}
  with self.assertRaises(ValueError):command(plan,arm)
 def test_data_novelty_control_routes_to_distinct_audited_input(self):
  plan={'profile':'science-mix','data_inputs':{'train':{'path':'training-v2/work/science-mix/train.jsonl'},'train_repeat':{'path':'training-v2/work/science-mix/train-repeat.jsonl'},'dev':{'path':'training-v2/work/science-mix/dev.jsonl'}},'training':{'teacher_targets':False,'epochs_each':2}}
  arm={'id':'float-repeat','checkpoint':'training-v2/work/science-mix/starts/float.pt','layers':2,'batch':1024,'micro_batch':128,'lr':4e-5,'data_key':'train_repeat'}
  _,args=command(plan,arm);self.assertTrue(args[args.index('--data')+1].endswith('train-repeat.jsonl'));self.assertTrue(args[args.index('--dev')+1].endswith('/dev.jsonl'))
if __name__=='__main__':unittest.main()
