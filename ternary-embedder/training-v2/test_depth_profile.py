import unittest
import pair_runpod as job
class ProfileTests(unittest.TestCase):
 def tearDown(self):job.configure('data-first')
 def test_profile_isolates_state_backups_workload_and_watchdog_inputs(self):
  old=job.STATE;job.configure('depth3')
  self.assertNotEqual(job.STATE,old);self.assertEqual(job.STATE.parent.name,'depth3-run');self.assertEqual(job.PLAN.parent.name,'depth3');self.assertIn("glob('depth3-*')",job.SNAPSHOT);self.assertEqual(job.RUN_GLOB,'depth3-*');self.assertIn('run_depth3.sh',job.FILES)
  job.configure('data-first');self.assertEqual(job.STATE,old);self.assertIn("glob('data-first-*')",job.SNAPSHOT)
 def test_continuation_has_a_separate_bill_and_same_resume_namespace(self):
  job.configure('scales-rest')
  self.assertEqual(job.STATE.parent.name,'scales-rest-run')
  self.assertEqual(job.RUN_GLOB,'scales-*')
  self.assertIn("glob('scales-*')",job.SNAPSHOT)
  self.assertIn('run_scales_rest.sh',job.FILES)
if __name__=='__main__':unittest.main()
