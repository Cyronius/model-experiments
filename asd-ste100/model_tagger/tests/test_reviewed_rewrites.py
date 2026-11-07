import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from safe_editor import validate


class ReviewedRewritesTests(unittest.TestCase):
    def test_only_reviewed_phrases_are_allowed(self):
        for a,b in [('Run `sync_v8()` in order to save the file.','Run `sync_v8()` to save the file.'),
                    ('Close `pipe_v8` prior to shutdown.','Close `pipe_v8` before shutdown.'),
                    ('In the event that `ok_v8` is false, stop.','If `ok_v8` is false, stop.'),
                    ('Stop due to the fact that the file is missing.','Stop because the file is missing.'),
                    ('Start at this point in time.','Start at this time.'),
                    ('Open the file subsequent to shutdown.','Open the file after shutdown.')]:
            self.assertFalse(validate(a,b),(a,b))
    def test_modals_negation_order_and_literals_remain_guarded(self):
        for a,b in [('You should stop.','You must stop.'),('Do not delete `v8`.','Delete `v8`.'),
                    ('Close the file before shutdown.','Close the file after shutdown.'),
                    ('Run `in order to`.','Run `to`.'),('The worker seeks a result.','The worker gets a result.')]:
            self.assertTrue(validate(a,b),(a,b))
    def test_duplicate_articles_outside_code_are_rejected(self):
        self.assertIn('duplicate_article',validate('Open the file.','Open the the file.'))
        self.assertNotIn('duplicate_article',validate('Use `the the`.','Use `the the`.'))

if __name__=='__main__':unittest.main()
