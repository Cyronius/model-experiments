import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pair_runpod as operator


class BroadJobTests(unittest.TestCase):
    def test_profile_uses_its_own_workload_and_recovery_namespace(self):
        try:
            operator.configure('broad-data')
            self.assertEqual(operator.WORKLOAD,'run_broad_data.sh')
            self.assertEqual(operator.RUN_GLOB,'broad-data-*')
            self.assertIn('run_broad_data.sh',operator.FILES)
            self.assertIn("glob('broad-data-*')",operator.SNAPSHOT)
        finally:operator.configure('data-first')

    def test_no_spending_without_new_approval(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(operator,'WORK',Path(directory)),patch.object(operator.pod,'api') as api:
            with self.assertRaisesRegex(RuntimeError,'Explicit job-specific spending approval'):
                operator.launch()
            api.assert_not_called()


if __name__=='__main__':unittest.main()
