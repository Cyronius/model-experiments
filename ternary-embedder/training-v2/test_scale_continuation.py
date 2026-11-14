import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import continue_scales as continuation
from pair_training_state import digest


class ScaleContinuationTests(unittest.TestCase):
    def setup_files(self, root, corrupt=False):
        original, rest = root / 'work/scales-run', root / 'work/scales-rest-run'
        for path in (original, rest):
            path.mkdir(parents=True)
            (path / 'pod.json').write_text(json.dumps(dict(deleted=True, deletion_verified_http_status=404)))
        original_plan = root / 'scales/runpod-job.json'
        original_plan.parent.mkdir()
        original_plan.write_text('{}')
        directory = root / 'scales-rest'
        directory.mkdir()
        (directory / 'runpod-job.json').write_text(json.dumps(dict(
            original_plan_sha256=digest(original_plan), training=dict(arms=[dict(id='3-learned')]))))
        source = rest / 'backups/scales-3-learned'
        source.mkdir(parents=True)
        files = {'config.json': json.dumps(dict(total_steps=1004)),
                 'completed.json': json.dumps(dict(step=1004)), 'best.pt': 'verified trained weights'}
        receipt = {}
        for name, data in files.items():
            path = source / name
            path.write_text(data)
            receipt['scales-3-learned/' + name] = dict(sha256=digest(path), verified_unix=0)
        if corrupt:
            (source / 'best.pt').write_text('corrupt transfer')
        (rest / 'backups/verified.json').write_text(json.dumps(receipt))
        (original / 'backups').mkdir()
        (original / 'backups/verified.json').write_text('{}')
        return original, rest

    def test_merges_only_verified_files_and_records_closed_pods(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, rest = self.setup_files(root)
            with patch.multiple(continuation, ROOT=root, ORIGINAL=original, REST=rest):
                continuation.consolidate()
            self.assertEqual((original / 'backups/scales-3-learned/best.pt').read_text(), 'verified trained weights')
            self.assertTrue((original / 'backups/scales-3-learned/completed.json').exists())
            record = json.loads((original / 'continuations.json').read_text())[0]
            self.assertEqual(record['state_sha256'], digest(rest / 'pod.json'))

    def test_corruption_cannot_publish_a_completion_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, rest = self.setup_files(root, corrupt=True)
            with patch.multiple(continuation, ROOT=root, ORIGINAL=original, REST=rest):
                with self.assertRaises(AssertionError):
                    continuation.consolidate()
            self.assertFalse((original / 'backups/scales-3-learned/completed.json').exists())
            self.assertFalse((original / 'continuations.json').exists())


if __name__ == '__main__':
    unittest.main()
