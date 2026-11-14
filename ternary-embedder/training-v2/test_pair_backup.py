import contextlib,io,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pair_runpod as job
from pair_training_state import digest

class BackupTests(unittest.TestCase):
 def run_backup(self,corrupt=False,previous=False):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);source=root/'remote.pt';source.write_bytes(b'complete checkpoint contents');rel='data-first-ternary/checkpoint-0000500.pt';target=root/'backups'/rel
   if previous:target.parent.mkdir(parents=True);target.write_bytes(b'previous verified checkpoint')
   payload={'snapshot':'/workspace/ternary-pilot/pull-test','files':[{'remote':str(source),'relative':rel,'sha256':digest(source),'bytes':source.stat().st_size}]}
   def remote(state,command,timeout=0):return json.dumps(payload) if command.startswith('python') else ''
   def scp(state,src,dst,timeout=0):Path(dst).write_bytes(b'corrupt' if corrupt else Path(src).read_bytes())
   with patch.object(job,'WORK',root),patch.object(job.pod,'remote',remote),patch.object(job.pod,'scp',scp):
    if corrupt:
     with self.assertRaisesRegex(RuntimeError,'checksum'):job.backup({})
     self.assertEqual(target.read_bytes(),b'previous verified checkpoint');self.assertFalse((root/'backups/verified.json').exists())
    else:
     self.assertTrue(job.backup({}));self.assertEqual(target.read_bytes(),source.read_bytes());receipt=json.loads((root/'backups/verified.json').read_text());self.assertEqual(receipt[rel]['sha256'],digest(target))
 def test_remote_snapshot_pins_matching_latest_pointer(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);run=root/'ternary-embedder/training-v2/runs/data-first-float';run.mkdir(parents=True)
   ck=run/'checkpoint-0000500.pt';ck.write_bytes(b'valid immutable checkpoint')
   (run/'latest.json').write_text(json.dumps({'file':ck.name,'sha256':digest(ck),'step':500}))
   capture=io.StringIO()
   with contextlib.redirect_stdout(capture):exec(compile(job.SNAPSHOT.replace("'/workspace/ternary-pilot'",repr(str(root))),'snapshot','exec'),{})
   result=json.loads(capture.getvalue());entries={x['relative']:x for x in result['files']}
   pointer=Path(entries['data-first-float/latest.json']['remote']);info=json.loads(pointer.read_text());snapshot=Path(entries['data-first-float/'+info['file']]['remote'])
   self.assertEqual(digest(snapshot),info['sha256']);self.assertGreater(snapshot.stat().st_nlink,1)
 def test_verified_local_backup(self):self.run_backup()
 def test_corrupt_transfer_cannot_replace_last_good_copy(self):self.run_backup(corrupt=True,previous=True)
if __name__=='__main__':unittest.main()
