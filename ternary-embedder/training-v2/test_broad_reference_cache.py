import tempfile
import unittest
from pathlib import Path
from broad_reference_cache import load_cache,save_cache


class ReferenceCacheTests(unittest.TestCase):
    def test_roundtrip_and_changed_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'keys.npz';identity={'reference_sha256':'old','token_cap':128}
            expected=({b'a'*16},{'b'*64},{b'c'*16})
            save_cache(path,identity,*expected)
            self.assertEqual(load_cache(path,identity),expected)
            self.assertIsNone(load_cache(path,dict(identity,reference_sha256='new')))

    def test_tampered_cache_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'keys.npz';identity={'token_cap':128}
            save_cache(path,identity,{b'a'*16},{'b'*64},{b'c'*16})
            path.write_bytes(path.read_bytes()+b'tampered')
            with self.assertRaisesRegex(RuntimeError,'checksum'):load_cache(path,identity)


if __name__=='__main__':unittest.main()
