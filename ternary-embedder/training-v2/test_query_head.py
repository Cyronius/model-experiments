import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tern-embed'))
from model import Student
from query_head import install_query_head, query_role


class QueryHeadTests(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parent / 'ternary-blends/results/frozen.json').exists(), 'Local model artifact unavailable')
    def test_actual_encoder_keeps_query_and_document_caches_separate(self):
        from evaluate_round2_mteb import Encoder
        root = Path(__file__).resolve().parent
        selection = json.loads((root / 'ternary-blends/results/frozen.json').read_text())['models']['ternary-3-25']
        source = torch.load(selection['path'], map_location='cpu', weights_only=False)
        torch.manual_seed(83)
        torch.set_num_threads(2)
        projection = torch.eye(384) + .001 * torch.randn(384, 384)
        source['meta'] = dict(query_projection=projection)
        texts = ['Scientific evidence supports the claim.', 'A small model produces an embedding.']
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'roles.pt'
            torch.save(source, path)
            original = Encoder(selection['path'])
            model = Encoder(path)
            documents = model.encode_documents(texts)
            self.assertTrue((documents == original.encode(texts)).all())
            expected = F.normalize(torch.from_numpy(documents) @ projection.T, dim=-1).numpy()
            torch.testing.assert_close(torch.from_numpy(model.encode_queries(texts)), torch.from_numpy(expected), atol=2e-6, rtol=2e-6)
            self.assertTrue((documents == model.encode(texts)).all())
            self.assertEqual(len(model.cache), 4)

    def test_query_fold_preserves_documents_and_ordinary_checkpoint_state(self):
        torch.manual_seed(83)
        torch.set_num_threads(2)
        original = Student(vocab=64, emb=8, d=8, heads=2, ffn=16, layers=3, max_pos=12, out_dim=8).eval()
        original.set_quant(True)
        with torch.no_grad():
            original.out.weight.copy_(torch.randn(8, 8))
        model = copy.deepcopy(original)
        matrix = torch.eye(8) + .03 * torch.randn(8, 8)
        ids = torch.randint(0, 64, (5, 12))
        mask = torch.arange(12)[None, :] < torch.tensor([12, 5, 9, 2, 6])[:, None]
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'query.pt'
            torch.save(dict(state=model.state_dict(), meta=dict(query_projection=matrix)), checkpoint)
            install_query_head(model, checkpoint)
        with torch.inference_mode():
            document = original(ids, mask)[2]
            self.assertTrue(torch.equal(document, model(ids, mask)[2]))
            with self.assertRaisesRegex(RuntimeError, 'restore role'):
                with query_role(model):
                    raise RuntimeError('restore role')
            self.assertTrue(torch.equal(document, model(ids, mask)[2]))
            with query_role(model):
                torch.testing.assert_close(model(ids, mask)[2], F.normalize(document @ matrix.T, dim=-1), atol=2e-6, rtol=2e-6)
            self.assertTrue(torch.equal(document, model(ids, mask)[2]))
        self.assertEqual(list(model.state_dict()), list(original.state_dict()))
        self.assertTrue(model.quantized)
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))


if __name__ == '__main__':
    unittest.main()
