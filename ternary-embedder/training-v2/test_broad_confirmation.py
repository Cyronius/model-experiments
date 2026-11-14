import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import confirm_broad_reserved as confirmation


class ConfirmationTests(unittest.TestCase):
    def test_failed_development_gate_never_fetches_judgments(self):
        with patch.object(confirmation,'read',return_value={'passed':False,'candidate':None}),patch.object(confirmation.urllib.request,'urlopen') as network:
            with self.assertRaises(AssertionError):confirmation.activate()
            network.assert_not_called()

    def test_preview_query_is_completely_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            corpus=root/'corpus.jsonl';corpus.write_text(json.dumps({'_id':'doc','text':'supporting text'})+'\n')
            queries=root/'queries.jsonl';queries.write_text(''.join(json.dumps({'_id':str(i),'text':'q'+str(i)})+'\n' for i in (1,2)))
            (root/'webis-touche2020-test.tsv').write_text('query-id\tcorpus-id\tscore\n1\tdoc\t2\n2\tdoc\t1\n')
            lock={'datasets':{'webis-touche2020':{'text_files':{'corpus.jsonl':{'path':str(corpus),'sha256':'corpus'},'queries.jsonl':{'path':str(queries),'sha256':'queries'}}}}}
            def metadata(path):
                return lock if path.name=='evaluation.lock.json' else {'excluded_query_ids':{'webis-touche2020':['1']}}
            with patch.object(confirmation,'OUT',root),patch.object(confirmation,'read',side_effect=metadata):
                result=confirmation.data('webis-touche2020')
            self.assertEqual(result['qids'],['2'])
            self.assertEqual(result['qrels'],{'2':{'doc':1}})

    def test_bootstrap_preserves_constant_positive_difference(self):
        left={'task':[{'query_id':str(i),'ndcg10':.2} for i in range(3)]}
        right={'task':[{'query_id':str(i),'ndcg10':.3} for i in range(3)]}
        result=confirmation.contrast(left,right,tasks=('task',))
        self.assertAlmostEqual(result['mean_delta_ndcg10'],10)
        for value in result['paired_query_bootstrap_95_ci']:self.assertAlmostEqual(value,10)
        self.assertEqual((result['seed'],result['resamples']),(20261025,5000))


if __name__=='__main__':unittest.main()
