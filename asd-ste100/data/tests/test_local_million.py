"""Tests for free-data extraction and the distinct checker-only dataset schema."""
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_local_million import SAFE_OPERATIONS, align_fast, encoded_pair, stage, target_quality
from build_pairs import align
from export_local_million import export
from harvest_sources import chunks, sha256, texts
from noise_rules import Noiser
from checker import Lexicon
from text_utils import protected_values
from validate_local_million import validate
from resume_local_stage import records_after, starting_offsets
from parallel_local_stage import stage_parallel


class HarvestTests(unittest.TestCase):
    def test_resume_placeholders_preserve_record_numbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            rows=[{'index':i} for i in range(9)]
            path=base/'raw.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            self.assertEqual(list(records_after(path,4)),[{}]*4+rows[4:])
            import pyarrow as pa
            import pyarrow.parquet as pq
            path=base/'raw.parquet'
            pq.write_table(pa.Table.from_pylist(rows),path,row_group_size=3)
            self.assertEqual(list(records_after(path,4)),[{}]*4+rows[4:])
            self.assertEqual(list(records_after(path,9)),[{}]*9)

    def test_message_schemas_and_roles(self):
        messages = [{'role': 'system', 'content': 'Use short words.'}, {'role': 'user', 'content': 'Open the file.'}]
        self.assertEqual(list(texts({'messages': messages}, 'messages')), list(texts({'messages': json.dumps(messages)}, 'messages')))
        self.assertEqual(list(texts({'problem': 'Write a function.', 'solution': 'Use a loop.'}, 'magicoder')),
                         [('user', 'Write a function.'), ('assistant', 'Use a loop.')])

    def test_no_mid_sentence_truncation_or_code_fragments(self):
        long = ' '.join(['word'] * 70) + '.'
        text = 'Read the file and keep the number.\n\n```python\nsecret_code(12)\n```\n\n' + long + '\n\nOpen the other file and read it.'
        result = list(chunks(text))
        self.assertEqual(result, ['Read the file and keep the number.', 'Open the other file and read it.'])


class LocalPairsTests(unittest.TestCase):
    def test_trimmed_alignment_has_same_minimum_cost(self):
        sequences = [list(seq) for n in range(4) for seq in itertools.product('ab', repeat=n)]
        for source in sequences:
            for target in sequences:
                reference, _, _ = align(source, target)
                actual, positions, required = align_fast(source, target)
                self.assertEqual(sum(x[0] != 'keep' for x in actual), sum(x[0] != 'keep' for x in reference))
                i = j = 0
                for operation, si, tj in actual:
                    self.assertEqual((si, tj), (i, j))
                    if operation != 'insert':
                        i += 1
                    if operation != 'delete':
                        j += 1
                self.assertEqual((i, j), (len(source), len(target)))
                self.assertEqual(len(positions), len(target))
                self.assertEqual(positions, sorted(set(positions)))
                self.assertGreaterEqual(required, 2 * len(source))

    def test_checker_only_labels_are_not_judge_scores(self):
        item = ('pair1', 'origin1', 'PROC', 'Open file.', 'Open the file.', 'drop_article',
                'fixture', json.dumps(['doc1']), json.dumps({'license': 'fixture'}), 'train', 'group1')
        split, bucket, line = encoded_pair(item)
        row = json.loads(line)
        self.assertEqual(row['target_review'], 'checker_only')
        self.assertEqual(row['meaning_review'], 'unjudged_rule')
        self.assertNotIn('score', row)
        self.assertNotIn('changed', row)
        self.assertEqual([t for t in row['source_canvas'] if t is not None], row['src_tokens'])
        self.assertEqual([t for t in row['target_canvas'] if t is not None], row['tgt_tokens'])

    def test_no_lexical_substitutions_in_safe_variant_pool(self):
        noiser = Noiser()
        text = 'Use the file. Open the other file.'
        for variant in noiser.variants(text, 'PROC', 100, 0, SAFE_OPERATIONS):
            self.assertNotEqual(variant['noise'], 'reverse_dictionary')
            self.assertEqual(protected_values(variant['src']), protected_values(text))

    def test_raw_targets_reject_errors_warnings_and_unknown_floods(self):
        lex = Lexicon()
        self.assertIsNone(target_quality('Open the file and read the text.', 'PROC', lex))
        self.assertEqual(target_quality('Utilize the file and read the text.', 'PROC', lex), 'checker')
        self.assertEqual(target_quality('Qzzza bzzzb zzzzc gzzzd hzzze.', 'DESC', lex), 'unknown_coverage')
        self.assertEqual(target_quality('The result is shown as follows:', 'DESC', lex), 'fragment')
        self.assertEqual(target_quality('! ! Close the file. ! CALL h5fclose_f(file_id, errcode)', 'PROC', lex), 'code_fragment')

    def test_variables_and_object_pronouns_survive_noise(self):
        noiser=Noiser()
        text='The arrays A and B have the same size.'
        self.assertTrue(all(' A ' in v['src'] for v in noiser.variants(text,'DESC',100,0,SAFE_OPERATIONS)))
        self.assertTrue(all(v['src'].startswith('A ') for v in noiser.variants('A and B are the arrays.','DESC',100,0,SAFE_OPERATIONS)))
        text='Use the steps to get you started.'
        self.assertNotIn('perfect_tense',{v['noise'] for v in noiser.variants(text,'DESC',100,0,SAFE_OPERATIONS)})

    def test_end_to_end_resume_grouping_and_compressed_reproducibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary); source=base/'source'; source.mkdir()
            manifest={'datasets':[],'files':[]}
            for key in ('fixture-a','fixture-b'):
                path=source/(key+'.jsonl')
                rows=[{'problem':f'Open the file{key[-1]}{i} and read the text.',
                       'solution':f'The file{key[-1]}{i} has the text of the test.'} for i in range(60)]
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
                manifest['datasets'].append({'key':key,'repo':key+'/fixture','revision':'test-fixture',
                                             'license':'mit','card':'https://example.invalid/fixture',
                                             'format':'magicoder','domain':'software','export_weight':1})
                manifest['files'].append({'dataset':key,'name':path.name,'path':path.name,'sha256':sha256(path)})
            (source/'download_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            output=base/'pairs'
            first=stage(source,output,n=100,exclude=None)
            self.assertEqual(first['staged_pairs'],100)
            self.assertEqual(first,stage(source,output,n=100,exclude=None))
            one=export(output,workers=1,shard_size=17)
            self.assertEqual(one['pairs'],100)
            self.assertEqual(validate(output)['pairs_checked'],100)
            original=json.loads((output/'summary.json').read_text(encoding='utf-8'))['shards']
            export(output,workers=2,shard_size=17)
            repeated=json.loads((output/'summary.json').read_text(encoding='utf-8'))['shards']
            self.assertEqual(original,repeated)
            self.assertEqual(validate(output)['status'],'passed')
            parallel_output=base/'parallel-pairs'
            parallel=stage_parallel(source,parallel_output,n=100,workers=2,exclude=None)
            self.assertEqual(parallel['staged_pairs'],100)
            export(parallel_output,workers=2,shard_size=17)
            self.assertEqual(validate(parallel_output)['status'],'passed')


if __name__ == '__main__':
    unittest.main()
