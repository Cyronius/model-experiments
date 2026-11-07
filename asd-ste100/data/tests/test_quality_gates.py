"""Deterministic quality gates: actual failures and counterexamples to broad rules."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quality_gates import unsafe_changes, target_issues, bad_merge_marker, inspect_pair
from check_quality import run


def rules(issues):
    return {i['rule'] for i in issues}


class GateTests(unittest.TestCase):
    def test_math_variables_not_articles(self):
        self.assertIn('math_variable_changed',rules(unsafe_changes('x ≥ 2 or ≥ 5.','x ≥ 2 or a ≥ 5.')))
        self.assertFalse(unsafe_changes('Open file.','Open the file.'))
        self.assertFalse(unsafe_changes('Use a <div> element.','Use the <div> element.'))
        self.assertFalse(unsafe_changes('Evaluate x ≥ 2 in result.','Evaluate x ≥ 2 in the result.'))

    def test_raw_calls_nested_strings_and_vb_are_protected(self):
        self.assertIn('code_literal_changed',rules(unsafe_changes('log("Open file.")','log("Open the file.")')))
        self.assertIn('code_literal_changed',rules(unsafe_changes('If x Is Nothing Then Fail("file") End If','If x Is Nothing Then Fail("the file") End If')))
        self.assertFalse(unsafe_changes('Call f(g("x)")) in file.','Call f(g("x)")) in the file.'))
        self.assertFalse(unsafe_changes('Use the function (optional).','Use a function (optional).'))

    def test_markup_and_inline_code_copy(self):
        self.assertIn('markup_changed',rules(unsafe_changes('Use <div>.','Use <span>.')))
        self.assertIn('code_literal_changed',rules(unsafe_changes('Use `a = 3`.','Use `a = 4`.')))
        self.assertIn('empty_markup_reference',rules(target_issues('Read the ` ` element.')))
        self.assertIn('empty_markup_reference',rules(target_issues('Read the ` ` tags.')))
        self.assertFalse(unsafe_changes('L<the caption|some::path>', 'L<a caption|some::path>'))
        self.assertNotIn('empty_markup_reference',rules(target_issues('Use ` ` as the delimiter.')))

    def test_contractions_in_prose_but_not_literals_or_possessives(self):
        self.assertIn('target_contraction',rules(target_issues('It’s a file.')))
        self.assertIn('target_contraction',rules(target_issues("They don't use it.")))
        self.assertNotIn('target_contraction',rules(target_issues("Use John’s file.")))
        self.assertNotIn('target_contraction',rules(target_issues('Replace "it’s" with "it is".')))
        self.assertNotIn('target_contraction',rules(target_issues('Print("It’s ready.")')))
        self.assertNotIn('target_contraction',rules(target_issues('Replace “it’s” with “it is”.')))
        self.assertNotIn('target_contraction',rules(target_issues('The title is ‘Who’s the reader?’.')))

    def test_damage_patterns_and_legitimate_endings(self):
        self.assertIn('explicit_truncation',rules(target_issues('Read the given...(truncated)')))
        self.assertIn('unfinished_copula',rules(target_issues('The recommended structure is')))
        for valid in ('The key is `is`.', 'The value is "is".', 'It is what it is.', 'What a day this is.', 'The larger the border, the larger the triangle will be.', 'The more points used, the more accurate the estimate will be.', 'The file is as it is.', 'The goal is to reach the same square the princess is.'):
            self.assertNotIn('unfinished_copula',rules(target_issues(valid)),valid)
        self.assertIn('dangling_step_heading',rules(target_issues('Add the values. Step 3.')))
        self.assertNotIn('dangling_step_heading',rules(target_issues('Go to Step 3.')))
        self.assertTrue(all(i['action']=='review' for i in target_issues('Step 3.')))

    def test_numbered_list_merge_only_known_noise_and_exact_change(self):
        self.assertTrue(bad_merge_marker('1 and Define the pattern','1. Define the pattern','merge_sentences'))
        self.assertFalse(bad_merge_marker('1 and 2 are numbers.','1 and 2 are numbers.','merge_sentences'))
        self.assertFalse(bad_merge_marker('1 and Define the pattern','1. Define the pattern',None))
        self.assertFalse(bad_merge_marker('Less than 30 and Next sentence.', 'Less than 30. Next sentence.', 'merge_sentences'))
        self.assertFalse(bad_merge_marker('Value 0.435 and Next sentence.', 'Value 0.435. Next sentence.', 'merge_sentences'))
        self.assertFalse(bad_merge_marker('Open file and read it.','Open file. Read it.','merge_sentences'))

    def test_future_builders_preserve_code_and_block_known_failures(self):
        from harvest_sources import chunks
        from noise_rules import Noiser
        from build_local_million import target_quality
        from test_pairs import lexicon
        code = 'Use `<div  class="box">` and `[a](b)` in the example.'
        self.assertEqual(list(chunks(code)),[code])
        literal = 'Use `Set  a = 3. Next  value.` in the example.'
        self.assertEqual(list(chunks(literal)),[literal])
        self.assertEqual(target_quality('The response structure is','PROC',None),'quality_unfinished_copula')
        with tempfile.TemporaryDirectory() as work:
            path=Path(work)/'lex';lexicon(path);noiser=Noiser(path)
            self.assertTrue(noiser.candidates('Open the file.','PROC'))
            self.assertFalse(any(op=='drop_article' for op,_ in noiser.candidates('Set a ≥ 3.','PROC')))
            self.assertFalse(noiser.candidates('Print("Open the file.")','PROC'))
            self.assertFalse(any(op=='merge_sentences' for op,_ in noiser.candidates('1. Open the file.','PROC')))
            self.assertTrue(any(op=='merge_sentences' for op,_ in noiser.candidates('Set the value to 30. Open the file.','PROC')))

    def test_filter_accounting_and_review_flags_do_not_exclude(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work);source=root/'source';source.mkdir()
            base={'mode':'PROC','kind':'local_rule','noise':'drop_article','split':'train','meaning_review':'unjudged_rule','bucket':64,'group':'fixed','origin_id':'fixed'}
            pairs=[dict(base,id='ok',src='Open file.',ste='Open the file.'),
                   dict(base,id='bad',src='The structure is',ste='The structure is'),
                   dict(base,id='review',src='Add values. Step 3.',ste='Add the values. Step 3.')]
            (source/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs))
            for split in ('dev','test'):(source/f'{split}.jsonl').write_text('')
            (source/'summary.json').write_text(json.dumps({'pairs':3}))
            out=root/'out';report=run(source,out,True)
            self.assertEqual((report['scanned'],report['kept'],report['excluded']),(3,2,1))
            from check_quality import input_paths,records
            kept=[r for p,_ in input_paths(out) for r in records(p)]
            self.assertEqual(kept,[pairs[0],pairs[2]])
            from validate_quality import validate
            self.assertTrue(validate(source,out)['passed'])
            self.assertTrue(all(r['group']=='fixed' for r in kept))
            # Rehash a tampered output: the independent audit must still catch
            # a changed review label rather than trusting the shard manifest.
            import gzip
            from check_quality import sha256
            summary=json.loads((out/'summary.json').read_text())
            shard=out/summary['shards'][0]['path']
            kept[0]['meaning_review']='invented_gold'
            with gzip.open(shard,'wt') as stream:
                for row in kept:stream.write(json.dumps(row)+'\n')
            summary['shards'][0]['sha256']=sha256(shard)
            (out/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,'payload'):
                validate(source,out)
            with self.assertRaises(ValueError):run(source,out,True)


if __name__=='__main__':unittest.main()
