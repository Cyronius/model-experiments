import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from broad_pairs import answer_window,candidates,instruction_pair,prepare_batch,bert_prefix
from broad_data_fetch import domain_for
from broad_lcqa_pilot import generated_pair


class BroadPairsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tk=AutoTokenizer.from_pretrained(Path(__file__).resolve().parents[1]/'tokenizer',local_files_only=True)

    def test_answer_near_end_survives_window(self):
        context='An ordinary introductory sentence. '*150+'The final explorer was Roald Amundsen. '+'A final sentence. '*30
        passage,evidence=answer_window(context,['Roald Amundsen'],self.tk)
        self.assertIn('Roald Amundsen',passage)
        self.assertLessEqual(len(self.tk(passage)['input_ids']),128)
        self.assertGreater(evidence['original_char_start'],100)

    def test_missing_or_oversize_answer_rejected(self):
        self.assertIsNone(answer_window('The explorer was Amundsen.',['Shackleton'],self.tk))
        text=' '.join('word'+str(i) for i in range(200))
        self.assertIsNone(answer_window(text,[text],self.tk))

    def test_annotated_occurrence_not_earlier_matching_name(self):
        prefix='Amundsen was mentioned in a different discussion. '+'An introductory sentence. '*160
        context=prefix+'The expedition leader was Amundsen, who reached the pole.'
        start=context.rfind('Amundsen')
        passage,evidence=answer_window(context,[dict(text='Amundsen',start=start)],self.tk)
        self.assertIn('expedition leader',passage)
        self.assertGreater(evidence['original_char_start'],100)

    def test_only_trusted_stack_answers_no_negative(self):
        row=dict(question='Why does water freeze in cold weather?',metadata=['https://physics.stackexchange.com/questions/1','https://physics.stackexchange.com'],
                 answers=[dict(answer_id=1,text='Accepted answer with a complete physical explanation.',pm_score=4,selected=True),dict(answer_id=2,text='A less preferred answer can still be relevant.',pm_score=-3,selected=False)])
        got=candidates(row,dict(kind='stack'),{})
        self.assertEqual(len(got),1);self.assertNotIn('negative',got[0]);self.assertEqual(got[0]['origin']['answer']['answer_id'],1)

    def test_instruction_reasoning_is_not_positive(self):
        text='### Instruction\nWhy is the sky blue?\n### Reasoning\nThis reasoning should be excluded.\n### Response\nAir scatters shorter wavelengths of sunlight more strongly.'
        q,a=instruction_pair(text);self.assertEqual(q,'Why is the sky blue?');self.assertNotIn('reasoning',a)
        self.assertIsNone(instruction_pair('Unstructured instruction and reasoning without reliable boundaries.'))

    def test_long_unannotated_positive_is_not_blindly_cropped(self):
        items=[dict(query='Why is this explanation useful?',positive='An ordinary long explanation. '*200,group='g',origin={})]
        rows,reasons=prepare_batch(items,self.tk)
        self.assertFalse(rows);self.assertEqual(reasons['positive_words_exceed_cap'],1)

    def test_community_cap_uses_kind_for_versioned_source(self):
        self.assertEqual(domain_for(dict(id='stack-balanced',kind='stack'),'data/money.stackexchange.com/train-000.parquet'),'money.stackexchange.com')
        self.assertEqual(domain_for(dict(id='paq'),'pair/train-000.parquet'),'paq')

    def test_generated_question_requires_exact_source_evidence(self):
        passage='The ship reached the harbor in September 1912 after a lengthy voyage across the ocean.'
        pair=generated_pair(dict(question='When did the ship reach the harbor?',answer_quote='September 1912'),passage,self.tk)
        self.assertIsNotNone(pair);self.assertIn('September 1912',pair['positive'])
        self.assertIsNone(generated_pair(dict(question='When did it arrive?',answer_quote='January 1900'),passage,self.tk))
        self.assertIsNone(generated_pair(dict(question='When did it arrive?',answer_quote='September 1912',think='extra field'),passage,self.tk))

    def test_prefix_has_identical_model_visible_tokens(self):
        texts=['Short input.', 'antidisestablishmentarianism '*400, ('[MASK] 中文a x\u200b e\u0301 1234 &amp; https://example.org ')*100, '中文没有空格']
        expected=self.tk(texts,truncation=True,max_length=128)['input_ids']
        actual=self.tk([bert_prefix(t) for t in texts],truncation=True,max_length=128)['input_ids']
        self.assertEqual(actual,expected)
        self.assertLess(len(bert_prefix(texts[1])),len(texts[1]))


if __name__=='__main__':unittest.main()
