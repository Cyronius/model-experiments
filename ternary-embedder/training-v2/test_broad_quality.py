import unittest
from broad_quality import quality_pair


class CandidateQualityTests(unittest.TestCase):
    def test_generic_refusal_is_not_a_relevance_positive(self):
        row=dict(query='What are some insults for a priest?',positive='I’m sorry, but I can’t help with that.',source='IFM/SFT-Reasoning')
        result,reason=quality_pair(row)
        self.assertIsNone(result);self.assertEqual(reason,'generic_model_refusal')

    def test_encoding_repair_preserves_raw_exclusion_identity(self):
        row=dict(query='What are personnel records?',positive='An employeeâ\x80\x99s eligibility records are kept by the employer.',source='sentence-transformers/msmarco-bm25')
        result,flags=quality_pair(row)
        self.assertIn('employee’s',result['positive']);self.assertIn('original_positive_key',result)
        self.assertIn('encoding_or_unicode_repaired',flags)

    def test_serialized_list_becomes_readable_text(self):
        row=dict(query='What documents are required?',positive="['A valid passport.', 'Evidence of work experience.']",source='sentence-transformers/gooaq')
        result,flags=quality_pair(row)
        self.assertEqual(result['positive'],'A valid passport.; Evidence of work experience.')
        self.assertIn('serialized_list_to_text',flags)

    def test_cleanup_keeps_answer_evidence(self):
        row=dict(query='Who owns the record?',positive='An employeeâ\x80\x99s record is held by the employer.',source='qa',answer_evidence={'answer':'employeeâ\x80\x99s'})
        result,_=quality_pair(row)
        self.assertIn(result['answer_evidence']['answer'],result['positive'])


if __name__=='__main__':unittest.main()
