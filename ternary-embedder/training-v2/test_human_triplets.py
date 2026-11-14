import unittest

from prepare_human_triplets import triplets


def row(premise, hypothesis, label, index=0):
    return dict(sentence1=premise, sentence2=hypothesis, visible1=premise,
                visible2=hypothesis, label=label, source_row=index)


class HumanTripletTests(unittest.TestCase):
    def test_multiple_positives_keep_same_premise_human_negative(self):
        rows = [row('a', 'yes1', 0), row('a', 'yes2', 0), row('a', 'no', 2),
                row('a', 'maybe', 1), row('other', 'other-no', 2)]
        result, _ = triplets(rows)
        self.assertEqual({r['positive'] for r in result}, {'yes1', 'yes2'})
        self.assertTrue(all(r['negative'] == 'no' and r['query'] == 'a' for r in result))
        self.assertEqual(result, triplets(list(reversed(rows)))[0])

    def test_visible_conflict_and_neutral_are_never_negative(self):
        result, _ = triplets([row('a', 'yes', 0), row('a', 'yes', 2), row('a', 'maybe', 1)])
        self.assertEqual(result, [])

    def test_positive_from_another_premise_cannot_be_used(self):
        result, _ = triplets([row('a', 'yes', 0), row('b', 'no', 2)])
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()
