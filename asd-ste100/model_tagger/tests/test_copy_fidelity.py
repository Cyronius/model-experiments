import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from copy_fidelity import preserved


class CopyTests(unittest.TestCase):
    def test_adjacent_names_can_merge_on_retokenization(self):
        slots=[{'surface':'Alpha Beta','count':1},{'surface':'Gamma Delta','count':1}]
        self.assertTrue(preserved('Use Alpha Beta Gamma Delta.',slots))

    def test_nested_literal_does_not_count_twice(self):
        slots=[{'surface':'`foo`','count':1},{'surface':'foo','count':1}]
        self.assertTrue(preserved('Use `foo` and foo.',slots))
        self.assertFalse(preserved('Use `foo`.',slots))

    def test_boundaries_case_and_counts(self):
        slots=[{'surface':'fooBar','count':1}]
        for bad in ('foobar','fooBarX','fooBar fooBar','no copy'):
            self.assertFalse(preserved(bad,slots),bad)
        self.assertTrue(preserved('fooBar.',slots))
        self.assertTrue(preserved('Use`x`.',[{'surface':'`x`','count':1}]))


if __name__=='__main__':unittest.main()
