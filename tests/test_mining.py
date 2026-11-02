import unittest
import numpy as np
from src.training.mining import select_pairs


class MiningTests(unittest.TestCase):
    def test_matched_positives_and_hard_order(self):
        y = np.array([1]*8 + [0]*8)
        d = np.arange(16, dtype=float)
        options = dict(size=8, seed=123, epoch=30, step=1)
        random = select_pairs(y, d, mode='random', **options)
        hard = select_pairs(y, d, mode='hard', **options)
        np.testing.assert_array_equal(random[:4], hard[:4])
        np.testing.assert_array_equal(hard[4:], [8, 9, 10, 11])
        self.assertEqual(len(set(hard)), 8)
        np.testing.assert_array_equal(hard, select_pairs(y, d, mode='hard', **options))
