import unittest
from src.training.diagnostics import operating_points, error_summary


class DiagnosticsTests(unittest.TestCase):
    def test_threshold_direction_and_ties(self):
        result = operating_points([.1, .3, .3, .8], [1, 1, 0, 0], targets=[0, .5])
        self.assertEqual(result[0]['far'], 0)
        self.assertEqual(result[0]['frr'], .5)
        self.assertEqual(result[1]['far'], .5)
        self.assertEqual(result[1]['frr'], 0)
        tied = operating_points([.5]*4, [1, 0, 1, 0], targets=[.01])[0]
        self.assertEqual(tied['far'], 0)
        self.assertEqual(tied['frr'], 1)

    def test_missing_bucket_class_is_not_zero_error(self):
        self.assertIsNone(error_summary([1], [True])['far'])
        self.assertIsNone(error_summary([], [])['frr'])
