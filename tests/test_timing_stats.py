import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.pair_generation import PairIndex
from src.data.timing_stats import inspect_training, summarize


class TimingStatsTests(unittest.TestCase):
    def test_excludes_heldout_padding_and_first_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.npz'
            features = np.array([[[1, 0, 0], [2, -0.1, 0.2], [999, 999, 999]],
                                 [[999, 999, 999]] * 3], dtype=np.float32)
            np.savez(path, features=features, mask=[[True, True, False], [True]*3], seq_len=[2, 3])
            index = PairIndex(np.array(['train']), np.array(['a']), {'a': np.array([0])},
                              (path,), np.array([0]), np.array([0]), np.array([1]))
            result = inspect_training(index)
            self.assertEqual(result['sampled_sequences'], 1)
            self.assertEqual(result['features']['H']['count'], 2)
            self.assertEqual(result['features']['H']['percentiles_seconds']['100'], 2)
            self.assertEqual(result['features']['UD']['count'], 1)
            self.assertEqual(result['features']['UD']['negative_fraction'], 1)
            self.assertEqual(result, inspect_training(index))

    def test_reports_invalid_values_without_poisoning_quantiles(self):
        result = summarize(np.array([1, 2, np.nan, np.inf]))
        self.assertEqual(result['nonfinite'], 2)
        self.assertEqual(result['percentiles_seconds']['50'], 1.5)
