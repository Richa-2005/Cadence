import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.data.normalization import TimingNormalizer, fit_normalizer
from src.data.pair_generation import load_pair_index, generate_fixed_pairs, write_pairs
from src.data.pair_dataset import PairDataset
from src.data.validity import valid_sequences


class NormalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # a and b have two valid sessions; c loses its only sample.
        # held-out data is extreme and must have no influence on fitting.
        for shard, subjects in enumerate((['a', 'b', 'c', 'test'], ['a', 'b', 'test'])):
            n = len(subjects)
            x = np.zeros((n, 4, 3), dtype=np.float32)
            for row, subject in enumerate(subjects):
                offset = shard * 0.02 + row * 0.01
                x[row, 0] = [0.1 + offset, 0, 0]
                x[row, 1] = [0.2 + offset, -0.05, 0.15 + offset]
                x[row, 2] = [0.15 + offset, 0.3 + offset, 0.5 + offset]
                x[row, 3] = np.nan  # masked padding must not contaminate statistics
                if subject == 'c':
                    x[row, 0, 0] = -1
                if subject == 'test':
                    x[row, :3] = 500
            mask = np.tile([True, True, True, False], (n, 1))
            np.savez(self.root / f'aalto_shard_{shard:04}.npz', features=x, mask=mask,
                     seq_len=np.full(n, 3), subject_id=np.array(subjects),
                     sample_id=np.array([f'{s}_{shard}' for s in subjects]),
                     session=np.full(n, shard + 1))
        self.split = self.root / 'split.json'
        self.split.write_text(json.dumps(dict(train=['a', 'b', 'c'], val=[], test=['test'])))
        self.index = load_pair_index('aalto', 'train', self.root, self.split)

    def test_filtering_preserves_assignments_and_excludes_bad_pairs(self):
        self.assertEqual(self.index.excluded_samples, 1)
        self.assertEqual(self.index.subjects_without_valid_samples, ('c',))
        self.assertEqual(set(self.index.subjects), {'a', 'b'})
        pairs = generate_fixed_pairs(self.index, 100)
        self.assertTrue(all(not str(x).startswith('c') for x in pairs[:, :2].flat))
        path = self.root / 'old_pairs.npz'
        write_pairs(np.array([['c_0', 'a_0', 0]], dtype=object), path)
        with self.assertRaises(ValueError):
            PairDataset(self.index, fixed_pairs_path=path, split_name='test')
        self.assertEqual(json.loads(self.split.read_text())['train'], ['a', 'b', 'c'])

    def test_fit_matches_direct_train_statistics_and_roundtrips(self):
        normalizer, counts = fit_normalizer([self.index])
        transformed = [[], [], []]
        for path in self.index.shard_paths:
            with np.load(path) as data:
                selected = np.isin(data['subject_id'], ['a', 'b'])
                x = data['features'][selected, :3].astype(np.float64)
            logged = np.sign(x) * np.log1p(np.abs(x) / 0.1)
            for channel in range(3):
                transformed[channel].extend(logged[:, (1 if channel else 0):, channel].ravel())
        self.assertEqual(counts.tolist(), [12, 8, 8])
        np.testing.assert_allclose(normalizer.mean, [np.mean(v) for v in transformed])
        np.testing.assert_allclose(normalizer.std, [max(np.std(v), 1e-8) for v in transformed])
        path = self.root / 'normalizer.json'
        normalizer.save(path)
        restored = TimingNormalizer.load(path)
        raw = PairDataset(self.index, loading='memory')
        normalized = PairDataset(self.index, loading='memory', normalizer=restored)
        for i in range(10):
            for side in ('a', 'b'):
                original = raw[i][side]
                result = normalized[i][side]['features'].numpy()
                expected = normalizer.transform(original['features'].numpy(), original['mask'].numpy())
                np.testing.assert_array_equal(result, expected)
                self.assertTrue(np.isfinite(result).all())
                self.assertTrue((result[3] == 0).all())
                self.assertTrue((result[0, 1:] == 0).all())
        heldout = load_pair_index('aalto', 'test', self.root, self.split)
        with self.assertRaises(ValueError):
            fit_normalizer([heldout])
        unfiltered = load_pair_index('aalto', 'train', self.root, self.split, filter_invalid=False)
        with self.assertRaises(ValueError):
            fit_normalizer([unfiltered])

    def test_validity_and_transform_guards(self):
        x = np.ones((5, 3, 3), dtype=np.float32) * 0.1
        mask = np.ones((5, 3), dtype=bool)
        x[0, 1, 1] = -0.1  # valid overlap
        x[1, 1, 0] = -0.1
        x[2, 1, 2] = -0.1
        x[3, 1, 1] = np.inf
        x[4, 1, 0] = np.nan
        np.testing.assert_array_equal(valid_sequences(x, mask), [True, False, False, False, False])
        normalizer = TimingNormalizer([0, 0, 0], [1, 1, 1])
        output = normalizer.transform(x[:1], mask[:1])
        self.assertLess(output[0, 1, 1], 0)
        for i in range(1, 5):
            with self.assertRaises(ValueError):
                normalizer.transform(x[i], mask[i])
        with self.assertRaises(ValueError):
            TimingNormalizer([0, 0, 0], [1, 0, 1])
