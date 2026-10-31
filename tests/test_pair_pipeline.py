"""Small synthetic shards exercise the same path used by CMU and Aalto."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.pair_generation import load_pair_index, generate_fixed_pairs, write_pairs
from src.data.pair_dataset import PairDataset, FeatureStore


class PairPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Each shard contains one sample per subject, in a different session.
        for shard in range(2):
            features = np.zeros((3, 70, 3), dtype=np.float32)
            features[:, :2] = np.arange(3)[:, None, None] + 10 * shard
            mask = np.zeros((3, 70), dtype=bool)
            mask[:, :2] = True
            np.savez_compressed(self.root / f'aalto_shard_{shard:04}.npz',
                                subject_id=np.array(['a', 'b', 'c']),
                                sample_id=np.array([f'{s}_{shard}' for s in 'abc']),
                                session=np.full(3, shard + 1), seq_len=np.full(3, 2),
                                features=features, mask=mask)
        self.split_path = self.root / 'split.json'
        self.split_path.write_text(json.dumps({'train': ['a', 'b'], 'val': [], 'test': ['c']}))
        self.index = load_pair_index('aalto', 'train', self.root, self.split_path, True)

    def test_fixed_reproducibility_labels_and_sessions(self):
        pairs = generate_fixed_pairs(self.index, 100, seed=9)
        np.testing.assert_array_equal(pairs, generate_fixed_pairs(self.index, 100, seed=9))
        self.assertEqual(sum(pairs[:, 2]), 50)
        path = self.root / 'pairs.npz'
        write_pairs(pairs, path)
        # Use this index as evaluation subjects for the synthetic fixture.
        dataset = PairDataset(self.index, fixed_pairs_path=path, split_name='test', cache_size=1)
        batch = next(iter(DataLoader(dataset, batch_size=8)))
        self.assertEqual(tuple(batch['a']['features'].shape), (8, 70, 3))
        self.assertEqual(batch['label'].dtype, torch.float32)
        for a, b, label in pairs:
            self.assertEqual(a[0] == b[0], bool(label))
            if label:
                self.assertNotEqual(a[-1], b[-1])

    def test_dynamic_epochs_and_cache_rows(self):
        dataset = PairDataset(self.index, pairs_per_epoch=32, cache_size=1)
        before = [(dataset[i]['sample_id_a'], dataset[i]['sample_id_b']) for i in range(32)]
        self.assertEqual(before, [(dataset[i]['sample_id_a'], dataset[i]['sample_id_b']) for i in range(32)])
        dataset.set_epoch(1)
        self.assertNotEqual(before, [(dataset[i]['sample_id_a'], dataset[i]['sample_id_b']) for i in range(32)])
        store = FeatureStore(self.index, cache_size=1)
        for i in range(len(self.index.sample_ids)):
            expected = int(self.index.row_indices[i]) + 10 * int(self.index.shard_indices[i])
            self.assertEqual(store.get(i)['features'][0, 0], expected)
            self.assertLessEqual(len(store._cache), 1)

    def test_memory_matches_cache_across_shards(self):
        cached = PairDataset(self.index, pairs_per_epoch=32, cache_size=1)
        memory = PairDataset(self.index, pairs_per_epoch=32, loading='memory')
        self.assertEqual(memory.store.arrays['features'].shape, (4, 70, 3))
        self.assertGreater(memory.store.nbytes, 0)
        for epoch in (0, 1):
            cached.set_epoch(epoch)
            memory.set_epoch(epoch)
            for i in range(32):
                a, b = cached[i], memory[i]
                self.assertEqual(a['sample_id_a'], b['sample_id_a'])
                self.assertEqual(a['sample_id_b'], b['sample_id_b'])
                self.assertTrue(torch.equal(a['label'], b['label']))
                for side in ('a', 'b'):
                    for key in ('features', 'mask', 'length'):
                        self.assertTrue(torch.equal(a[side][key], b[side][key]))
        # Returned tensors cannot mutate the backing arrays.
        sample = memory.store.get(0)
        sample['features'].fill_(999)
        self.assertFalse((memory.store.get(0)['features'] == 999).any())
        with self.assertRaises(ValueError):
            PairDataset(self.index, loading='unknown')

    def test_rejects_bad_evaluation_pairs(self):
        for rows in ([('a_0', 'c_0', 0)], [('a_0', 'a_0', 1)],
                     [('a_0', 'b_0', 1)], [('a_0', 'b_0', 3)]):
            path = self.root / 'bad.npz'
            write_pairs(np.array(rows, dtype=object), path)
            with self.assertRaises(ValueError):
                PairDataset(self.index, fixed_pairs_path=path, split_name='test')
        with self.assertRaises(ValueError):
            PairDataset(self.index, split_name='test')

    def test_rejects_same_session_genuine_pairs(self):
        # Two distinct samples of one subject, both from the same session.
        self.index.sessions[:] = 1
        path = self.root / 'same_session.npz'
        write_pairs(np.array([('a_0', 'a_1', 1)], dtype=object), path)
        with self.assertRaises(ValueError):
            PairDataset(self.index, fixed_pairs_path=path, split_name='test')

    def test_rejects_duplicate_sample_ids(self):
        path = self.root / 'aalto_shard_0001.npz'
        with np.load(path) as data:
            arrays = dict(data)
        arrays['sample_id'] = np.array(['a_0', 'b_0', 'c_0'])
        np.savez_compressed(path, **arrays)
        with self.assertRaises(ValueError):
            load_pair_index('aalto', 'train', self.root, self.split_path)

    def test_rejects_split_overlap_and_missing_subjects(self):
        self.split_path.write_text(json.dumps({'train': ['a'], 'val': [], 'test': ['a']}))
        with self.assertRaises(ValueError):
            load_pair_index('aalto', 'train', self.root, self.split_path)
        self.split_path.write_text(json.dumps({'train': ['missing'], 'val': [], 'test': []}))
        with self.assertRaises(ValueError):
            load_pair_index('aalto', 'train', self.root, self.split_path)


if __name__ == '__main__':
    unittest.main()
