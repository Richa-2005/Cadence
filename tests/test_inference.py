import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from src.models.encoder import KeystrokeEncoder
from src.data.normalization import TimingNormalizer
from src.inference.service import InferenceService, EnrolledReference


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        torch.manual_seed(42)
        torch.save(KeystrokeEncoder().state_dict(), self.root / 'weights.pt')
        TimingNormalizer([.5, .5, .5], [1, 1, 1]).save(self.root / 'norm.json')
        self.service = InferenceService(self.root / 'weights.pt', self.root / 'norm.json')
        rng = np.random.default_rng(42)
        self.x = rng.uniform(.01, .2, (6, 70, 3)).astype(np.float32)
        self.mask = np.tile(np.arange(70) < 11, (6, 1))
        self.x[:, 11:] = 0
        self.x[:, 0, 1:] = 0

    def test_notebook_equivalence_and_reference_roundtrip(self):
        vectors = self.service.embed(self.x, self.mask)
        expected = vectors[:5].mean(axis=0)
        expected /= np.linalg.norm(expected)
        reference = self.service.enrol(self.x[:5], self.mask[:5])
        np.testing.assert_allclose(reference.embedding, expected, atol=1e-7)
        reference.save(self.root / 'reference.json')
        loaded = EnrolledReference.load(self.root / 'reference.json')
        result = self.service.verify(self.x[5:], self.mask[5:], loaded)
        self.assertIsNone(result['accepted'])
        self.assertAlmostEqual(result['distance'], float(np.linalg.norm(vectors[5] - expected)), places=6)
        self.assertTrue(self.service.verify(self.x[5:], self.mask[5:], loaded, threshold=2)['accepted'])
        self.assertFalse(self.service.verify(self.x[5:], self.mask[5:], loaded, threshold=0)['accepted'])

    def test_rejects_invalid_input_and_incompatible_reference(self):
        reference = self.service.enrol(self.x[:5], self.mask[:5])
        other = EnrolledReference(reference.embedding, 'different', reference.normalizer_sha256)
        with self.assertRaises(ValueError):
            self.service.verify(self.x[5:], self.mask[5:], other)
        for threshold in [float('nan'), -1, 3]:
            with self.assertRaises(ValueError):
                self.service.verify(self.x[5:], self.mask[5:], reference, threshold=threshold)
        with self.assertRaises(ValueError):
            self.service.enrol(self.x[:4], self.mask[:4])
        with self.assertRaises(ValueError):
            self.service.embed(self.x, self.mask.astype(int))
        invalid = self.x.copy()
        invalid[0, 1, 0] = -1
        with self.assertRaises(ValueError):
            self.service.embed(invalid, self.mask)
