"""Local timing inference. Thresholds must be supplied explicitly by callers."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.data.normalization import TimingNormalizer
from src.models.encoder import KeystrokeEncoder


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class EnrolledReference:
    embedding: list[float]
    model_sha256: str
    normalizer_sha256: str
    samples: int = 5

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2, allow_nan=False) + '\n')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class InferenceService:
    """Accept raw seconds [N,70,3] with bool mask [N,70], never normalized input.

    No global model choice, threshold, persistent user database, or browser-event
    conversion is implied. References are tied to exact model/preprocessing files.
    """

    def __init__(self, model_path, normalizer_path):
        self.model_sha256 = fingerprint(model_path)
        self.normalizer_sha256 = fingerprint(normalizer_path)
        self.normalizer = TimingNormalizer.load(normalizer_path)
        self.encoder = KeystrokeEncoder()
        self.encoder.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        self.encoder.eval()

    def embed(self, features, mask):
        features, mask = np.asarray(features), np.asarray(mask)
        if features.ndim != 3 or features.shape[1:] != (70, 3) or not len(features):
            raise ValueError('Expected nonempty raw features [N,70,3] in seconds')
        if mask.shape != features.shape[:2] or mask.dtype != np.bool_:
            raise ValueError('Expected boolean mask [N,70]')
        if not np.allclose(features[:, 0, 1:], 0, atol=0, rtol=0):
            raise ValueError('First-key UD/DD must be zero')
        x = self.normalizer.transform(features, mask)
        lengths = torch.from_numpy(mask.sum(axis=1).astype(np.int64))
        with torch.inference_mode():
            embeddings = self.encoder(torch.from_numpy(x), lengths).numpy().copy()
        if not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5):
            raise ValueError('Invalid encoder output')
        return embeddings

    def enrol(self, features, mask):
        if len(features) != 5:
            raise ValueError('V1 enrolment requires exactly five separate typing attempts')
        embeddings = self.embed(features, mask)
        reference = embeddings.mean(axis=0)
        norm = np.linalg.norm(reference)
        if norm < 1e-8:
            raise ValueError('Degenerate enrolment reference')
        return EnrolledReference((reference / norm).tolist(), self.model_sha256,
                                 self.normalizer_sha256)

    def verify(self, features, mask, reference, *, threshold=None):
        if (reference.model_sha256 != self.model_sha256
                or reference.normalizer_sha256 != self.normalizer_sha256):
            raise ValueError('Reference belongs to a different model or normalizer; re-enrol')
        vector = np.asarray(reference.embedding, dtype=np.float32)
        if (reference.samples != 5 or vector.shape != (64,) or not np.isfinite(vector).all()
                or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5)):
            raise ValueError('Invalid enrolled reference')
        if len(features) != 1:
            raise ValueError('Verification requires one typing attempt')
        if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 2):
            raise ValueError('Threshold must be finite and in [0,2]')
        distance = float(np.linalg.norm(self.embed(features, mask)[0] - vector))
        return dict(distance=distance, threshold=threshold,
                    accepted=None if threshold is None else distance <= threshold)
