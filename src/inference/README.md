# Local inference

Run from the repository root. Local v1 assets are in `artifacts/v1` (ignored by Git).

```python
from src.inference.service import InferenceService

service = InferenceService(
    'artifacts/v1/encoder_seed123.pt',
    'artifacts/v1/normalizer.json',
)
# enrol_features: five raw samples [5,70,3]; enrol_mask: bool [5,70].
reference = service.enrol(enrol_features, enrol_mask)
# probe_features: one NEW sample [1,70,3]; probe_mask: bool [1,70].
result = service.verify(probe_features, probe_mask, reference)
# result contains distance; accepted=None until an explicit threshold is supplied.
reference.save('artifacts/v1/local_reference.json')
```

Inputs are H/UD/DD timings in seconds. They must have valid contiguous masks,
nonnegative holds and DD transitions, finite real events, and first-key UD/DD=0.
Negative UD is retained. Do not normalize before calling this service.

The service runs on CPU and loads only encoder weights. Five embeddings are
averaged and L2-normalized, matching the evaluated notebook protocol. References
record model/normalizer hashes; changing either requires re-enrolment.

`verify(..., threshold=...)` returns an experimental decision using distance <=
threshold. There is no default model selection or authentication threshold.
Dataset-calibrated thresholds have not been validated for browser-captured data.
Distance is not a probability or confidence percentage.

This is a Python inference service, not an HTTP API. Browser capture, event-to-
timing conversion, identity storage, and API endpoints remain separate work.
Treat saved reference files as user biometric data and keep them out of Git.
