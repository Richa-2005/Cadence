"""Server-owned demo policy selected from frozen DEVELOPMENT calibration."""
import hashlib
import json
from pathlib import Path


def load_decision_policy(assets: Path, seed='123'):
    manifest = json.loads((assets / 'manifest.json').read_text())
    calibration = json.loads((assets / 'enrolment_calibration_v1.json').read_text())
    model = manifest['models'][seed]
    recorded = calibration['models'][seed]
    if model['source_checkpoint_sha256'] != recorded['checkpoint_sha256']:
        raise ValueError('Model does not match threshold calibration')
    for filename in (model['file'], 'normalizer.json', 'enrolment_calibration_v1.json'):
        digest = hashlib.sha256((assets / filename).read_bytes()).hexdigest()
        if digest != manifest['files'][filename]:
            raise ValueError(f'Inference asset changed: {filename}')
    if hashlib.sha256((assets / 'normalizer.json').read_bytes()).hexdigest() != calibration['normalizer_sha256']:
        raise ValueError('Normalizer does not match calibration')
    point = next(p for p in recorded['domains']['cmu']['far_targets'] if p['target_far'] == .01)
    threshold = float(point['threshold'])
    if not 0 <= threshold <= 2:
        raise ValueError('Invalid calibrated threshold')
    return dict(threshold=threshold, source='CMU development · five-sample enrolment',
                target_development_far=.01, live_browser_calibrated=False)
