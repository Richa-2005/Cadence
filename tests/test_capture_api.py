import importlib
from contextlib import contextmanager
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest

import numpy as np
from fastapi.testclient import TestClient
from src.api.capture import timing_sample
from src.api.app import Event, PHRASE


class CaptureTests(unittest.TestCase):
    def test_sorting_overlap_and_units(self):
        x, mask = timing_sample([Event(press=150,release=190),Event(press=100,release=168)])
        np.testing.assert_allclose(x[0,:2], [[.068,0,0],[.04,-.018,.05]], atol=1e-7)
        self.assertEqual(mask.sum(),2)
        self.assertTrue((x[~mask]==0).all())
        with self.assertRaises(ValueError):
            timing_sample([Event(press=10,release=5),Event(press=20,release=30)])

    def test_real_local_api_enrolment_and_verification(self):
        module = importlib.import_module('src.api.app')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'profiles.sqlite3'
            @contextmanager
            def temporary_db():
                connection = sqlite3.connect(path)
                try:
                    with connection:
                        yield connection
                finally:
                    connection.close()
            with patch.object(module, 'db', temporary_db):
                with TestClient(module.app) as client:
                    self.assertTrue(client.get('/api/health').json()['ready'])
                    attempt=dict(text=PHRASE,events=[dict(press=i*120,release=i*120+70) for i in range(len(PHRASE))])
                    response=client.post('/api/enrol',json=dict(name='Local test',attempts=[attempt]*5))
                    self.assertEqual(response.status_code,200,response.text)
                    profile=response.json()
                    result=client.post('/api/verify',json=dict(profile_id=profile['id'],attempt=attempt))
                    self.assertEqual(result.status_code,200,result.text)
                    self.assertTrue(result.json()['accepted'])
                    self.assertAlmostEqual(result.json()['threshold'], 0.3442268967628479)
                    self.assertFalse(result.json()['decision_policy']['live_browser_calibrated'])
                    self.assertLess(result.json()['distance'],1e-5)
                    decision=client.post('/api/verify',json=dict(profile_id=profile['id'],attempt=attempt,threshold=.5))
                    self.assertEqual(decision.status_code,422)
                    bad=client.post('/api/verify',json=dict(profile_id=profile['id'],attempt={**attempt,'text':'wrong'}))
                    self.assertEqual(bad.status_code,422)
                    self.assertEqual(len(client.get('/api/profiles').json()),1)
                    phrase='every keystroke tells a story'
                    updated_attempt=dict(text=phrase,events=[dict(press=i*120,release=i*120+70) for i in range(len(phrase))])
                    updated=client.post('/api/enrol',json=dict(name='Local test',phrase=phrase,replace_profile_id=profile['id'],attempts=[updated_attempt]*5))
                    self.assertEqual(updated.status_code,200,updated.text)
                    self.assertEqual(updated.json()['id'],profile['id'])
                    records=client.get('/api/profiles').json()
                    self.assertEqual(len(records),1)
                    self.assertEqual(records[0]['phrase'],phrase)
                    self.assertEqual(client.post('/api/verify',json=dict(profile_id=profile['id'],attempt=attempt)).status_code,422)
                    self.assertEqual(client.post('/api/verify',json=dict(profile_id=profile['id'],attempt=updated_attempt)).status_code,200)
                    missing=client.post('/api/enrol',json=dict(name='Local test',phrase=phrase,replace_profile_id=999,attempts=[updated_attempt]*5))
                    self.assertEqual(missing.status_code,404)

