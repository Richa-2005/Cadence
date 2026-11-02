"""Local demo API. Start: venv/bin/python -m uvicorn src.api.app:app --port 8000."""
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import threading

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from src.api.capture import timing_sample
from src.api.decision_policy import load_decision_policy
from src.inference.service import InferenceService, EnrolledReference

ROOT = Path(__file__).resolve().parents[2]
PHRASE = 'the rhythm is yours'
LOCK = threading.Lock()


class Event(BaseModel):
    press: float = Field(ge=0)
    release: float = Field(ge=0)


class Attempt(BaseModel):
    text: str = Field(max_length=100)
    events: list[Event] = Field(min_length=2, max_length=500)


class EnrolRequest(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    phrase: str = Field(default=PHRASE, min_length=12, max_length=60)
    replace_profile_id: int | None = None
    attempts: list[Attempt] = Field(min_length=5, max_length=5)


class VerifyRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    profile_id: int
    attempt: Attempt


@contextmanager
def db():
    connection = sqlite3.connect(ROOT / 'artifacts/local_profiles.sqlite3')
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def sample(attempt, phrase=PHRASE):
    if attempt.text != phrase:
        raise ValueError('Type the displayed phrase exactly')
    if len(attempt.events) < len(phrase):
        raise ValueError('Incomplete keystroke capture; please retry')
    return timing_sample(attempt.events)


@asynccontextmanager
async def lifespan(app):
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    with db() as connection:
        connection.execute('CREATE TABLE IF NOT EXISTS profiles (id INTEGER PRIMARY KEY, name TEXT NOT NULL, reference TEXT NOT NULL, rhythm TEXT NOT NULL)')
        columns = {row[1] for row in connection.execute('PRAGMA table_info(profiles)')}
        if 'phrase' not in columns:
            connection.execute("ALTER TABLE profiles ADD COLUMN phrase TEXT NOT NULL DEFAULT 'the rhythm is yours'")
    try:
        app.state.policy = load_decision_policy(ROOT / 'artifacts/v1')
        app.state.service = InferenceService(ROOT / 'artifacts/v1/encoder_seed123.pt', ROOT / 'artifacts/v1/normalizer.json')
    except (OSError, RuntimeError, ValueError):
        app.state.service = None
    yield


app = FastAPI(title='Cadence local demo', lifespan=lifespan)


def service():
    if app.state.service is None:
        raise HTTPException(503, 'Model assets missing. Restore artifacts/v1 before enrolment.')
    return app.state.service


@app.get('/api/health')
def health():
    return dict(ready=app.state.service is not None, phrase=PHRASE, model='v1 · seed 123', local=True)


@app.get('/api/profiles')
def profiles():
    with db() as connection:
        rows = connection.execute('SELECT id, name, phrase FROM profiles ORDER BY id DESC').fetchall()
    return [dict(id=row[0], name=row[1], phrase=row[2]) for row in rows]


@app.post('/api/enrol')
def enrol(request: EnrolRequest):
    try:
        if not request.name.strip():
            raise ValueError('Enter a profile name')
        if not request.phrase.isascii() or request.phrase.strip() != request.phrase or not request.phrase.strip():
            raise ValueError('Use a phrase of 12–60 characters without leading or trailing spaces')
        samples = [sample(attempt, request.phrase) for attempt in request.attempts]
        x, mask = np.concatenate([a for a, _ in samples]), np.concatenate([b for _, b in samples])
        with LOCK:
            reference = service().enrol(x, mask)
        rhythm = x[0, mask[0], 0].tolist()
        with db() as connection:
            if request.replace_profile_id is None:
                cursor = connection.execute('INSERT INTO profiles (name, reference, rhythm, phrase) VALUES (?, ?, ?, ?)',
                                            (request.name.strip(), json.dumps(asdict(reference)), json.dumps(rhythm), request.phrase))
                profile_id = cursor.lastrowid
            else:
                profile_id = request.replace_profile_id
                cursor = connection.execute('UPDATE profiles SET name=?, reference=?, rhythm=?, phrase=? WHERE id=?',
                                            (request.name.strip(), json.dumps(asdict(reference)), json.dumps(rhythm), request.phrase, profile_id))
                if cursor.rowcount != 1:
                    raise HTTPException(404, 'Profile not found')
        return dict(id=profile_id, name=request.name.strip(), phrase=request.phrase)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post('/api/verify')
def verify(request: VerifyRequest):
    with db() as connection:
        row = connection.execute('SELECT reference, rhythm, phrase FROM profiles WHERE id=?', (request.profile_id,)).fetchone()
    if row is None:
        raise HTTPException(404, 'Profile not found')
    try:
        x, mask = sample(request.attempt, row[2])
        with LOCK:
            result = service().verify(x, mask, EnrolledReference(**json.loads(row[0])), threshold=app.state.policy['threshold'])
        return dict(**result, rhythm=x[0, mask[0], 0].tolist(), reference_rhythm=json.loads(row[1]),
                    decision_policy=app.state.policy,
                    note='Uses a frozen CMU development threshold; not calibrated for live browser typing.')
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
