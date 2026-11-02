# Cadence local interface

From the repository root, start the model API:

```sh
venv/bin/python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

In another terminal, start the interface:

```sh
npm install --prefix frontend
npm run dev --prefix frontend -- --port 5173
```

Open http://127.0.0.1:5173. Vite proxies `/api` to the local API. The backend
requires `artifacts/v1/encoder_seed123.pt` and `normalizer.json`. These private
local assets are ignored by Git. Profile references and a hold-time preview
are stored in `artifacts/local_profiles.sqlite3`. The demo has no online account
authentication and is intended for loopback/local use.

Choose or randomise a 12–60 character reference phrase and type it five times
to enrol, then make a separate verification attempt. To replace an existing
reference, select it under enrolment or choose Change my reference. Its old
reference stays active until all five replacement attempts have been submitted. Physical-keyboard timing capture retains completed Shift/Backspace
keystrokes, ignores repeats, sorts press times stably, and converts milliseconds
to seconds. Partial key capture is cleared if focus is lost. Pasting is disabled.
The first 70 completed key events are used, matching training truncation.
Browser capture has not been calibrated against production conditions.

Scores are Euclidean distances, not probabilities. Decisions use a backend-owned
threshold of 0.3442268967628479, selected from seed-123 CMU five-sample development
calibration at a target 1% development FAR. This is not a 1% browser FAR guarantee.
The API verifies model, normalizer and calibration hashes at startup and rejects
request-supplied thresholds. No decision controls are exposed to users.
The shared threshold is provisional for custom browser phrases; live calibration
is still needed. The illustrative waveforms in the explanation are not model
outputs. Results show hold-time previews; the actual score uses all three timing
channels and the averaged enrolment embedding.

Typography loads from Google Fonts with readable local fallbacks. Build with
`npm run build --prefix frontend`. Backend tests: `venv/bin/python -m unittest
discover -s tests -v` from the repository root.
