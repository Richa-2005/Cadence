"""Convert completed physical-key events (browser performance.now milliseconds)."""
import numpy as np


def timing_sample(events):
    if not 2 <= len(events) <= 500:
        raise ValueError('Capture between 2 and 500 completed keystrokes')
    times = np.asarray([(event.press, event.release) for event in events], dtype=np.float64)
    if not np.isfinite(times).all() or (times < 0).any() or (times[:, 1] < times[:, 0]).any():
        raise ValueError('Invalid key press or release timestamps')
    times = times[np.argsort(times[:, 0], kind='stable')][:70]
    press, release = times[:, 0], times[:, 1]
    x = np.zeros((1, 70, 3), dtype=np.float32)
    n = len(times)
    x[0, :n, 0] = (release - press) / 1000
    x[0, 1:n, 1] = (press[1:] - release[:-1]) / 1000
    x[0, 1:n, 2] = (press[1:] - press[:-1]) / 1000
    mask = np.arange(70)[None, :] < n
    return x, mask
