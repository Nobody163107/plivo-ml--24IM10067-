"""Audio utilities for the EOT assignment.

These are UTILITIES, not features. Turning them into informative features
(slopes, ratios, statistics over time) is your job.

Causality reminder: for a pause at `pause_start`, you may only touch
audio[0 : pause_start]. Note that `pause_end` is FUTURE information for a
hold pause — using it (e.g., pause duration) in features is a violation.
"""

import numpy as np
import soundfile as sf

FRAME_MS = 25
HOP_MS = 10


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a mono waveform and sample rate from disk."""
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def speech_before(
    x: np.ndarray,
    sr: int,
    pause_start: float,
    window_s: float = 1.5,
) -> np.ndarray:
    """Return the audio segment strictly before the pause."""
    end = int(pause_start * sr)
    start = max(0, end - int(window_s * sr))
    return x[start:end]


def frames(
    x: np.ndarray,
    sr: int,
    frame_ms: int = FRAME_MS,
    hop_ms: int = HOP_MS,
) -> np.ndarray:
    """Split the signal into overlapping analysis frames."""
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32)
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx]


def frame_energy_db(x: np.ndarray, sr: int) -> np.ndarray:
    """Compute short-time energy per frame, in dB."""
    fr = frames(x, sr)
    rms = np.sqrt(np.mean(fr**2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def autocorr_f0(
    frame: np.ndarray,
    sr: int,
    fmin: float = 60.0,
    fmax: float = 400.0,
    voicing_thresh: float = 0.30,
) -> float:
    """Estimate the fundamental frequency of one frame via autocorrelation."""
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo = int(sr / fmax)
    hi = min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < voicing_thresh:
        return 0.0
    return float(sr / lag)


def f0_contour(
    x: np.ndarray,
    sr: int,
    frame_ms: int = 40,
    hop_ms: int = HOP_MS,
) -> np.ndarray:
    """Return a per-frame F0 contour with 0.0 for unvoiced frames."""
    fr = frames(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    return np.array([autocorr_f0(f, sr) for f in fr], dtype=np.float32)
