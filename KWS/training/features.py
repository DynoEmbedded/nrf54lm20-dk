"""Canonical MFCC frontend for the KWS model.

This file is the single source of truth for the feature definition. The
firmware reimplements exactly this math in Rust (firmware/src/mfcc.rs); any
change here must be mirrored there. The firmware self-test compares the two
implementations on an embedded PCM test vector.

Spec
----
  sample rate   16000 Hz, mono, i16 PCM scaled to f32 by 1/32768
  clip          16000 samples (1.0 s), zero-padded / truncated at the end
  framing       480-sample frames (30 ms), 320-sample hop (20 ms) -> 49 frames
  window        periodic Hann: w[i] = 0.5 - 0.5*cos(2*pi*i/480)
  FFT           512-point real FFT (frame zero-padded), power = re^2 + im^2
  mel           40 triangular filters, HTK mel scale 2595*log10(1+f/700),
                spanning 20..4000 Hz, evaluated at bin centers k*31.25 Hz
                (bins 0..=128), weights NOT area-normalized
  log           ln(mel_energy + 1e-6)
  DCT           orthonormal DCT-II over the 40 log-mel values, keep c0..c9
  output        49 x 10 f32, frame-major
"""

import numpy as np

SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000
FRAME_LEN = 480
FRAME_HOP = 320
NUM_FRAMES = 49
FFT_LEN = 512
NUM_MEL_BINS = 129  # bins 0..=128 cover 0..4000 Hz at 31.25 Hz/bin
MEL_FILTERS = 40
MEL_LOW_HZ = 20.0
MEL_HIGH_HZ = 4000.0
NUM_MFCC = 10
LOG_FLOOR = 1e-6

LABELS = [
    "silence", "unknown",
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
]
NUM_CLASSES = len(LABELS)


def _hann() -> np.ndarray:
    i = np.arange(FRAME_LEN, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * i / FRAME_LEN)).astype(np.float32)


def _mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_weights() -> np.ndarray:
    """[MEL_FILTERS, NUM_MEL_BINS] triangular filterbank matrix."""
    mel_pts = np.linspace(_mel(MEL_LOW_HZ), _mel(MEL_HIGH_HZ), MEL_FILTERS + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bin_hz = np.arange(NUM_MEL_BINS, dtype=np.float64) * SAMPLE_RATE / FFT_LEN
    w = np.zeros((MEL_FILTERS, NUM_MEL_BINS), dtype=np.float64)
    for m in range(MEL_FILTERS):
        lo, ctr, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        rising = (bin_hz - lo) / (ctr - lo)
        falling = (hi - bin_hz) / (hi - ctr)
        w[m] = np.clip(np.minimum(rising, falling), 0.0, None)
    return w.astype(np.float32)


def _dct_table() -> np.ndarray:
    """[NUM_MFCC, MEL_FILTERS] orthonormal DCT-II basis."""
    j = np.arange(NUM_MFCC, dtype=np.float64)[:, None]
    m = np.arange(MEL_FILTERS, dtype=np.float64)[None, :]
    basis = np.cos(np.pi * j * (2.0 * m + 1.0) / (2.0 * MEL_FILTERS))
    scale = np.full((NUM_MFCC, 1), np.sqrt(2.0 / MEL_FILTERS))
    scale[0, 0] = np.sqrt(1.0 / MEL_FILTERS)
    return (scale * basis).astype(np.float32)


_WINDOW = _hann()
_MELW = _mel_weights()
_DCT = _dct_table()


def clip_to_length(pcm: np.ndarray) -> np.ndarray:
    """Zero-pad or truncate i16/f32 PCM to exactly CLIP_SAMPLES (at the end)."""
    if len(pcm) >= CLIP_SAMPLES:
        return pcm[:CLIP_SAMPLES]
    out = np.zeros(CLIP_SAMPLES, dtype=pcm.dtype)
    out[: len(pcm)] = pcm
    return out


def mfcc_batch(pcm: np.ndarray) -> np.ndarray:
    """[B, CLIP_SAMPLES] f32 in [-1, 1] -> [B, NUM_FRAMES, NUM_MFCC] f32."""
    idx = (
        np.arange(NUM_FRAMES)[:, None] * FRAME_HOP + np.arange(FRAME_LEN)[None, :]
    )
    frames = pcm[:, idx] * _WINDOW  # [B, 49, 480]
    spec = np.fft.rfft(frames, n=FFT_LEN, axis=-1)  # [B, 49, 257]
    power = (spec.real**2 + spec.imag**2)[..., :NUM_MEL_BINS].astype(np.float32)
    mel = power @ _MELW.T  # [B, 49, 40]
    logmel = np.log(mel + LOG_FLOOR)
    return logmel @ _DCT.T  # [B, 49, 10]


def mfcc_single(pcm_i16: np.ndarray) -> np.ndarray:
    """i16 PCM (any length) -> [NUM_FRAMES, NUM_MFCC] f32."""
    x = clip_to_length(pcm_i16).astype(np.float32) / 32768.0
    return mfcc_batch(x[None, :])[0]
