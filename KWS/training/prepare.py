"""Materialize Speech Commands v2 into flat numpy arrays for fast training.

Reads data/speech_commands_v0.02/, applies the official validation/testing
list split, and writes to out/:
  train_x.npy  i16 [N, 16000]   all train-split clips (targets + unknown pool)
  train_y.npy  u8  [N]          label index (unknown words -> 1)
  val_x.npy / val_y.npy         fixed eval set incl. silence + unknown
  test_x.npy / test_y.npy       fixed eval set incl. silence + unknown
  noise.npy    i16 [M]          concatenated _background_noise_ audio
  noise_spans.npy i64 [K, 2]    (start, end) of each source noise file
"""

import sys
import wave
from pathlib import Path

import numpy as np

from features import CLIP_SAMPLES, LABELS, clip_to_length

DATA = Path("data/speech_commands_v0.02")
OUT = Path("out")
TARGET_WORDS = LABELS[2:]
SILENCE, UNKNOWN = 0, 1
EVAL_SILENCE_FRAC = 0.10
EVAL_UNKNOWN_FRAC = 0.10
SEED = 1234


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def main():
    if not DATA.is_dir():
        sys.exit(f"dataset not found at {DATA}")
    OUT.mkdir(exist_ok=True)

    val_list = set((DATA / "validation_list.txt").read_text().split())
    test_list = set((DATA / "testing_list.txt").read_text().split())

    splits = {"train": [], "val": [], "test": []}  # (relpath, label)
    for word_dir in sorted(DATA.iterdir()):
        if not word_dir.is_dir() or word_dir.name == "_background_noise_":
            continue
        word = word_dir.name
        label = LABELS.index(word) if word in TARGET_WORDS else UNKNOWN
        for f in sorted(word_dir.glob("*.wav")):
            rel = f"{word}/{f.name}"
            split = "val" if rel in val_list else "test" if rel in test_list else "train"
            splits[split].append((f, label))

    noise_chunks = [read_wav(f) for f in sorted((DATA / "_background_noise_").glob("*.wav"))]
    spans, pos = [], 0
    for c in noise_chunks:
        spans.append((pos, pos + len(c)))
        pos += len(c)
    noise = np.concatenate(noise_chunks)
    np.save(OUT / "noise.npy", noise)
    np.save(OUT / "noise_spans.npy", np.array(spans, dtype=np.int64))

    rng = np.random.default_rng(SEED)

    def noise_window(volume: float) -> np.ndarray:
        s, e = spans[rng.integers(len(spans))]
        start = rng.integers(s, e - CLIP_SAMPLES)
        return (noise[start : start + CLIP_SAMPLES].astype(np.float32) * volume).astype(np.int16)

    for split, files in splits.items():
        n_real = len(files)
        print(f"{split}: {n_real} clips", flush=True)
        x = np.zeros((n_real, CLIP_SAMPLES), dtype=np.int16)
        y = np.zeros(n_real, dtype=np.uint8)
        for i, (path, label) in enumerate(files):
            x[i] = clip_to_length(read_wav(path))
            y[i] = label
            if i % 10000 == 0:
                print(f"  {i}/{n_real}", flush=True)

        if split in ("val", "test"):
            # Fixed-eval convention: silence and unknown each ~10% of the set.
            # The word dirs over-supply unknowns, so subsample them.
            target_idx = np.flatnonzero(y != UNKNOWN)
            unk_idx = np.flatnonzero(y == UNKNOWN)
            n_keep = int(len(target_idx) / (1 - EVAL_SILENCE_FRAC - EVAL_UNKNOWN_FRAC)
                         * EVAL_UNKNOWN_FRAC)
            unk_keep = rng.choice(unk_idx, size=min(n_keep, len(unk_idx)), replace=False)
            n_sil = n_keep
            sil_x = np.stack(
                [noise_window(rng.uniform(0.0, 0.3)) for _ in range(n_sil)]
            )
            x = np.concatenate([x[target_idx], x[unk_keep], sil_x])
            y = np.concatenate(
                [y[target_idx], y[unk_keep], np.full(n_sil, SILENCE, dtype=np.uint8)]
            )

        np.save(OUT / f"{split}_x.npy", x)
        np.save(OUT / f"{split}_y.npy", y)
        counts = np.bincount(y, minlength=len(LABELS))
        print(f"  -> {len(y)} clips; per-class {counts.tolist()}", flush=True)


if __name__ == "__main__":
    main()
