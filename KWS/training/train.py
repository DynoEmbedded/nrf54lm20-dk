"""Train a DS-CNN-small keyword spotter on the materialized dataset.

Architecture is the MLPerf Tiny KWS shape (DS-CNN), restricted to ops the
Axon NPU compiler supports: Conv2D, DepthwiseConv2D (multiplier 1), pointwise
Conv2D, GlobalAveragePooling (MEAN over HxW), Dense. BN folds into the convs
at conversion. The model ends at logits -- no softmax in the graph, so the
whole network runs on the NPU and the firmware does argmax.

Augmentation per batch (training only): random time shift +-100 ms, background
noise mix, and synthesized silence clips from _background_noise_.
"""

import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from features import CLIP_SAMPLES, NUM_CLASSES, NUM_FRAMES, NUM_MFCC, mfcc_batch

OUT = Path("out")
SILENCE, UNKNOWN = 0, 1
BATCH = 100
EPOCHS = int(os.environ.get("EPOCHS", "30"))
SHIFT_MAX = 1600  # +-100 ms
NOISE_PROB = 0.8
NOISE_VOL = 0.1
SILENCE_FRAC = 0.10
UNKNOWN_FRAC = 0.10
SEED = 4321


def build_model() -> tf.keras.Model:
    inp = tf.keras.layers.Input((NUM_FRAMES, NUM_MFCC, 1))
    x = tf.keras.layers.Conv2D(64, (10, 4), strides=(2, 2), padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    for _ in range(4):
        x = tf.keras.layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(64, (1, 1), use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    logits = tf.keras.layers.Dense(NUM_CLASSES)(x)
    return tf.keras.Model(inp, logits)


class Data:
    def __init__(self):
        self.train_x = np.load(OUT / "train_x.npy", mmap_mode="r")
        self.train_y = np.load(OUT / "train_y.npy")
        self.noise = np.load(OUT / "noise.npy")
        self.spans = np.load(OUT / "noise_spans.npy")
        self.target_idx = np.flatnonzero(self.train_y != UNKNOWN)
        self.unknown_idx = np.flatnonzero(self.train_y == UNKNOWN)
        self.rng = np.random.default_rng(SEED)
        n_t = len(self.target_idx)
        total = int(n_t / (1.0 - SILENCE_FRAC - UNKNOWN_FRAC))
        self.n_unknown = int(total * UNKNOWN_FRAC)
        self.n_silence = int(total * SILENCE_FRAC)
        self.steps = (n_t + self.n_unknown + self.n_silence) // BATCH

    def noise_windows(self, n: int) -> np.ndarray:
        out = np.empty((n, CLIP_SAMPLES), dtype=np.float32)
        for i in range(n):
            s, e = self.spans[self.rng.integers(len(self.spans))]
            start = self.rng.integers(s, e - CLIP_SAMPLES)
            out[i] = self.noise[start : start + CLIP_SAMPLES]
        return out / 32768.0

    def epoch_indices(self) -> np.ndarray:
        unk = self.rng.choice(self.unknown_idx, self.n_unknown, replace=False)
        real = np.concatenate([self.target_idx, unk])
        self.rng.shuffle(real)
        # silence slots are marked with -1 and synthesized in the generator
        sil_at = self.rng.choice(len(real), min(self.n_silence, len(real)), replace=False)
        idx = real.astype(np.int64)
        idx[sil_at] = -1
        return idx

    def batches(self):
        """Yield (features [B,49,10,1] f32, labels [B] u8) forever."""
        while True:
            idx = self.epoch_indices()
            for off in range(0, len(idx) - BATCH + 1, BATCH):
                chunk = idx[off : off + BATCH]
                x = np.zeros((BATCH, CLIP_SAMPLES), dtype=np.float32)
                y = np.zeros(BATCH, dtype=np.uint8)
                real = chunk >= 0
                x[real] = self.train_x[chunk[real]].astype(np.float32) / 32768.0
                y[real] = self.train_y[chunk[real]]
                # time shift the real clips
                shifts = self.rng.integers(-SHIFT_MAX, SHIFT_MAX + 1, real.sum())
                for row, sh in zip(np.flatnonzero(real), shifts):
                    if sh > 0:
                        x[row, sh:] = x[row, : CLIP_SAMPLES - sh]
                        x[row, :sh] = 0.0
                    elif sh < 0:
                        x[row, :sh] = x[row, -sh:]
                        x[row, sh:] = 0.0
                # background noise on real clips, full noise for silence slots
                n_sil = int((~real).sum())
                if n_sil:
                    vols = self.rng.uniform(0.0, 0.3, n_sil)[:, None]
                    x[~real] = self.noise_windows(n_sil) * vols
                    y[~real] = SILENCE
                mix = real & (self.rng.uniform(size=BATCH) < NOISE_PROB)
                n_mix = int(mix.sum())
                if n_mix:
                    vols = self.rng.uniform(0.0, NOISE_VOL, n_mix)[:, None]
                    x[mix] = np.clip(x[mix] + self.noise_windows(n_mix) * vols, -1.0, 1.0)
                yield mfcc_batch(x)[..., None], y


def eval_features(split: str):
    x = np.load(OUT / f"{split}_x.npy", mmap_mode="r")
    y = np.load(OUT / f"{split}_y.npy")
    feats = np.empty((len(x), NUM_FRAMES, NUM_MFCC, 1), dtype=np.float32)
    for off in range(0, len(x), 512):
        chunk = x[off : off + 512].astype(np.float32) / 32768.0
        feats[off : off + len(chunk)] = mfcc_batch(chunk)[..., None]
    return feats, y


def main():
    tf.keras.utils.set_random_seed(SEED)
    data = Data()
    print(f"steps/epoch={data.steps}  epochs={EPOCHS}", flush=True)

    val = eval_features("val")
    np.save(OUT / "val_feats.npy", val[0])

    model = build_model()
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            tf.keras.optimizers.schedules.CosineDecay(1e-3, data.steps * EPOCHS)
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    ds = tf.data.Dataset.from_generator(
        data.batches,
        output_signature=(
            tf.TensorSpec((BATCH, NUM_FRAMES, NUM_MFCC, 1), tf.float32),
            tf.TensorSpec((BATCH,), tf.uint8),
        ),
    ).prefetch(4)

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        str(OUT / "kws_best.keras"), monitor="val_accuracy",
        save_best_only=True, verbose=1,
    )
    model.fit(
        ds, steps_per_epoch=data.steps, epochs=EPOCHS,
        validation_data=val, callbacks=[ckpt], verbose=2,
    )

    best = tf.keras.models.load_model(OUT / "kws_best.keras")
    test_x, test_y = eval_features("test")
    _, acc = best.evaluate(test_x, test_y, verbose=0)
    print(f"float test accuracy: {acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
