"""Generate a small int8-quantized sine-regression TFLite model.

Run inside the Axon Compiler container (has TensorFlow 2.19):
  podman run --rm --entrypoint python3 -v <models>:/work nrf-axon-compiler /work/gen_sine_model.py

Writes /work/sine.tflite (int8 in/out), the same shape of problem as Nordic's
hello_axon example: input x in [0, 2pi] -> output ~sin(x).
"""
import numpy as np
import tensorflow as tf

rng = np.random.default_rng(0)
x = rng.uniform(0.0, 2.0 * np.pi, 4000).astype(np.float32)
y = np.sin(x).astype(np.float32)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input((1,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)
model.compile(optimizer="adam", loss="mse")
model.fit(x, y, epochs=60, batch_size=64, verbose=0)


def representative_dataset():
    for v in x[:500]:
        yield [np.array([[v]], dtype=np.float32)]


conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representative_dataset
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8

tflite_bytes = conv.convert()
with open("/work/sine.tflite", "wb") as f:
    f.write(tflite_bytes)
print(f"wrote /work/sine.tflite ({len(tflite_bytes)} bytes)")
