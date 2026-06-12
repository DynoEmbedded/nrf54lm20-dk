# KWS — voice keyword spotting on the nRF54LM20-DK Axon NPU

End-to-end keyword spotting: the PDM MEMS mic is captured at 16 kHz (PDM-MIC
project), MFCC features are computed on the Cortex-M33, and a DS-CNN runs as
int8 on the Axon NPU (npu project). Detects the 10 Speech Commands keywords:
yes, no, up, down, left, right, on, off, stop, go (+ silence / unknown).

    PDM mic -> 16 kHz PCM -> MFCC (M33) -> 49x10 int8 -> DS-CNN (Axon) -> argmax

## Layout

    training/   pixi project (python 3.11 + tensorflow 2.19, matching the Axon
                compiler container)
        features.py   THE frontend spec; firmware/src/mfcc.rs mirrors it
        prepare.py    materialize Speech Commands v2 into numpy arrays
        train.py      DS-CNN small, augmentation, ~30 epochs
        convert.py    full-int8 TFLite + firmware test vectors
    firmware/   bare-metal Rust, no Zephyr
        src/pdm.rs        PDM driver (ported from PDM-MIC)
        src/mfcc.rs       MFCC frontend (mirrors features.py)
        src/platform.rs   Axon platform shims (from npu)
        src/main.rs       boot self-tests + sliding-window detect loop
        build.rs          links Nordic driver blob via ../../npu/vendor
        generated/        model header + testvec.rs land here (gitignored)

## Workflow

    cd training
    pixi run prepare                       # once: dataset -> out/*.npy
    pixi run train                         # ~40 min CPU; writes out/kws_best.keras
    pixi run convert                       # out/kws.tflite + ../firmware/generated/testvec.rs

    # Compile for the NPU (uses the npu project's containerized Axon compiler;
    # the workspace lands next to the .tflite, the header in firmware/generated/):
    INSTALL_DIR=$PWD/../firmware/generated ../../npu/tools/compile-model.sh out/kws.tflite kws

    cd ../firmware
    cargo run                              # build, flash, stream RTT log

If the Axon compiler reports `interlayer_buffer_needed` / `psum_buffer_needed`
above 65536, bump the sizes in `build.rs` AND `src/main.rs`.

## What the firmware logs (RTT "log" channel)

    selftest nn:   NPU logits vs the int8 TFLite interpreter on a known input
    selftest mfcc: Rust features vs Python features on a known clip (max err)
    selftest e2e:  both stages chained
    DETECT <word> margin=N       smoothed detection (tune DETECT_MARGIN)
    kws: t=..s overruns=N infer=Nus   5 s heartbeat

The three self-tests separate "model is wrong on the NPU" from "frontend
drifted from training" before any live audio is involved. Verified on the DK:
logit-maxdiff=0 (NPU bit-exact vs the int8 TFLite interpreter), mfcc maxerr
< 1e-3, inference 5.8 ms.

Debug-workflow note: killing a session mid-inference can leave the Axon
engine wedged across the soft reset; the next session's first inference then
hangs ~500 ms until the inference watchdog resets the chip. probe-rs catches
that reset and exits -- just run again (or power-cycle the board). See
NOTES.md for the full bring-up story.

## Dependencies on sibling projects

- `../npu/vendor/` must be populated (Nordic Edge AI add-on blobs; see
  npu/README.md) — the firmware links the driver blob from there.
- `firmware/targets/nRF54LM20B.yaml` is the probe-rs target cloned from
  PDM-MIC (probe-rs ships no B variant).

## Custom wake word

Record your own utterances with the PDM-MIC host capture tool (same mic, same
acoustic path), drop them into a new word directory inside the dataset tree,
add the word to LABELS in features.py, and retrain. 150-300 utterances over
several sessions is a workable starting point for a single-speaker wake word;
also record 30+ min of same-mic negatives.
