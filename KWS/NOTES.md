# Notes

## Results (2026-06-11, verified on the DK)

- DS-CNN small (~24k params), 30 epochs, Speech Commands v2, 12 classes.
- Float test accuracy 0.9430; full-int8 0.9403 (quantization cost 0.27%).
- Axon compile: all 10 layers on the NPU, cmd_buffer 29192 B, constants
  22064 B, interlayer needed 13548 B, psum needed 0 (provisioned 16K/4K).
- Firmware release build: ~101 KB text, 44 KB bss, zero undefined symbols.
- Input quant: scale 0.4937, zp 49.
- On hardware: selftest nn logit-maxdiff=0 vs the int8 TFLite interpreter
  (bit-exact), selftest mfcc maxerr < 1e-3, e2e pass; NPU inference 5.8 ms;
  zero steady-state PDM overruns. First inference of the very first session
  after flashing counts one startup overrun -- harmless.

## Hardware bring-up findings (this was the first firmware to run the Axon)

Everything below was found on the DK with the probe attached; none of it is
in the (preliminary) docs.

1. **Top of RAM is not CPU-usable.** The last ~512 B of the 512 KB RAM
   (0x2007FF00..0x20080000) bus-fault (KMU/reserved words; verified by SWD
   reads: 0x2007FE00 reads, 0x2007FF00 FAULTs). cortex-m-rt puts the initial
   SP at the end of the RAM region, so declaring 512K hard-faults on the
   first stack push, before main() -- with a garbage backtrace
   (`__aeabi_dcmpgt` / EXC_RETURN frames). memory.x declares 510K.

2. **Model inference is interrupt-driven, full stop.** nrf_axon_nn_infer.c
   hardcodes NRF_AXON_SYNC_MODE_BLOCKING_EVENT; the AXONS IRQ (86) must be
   delivered. Wiring it needs all three of:
   - cortex-m-rt feature "device" + our own `__INTERRUPTS` table (no PAC
     exposes AXONS; without the table the IRQ vectors into code bytes ->
     INVSTATE UsageFault, CFSR=0x00020000, escalated FORCED HardFault).
   - build.rs emits an empty device.x (link.x INCLUDEs it under "device").
   - NVIC::unmask AFTER nrf_axon_driver_init (Zephyr order: driver_init,
     IRQ_CONNECT, irq_enable). Unmasking before init lets a stale IRQ hit an
     uninitialized driver: it generates a spurious user event and every
     infer_sync thereafter returns one completion EARLY -- 37 us "inferences",
     wrong logits, and the still-running NPU DMA-writes its result into a
     stack frame that later code has reused (this corrupted the MFCC selftest,
     which is pure CPU code, into maxerr ~2.6).
   - The wait shim ALSO polls nrf_axon_handle_interrupt() (benign when idle,
     and the doc explicitly permits direct bare-metal calls) as belt and
     braces against missed events.

3. **Power-cycle the engine around every inference** (reserve/free shims
   carry refcounted votes: ENABLE=1 + driver_power_on / driver_power_off +
   ENABLE=0, exactly Zephyr's onoff design). Reflashing is only a soft reset:
   engine state survives, and a session killed mid-inference leaves the
   engine wedged so the next session hangs in its first infer_sync. With
   per-inference cycling the engine is unpowered ~94% of the time, so kills
   land on a clean engine; verified by kill-and-reflash loops (3/3 clean).

4. **SysTick inference watchdog** (500 ms) as the last resort for the
   remaining window: a wedged first inference triggers sys_reset and the
   boot-time ENABLE cycle recovers. Note probe-rs run vector-catches the
   reset and exits the session -- standalone the device self-heals; under the
   debugger just rerun.

5. RTT log channel is NoBlockSkip on purpose: BlockIfFull freezes the
   firmware once the buffer fills with no host attached. 2048 B holds the
   whole boot/selftest sequence for a late-attaching host.

Non-obvious decisions in the KWS pipeline. The PDM and Axon groundwork is
documented in ../PDM-MIC/NOTES.md and ../npu/NOTES.md; this covers what is new
here.

## One PDM buffer = one MFCC hop

The PDM double buffers are 320 samples (20 ms) — exactly the MFCC hop. Every
STARTED event yields one feature frame: keep the previous 160 samples as tail,
prepend to the new 320 to get the 480-sample (30 ms) analysis frame. No
separate audio ring buffer, no resync logic. The deadline to re-queue the next
EasyDMA pointer is the full 20 ms hop; MFCC (<1 ms) + inference every 5th hop
easily fits. `Stream::overruns` counts missed deadlines (STARTED already
pending on entry) — watch it in the heartbeat log.

## Frontend match strategy

features.py is the spec; mfcc.rs mirrors it operation by operation
(same Hann, same 512-pt real FFT, same un-normalized HTK mel triangles
20..4000 Hz over bins 0..=128, same ln(x+1e-6), same orthonormal DCT-II).
Two deliberate simplifications vs common KWS frontends: no pre-emphasis and no
feature normalization — int8 quantization calibrates the range instead
(representative dataset includes silence/noise clips so the low end is
covered). Training computes features with the SAME numpy code, so there is no
tf.signal-vs-device drift; the only residual difference is f32 vs f64 and the
FFT implementation, checked at boot by the embedded test vector (expect max
err ~1e-3 on MFCC scale).

microfft packing gotcha: `rfft_512` packs the real Nyquist bin into
`spec[0].im` (numpy keeps a separate bin 257). Zero it before computing power;
the mel range stops at bin 128 so only DC would be corrupted.

## Why the model ends at logits (no softmax)

The Axon compiler runs SOFTMAX on the CPU (`cpu_op_codes_list {25:..}`) or
skips it. Ending the Keras graph at Dense(12) keeps the entire network on the
NPU and the firmware argmaxes int8 logits directly. Loss uses
`from_logits=True`. Detection thresholds (DETECT_MARGIN) are therefore in
int8-logit units, not probabilities.

## Quantization formula parity

convert.py quantizes with `round(x/scale)+zp` (numpy round = half-to-even);
the firmware uses the descriptor's multiply form
`round(x * quant_mult/2^quant_round)+zp` (libm round = half-away). The two
differ only on exact .5 boundaries — irrelevant in practice and covered by the
self-test tolerance. The npu hello_axon example TRUNCATED instead of rounding
(fine for a demo, wrong for features): rounding matters here because MFCC
values cluster near zero where truncation bias is visible.

## Stub-driven type-check of cfg'd-out code

`run()`/`selftest()` only compile when `has_model`/`has_testvec` cfgs are on,
i.e. after training + model compile. To avoid discovering Rust errors after a
30-minute training run, the build was smoke-tested once with a stub model
header (`const nrf_axon_nn_compiled_model_s model_stub = {0};`) and a zeroed
testvec.rs. Zero-init C descriptor links fine (validate() would fail at
runtime, but only the type-check matters).

## Training setup choices

- pixi env pins tensorflow==2.19.0 == the Axon compiler container's pin, so
  the converter that writes kws.tflite matches what Nordic tests.
- prepare.py materializes all 105k clips into flat i16 .npy (2.9 GB) once;
  train.py mmaps them and does augmentation (shift +-100 ms, background noise
  mix p=0.8 vol U(0,0.1), synthesized silence at vol U(0,0.3)) per batch in
  vectorized numpy, then MFCCs the whole batch in one rfft call. ~75 s/epoch
  on 8 CPU cores; no GPU needed at this model size (~24k params).
- Eval sets get fixed silence/unknown at 10%/10% (seeded), so val/test
  accuracy is comparable across runs.
- Labels: ["silence","unknown", then the 10 words]. Nordic's sample yaml
  orders them differently — irrelevant as long as training and firmware agree
  (firmware LABELS const + generated testvec.rs both come from features.py's
  list).
