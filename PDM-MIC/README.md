# nRF54LM20-DK PDM MEMS microphone -> host (bare-metal Rust)

Reads a PDM MEMS microphone with the nRF54LM20's PDM peripheral and streams the
decimated PCM audio to a computer over RTT (via the DK's onboard debug probe), so
you can record a `.wav` and confirm the mic works. No Zephyr, no HAL crate - direct
register access.

## Wiring

The breakout exposes CLK, DAT, 3V3, GND. Connect:

| Mic pin | DK pin    | Notes                                   |
|---------|-----------|-----------------------------------------|
| CLK     | **P1.23** | PDM clock out -- must be a "clock pin"   |
| DAT     | **P1.24** | PDM data in                             |
| 3V3     | VDD header| 3.3 V                                   |
| GND     | GND header| ground                                  |

These are the exact pins Nordic uses for PDM on the nRF54LM20-DK (its dmic test:
`PDM_CLK=P1.23`, `PDM_DIN=P1.24`). **Pin choice is not free for the clock**: on the
nRF54L family, peripheral clock signals must use a dedicated "clock pin". P1.23 is
one; an ordinary GPIO such as P1.02 will not output the PDM clock, so the mic is
never clocked and the data line stays constant (decimates to digital silence).

Pins are set in [`src/main.rs`](src/main.rs) (`CLK`/`DIN`). They are assumed to be
on port 1 (the GPIO config/probe in [`src/pdm.rs`](src/pdm.rs) uses `P1_BASE`); keep
CLK on a port-1 clock pin if you change them.

Most PDM breakouts that expose only CLK/DAT/3V3/GND tie the L/R select internally
to GND (left channel). We capture mono-left. If the recording is silent or sounds
wrong, flip the clock edge - see NOTES.

## Build options

Cargo features (combine freely):

| Feature        | Effect                                                        |
|----------------|---------------------------------------------------------------|
| `gain-10db`    | +10 dB digital gain (default is 0 dB)                         |
| `gain-20db`    | +20 dB digital gain (hardware max; wins over `gain-10db`)     |
| `edge-falling` | latch data on the falling PDM_CLK edge (default is rising)    |

    cargo build --release --features gain-10db

## Audio format

16 kHz, mono, signed 16-bit PCM. The PDM hardware decimates the mic's 1-bit stream
in-hardware: `PDM_CLK = 32 MHz / 25 = 1.28 MHz`, decimation ratio 80 -> 16 kHz.

## Toolchain

- Rust target `thumbv8m.main-none-eabihf` (`rustup target add thumbv8m.main-none-eabihf`)
- `probe-rs` (the DK's onboard J-Link is the probe)
- For the host capture tool: a libusb dev package (`libusb-1.0-0-dev`) and
  `libudev-dev` on Linux.

> **Chip target:** probe-rs ships no `nRF54LM20B` target, so this repo provides one
> in [`targets/nRF54LM20B.yaml`](targets/nRF54LM20B.yaml) (cloned from probe-rs's
> `nRF54LM20A` definition, which is what actually flashes this DK). The runner and
> host tool reference it automatically; see NOTES.md.

## Record audio (one command)

The host tool flashes the firmware, resets, attaches RTT, and records until Ctrl-C:

```
cargo build --release                                  # build firmware
cd host
cargo run --release -- \
    ../target/thumbv8m.main-none-eabihf/release/nrf54lm20-pdm-mic \
    out.wav
# ...speak into the mic..., then Ctrl-C
```

`out.wav` is a normal 16 kHz mono WAV - play it in anything.

## Quick smoke test (no recording)

`cargo run` flashes and runs the firmware and prints the `log` RTT channel:

```
cargo run --release
# -> "pdm-mic: starting ..." then "pdm-mic: streaming"
```

(The `pcm` channel will show as binary garbage in this view - use the host tool to
record it properly.)

## CLI-only fallback (no custom tool)

If you would rather not build the host crate, capture the raw channel with the
probe-rs CLI and convert it:

```
probe-rs attach --chip nRF54LM20B --chip-description-path targets/nRF54LM20B.yaml \
    target/thumbv8m.main-none-eabihf/release/nrf54lm20-pdm-mic \
    --target-output-file rtt:pcm=audio.raw
# Ctrl-C to stop, then:
ffmpeg -f s16le -ar 16000 -ac 1 -i audio.raw out.wav
#   (or: ffplay -f s16le -ar 16000 -ac 1 audio.raw)
```

## Layout

    src/main.rs   entry, RTT channels, pin map, capture loop
    src/pdm.rs    the PDM peripheral driver (registers, clocking, double buffering)
    host/         host capture tool (probe-rs + hound -> .wav)
    memory.x      nRF54LM20A memory map
    NOTES.md      register/clocking derivation and gotchas


`cargo run --release -- ../target/thumbv8m.main-none-eabihf/release/nrf54lm20-pdm-mic out.wav`


