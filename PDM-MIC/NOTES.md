# Notes

Concise record of the non-obvious decisions behind this PDM capture firmware.

## Register source

There is no public PDM register doc and no PAC/HAL crate for this part. Everything
was lifted from the Nordic MDK shipped with nrfx (found locally under
`.../nrfx/bsp/stable/mdk/nrf54lm20{a,b}_application.svd` and `*_types.h`):

- PDM20 secure base `0x500D0000` (PDM21 at `0x500D1000`; NS aliases at `0x400Dxxxx`).
- Key offsets: ENABLE `0x500`, MODE `0x508`, GAINL/R `0x518/0x51C`, RATIO `0x520`,
  PSEL.CLK/DIN `0x540/0x544`, CLKSELECT `0x54C`, SAMPLE.PTR/MAXCNT `0x560/0x564`,
  PRESCALER `0x580`. Tasks START/STOP `0x000/0x004`, events STARTED/STOPPED/END
  `0x100/0x104/0x108`.

## This is PDM v2, not the nRF52 PDM

Two things differ from the well-known nRF52 PDM and will bite if you copy old code:

1. **No PDMCLKCTRL.** Clock is `CLKSELECT` (source) + `PRESCALER` (divisor):
   `PDM_CLK = base / PRESCALER`, `sample_rate = PDM_CLK / RATIO`.
   Sources: `PCLK32M` (32 MHz, SRC=0) or `ACLK` (24 MHz, SRC=1). We use PCLK32M.
   `32 MHz / 25 = 1.28 MHz`, RATIO=80 -> 16000 Hz exactly. Prescaler range is 4..126.
   1.28 MHz is a standard PDM mic clock and sits mid-range (mics want ~1-3.2 MHz).

2. **SAMPLE.MAXCNT is a BYTE count**, not a sample count (the SVD says "number of
   bytes", and nrf_pdm.h multiplies by `sizeof(int16_t)` under
   `DMA_BUFFER_UNIFIED_BYTE_ACCESS`). So for N 16-bit samples, MAXCNT = `N * 2`.

## PSEL encoding

`(port << 5) | pin`, with the CONNECT bit (bit 31) = 0 meaning connected. PIN is
bits [4:0], PORT bits [6:5]. Reset value is `0xFFFFFFFF` (disconnected). Verified
against the board's `NRF_PSEL(UART_TX, 1, 16) == 48`.

## GPIO register map moved

On nRF54 GPIO, `PIN_CNF[n]` is at `0x080 + 4*n` (not `0x700` like nRF52). DIRSET
`0x014`, OUTCLR `0x008`. We pre-configure CLK as output-low and DIN as input before
enabling PDM (mirrors what nrfx does).

## Chip target: providing nRF54LM20B ourselves

The DK silicon is the nRF54LM20B, but probe-rs (0.31, and upstream master) ships
**no** `nRF54LM20B` target - only `nRF54LM20A`. Telling detail: that A target's
FICR auto-detect id is `0x41414142` (the trailing byte `0x42` is ASCII 'B'), and
its flash algorithm + memory map are exactly what programs this board. In other
words the single "A" entry is really the DK's flasher.

So `--chip nRF54LM20B` would error "not found". Rather than fall back to the A
name, we ship our own target description at `targets/nRF54LM20B.yaml`: a verbatim
clone of probe-rs's A entry (same flash algorithm blob from
embassy-rs/nrf54l-flash-algo, same 2036K NVM + 512K RAM map) renamed to
`nRF54LM20B`. The A/B peripheral maps are identical, so this is correct.

Wired in two places:
- CLI / `cargo run`: the runner adds `--chip nRF54LM20B
  --chip-description-path targets/nRF54LM20B.yaml`.
- host tool: `Registry::from_builtin_families()` +
  `add_target_family_from_yaml(include_str!("../../targets/nRF54LM20B.yaml"))`,
  then `Probe::attach_with_registry(...)`.

If a future probe-rs ships a real B target, drop the YAML and the
`--chip-description-path` / registry plumbing.

## Double buffering (polled, no IRQ)

Classic Nordic ping-pong: set SAMPLE.PTR to buffer A, START, wait STARTED. Then on
each loop: set PTR to the *other* buffer, wait the next STARTED (hardware has
latched it and swapped), and the just-finished buffer is ready to ship. Setting the
next PTR happens well before the current 16 ms buffer fills, so there is no race.
First buffer may contain filter-settling transient - fine for a test capture.

## Host transport

RTT over the onboard probe, two up-channels: `log` (text status) and `pcm` (raw
LE i16). The `host/` tool uses the probe-rs *library* (not the CLI) so binary audio
is byte-exact, and writes a real `.wav` with `hound`. The CLI
`--target-output-file rtt:pcm=...` route is documented as a fallback. `pcm` uses
BlockIfFull with an 8 KB buffer; SWD RTT drains far faster than the 32 KB/s stream,
so the recording stays gapless.

## Host crate vs firmware target

The repo `.cargo/config.toml` pins `build.target` to thumbv8m for the firmware.
That config is inherited by `host/`, so `host/.cargo/config.toml` pins the host
triple back to `x86_64-unknown-linux-gnu` to override it. `host/` also has an empty
`[workspace]` so it is not pulled into the firmware's workspace.

## Pin choice: PDM_CLK needs a dedicated "clock pin"

This cost real debug time. On the nRF54L family the GPIO ports map to power
domains (P0 always-on, P1 PERI, P2 MCU/high-speed), and peripherals must use pins
in their own domain. PDM20 is PERI, so CLK/DIN must be on P1 (P1 max signal rate
8 MHz, fine for a 1.28 MHz clock). But there is a finer rule: **peripheral clock
signals (SPI SCK, TWI SCL, PDM_CLK, ...) must use a dedicated "clock pin"** -
marked with a cross in the datasheet pin-assignment table. An ordinary GPIO does
not output the clock.

Symptom of getting this wrong: the capture runs at the exact sample rate (the
internal decimator clock is independent of the pin) and EasyDMA fills the buffer,
but every sample is exactly 0 - because the mic never receives a clock, holds its
data line constant, and a CIC decimator turns a constant input into 0.

Board-validated pins, from Nordic's own dmic test for the nRF54LM20-DK
(`tests/drivers/audio/dmic_api/boards/nrf54lm20dk_..._common.dtsi`):
`PDM_CLK = P1.23` (a clock pin), `PDM_DIN = P1.24`. We use those. P1.12/P1.13 are
the nRF54L15 reference pins (different package/pinout) - do not assume they carry
over. The DAT-activity probe in `src/pdm.rs` (`probe_pin_activity`) reports whether
the line actually toggles, which distinguishes "wrong/!clock pin" from a wiring
fault; it stays in as a startup self-check since this is a mic test tool.

Confirmed working on hardware with P1.23/P1.24: ~-90 dBFS noise floor in silence,
clear signal bursts, zero-centered. Rising edge (the default) was correct for this
mic; the falling-edge build was never needed once the clock pin was right.

## Clock source

We select PCLK32M and assume it is available without manual gating (the nRF54L
peripheral clock framework normally provides it; unlike nRF52, there is no explicit
HFCLK START here). If `EVENTS_STARTED` never fires on hardware, the first thing to
check is whether the 32 MHz peripheral clock needs to be requested (or switch
`CLKSELECT` to ACLK / verify via the CLOCK peripheral).

## Edge / channel caveat

`MODE.EDGE` defaults here to LeftFalling (mono sampled on the falling PDM_CLK edge).
A mic with L/R-select tied to GND drives data for the left slot; if your recording
is silent or noisy, flip `MODE_EDGE_LEFTFALLING` to LeftRising in `src/pdm.rs`.
