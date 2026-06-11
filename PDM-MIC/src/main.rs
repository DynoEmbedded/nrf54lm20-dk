#![no_std]
#![no_main]

//! PDM MEMS microphone capture on the nRF54LM20-DK, streamed to a host over RTT.
//!
//! Wiring (mic breakout -> DK):
//!   CLK -> P1.23      mic 3V3 -> VDD header
//!   DAT -> P1.24      mic GND -> GND header
//!
//! Pin choice matters: on the nRF54L family, peripheral clock signals must use a
//! "Clock pin". P1.23 is the dedicated clock pin Nordic uses for PDM_CLK on this
//! exact DK (PDM_DIN on P1.24); a non-clock pin (e.g. P1.02) does not output the
//! mic clock, so the mic never drives data.
//!
//! The PDM hardware decimates the mic's 1-bit stream to 16 kHz mono signed-16
//! PCM. We double-buffer it and push raw little-endian samples down RTT up-channel
//! 1 ("pcm"); status text goes to up-channel 0 ("log"). The host `capture` tool
//! reads channel 1 into a .wav.

use cortex_m_rt::entry;
use panic_halt as _;
use rtt_target::{rtt_init, ChannelMode};

mod pdm;
use pdm::{Pdm, Pin};

const CLK: Pin = Pin { port: 1, pin: 23 }; // P1.23: dedicated clock pin (PDM_CLK)
const DIN: Pin = Pin { port: 1, pin: 24 }; // P1.24: PDM_DIN

// 256 samples = 16 ms per buffer at 16 kHz; 512 bytes shipped per STARTED event.
const SAMPLES: usize = 256;

// EasyDMA needs word-aligned RAM buffers.
#[repr(C, align(4))]
struct Buf([i16; SAMPLES]);

static mut BUF0: Buf = Buf([0; SAMPLES]);
static mut BUF1: Buf = Buf([0; SAMPLES]);

#[entry]
fn main() -> ! {
    // up.0 "log": small, lossy, human-readable status.
    // up.1 "pcm": large, lossless binary audio. BlockIfFull keeps the stream
    // gapless as long as the host drains it (SWD RTT easily exceeds 32 KB/s).
    let channels = rtt_init! {
        up: {
            0: { size: 256, mode: ChannelMode::NoBlockSkip, name: "log" }
            1: { size: 8192, mode: ChannelMode::BlockIfFull, name: "pcm" }
        }
    };
    let mut log = channels.up.0;
    let mut pcm = channels.up.1;

    let _ = log.write(b"pdm-mic: starting 16kHz mono capture on P1.02/P1.03\n");

    // SAFETY: single-threaded, no other code touches the PDM block or these
    // buffers. The two static buffers outlive the stream (program never returns).
    let mut stream = unsafe {
        let pdm = Pdm::init(CLK, DIN);
        let b0 = &mut *core::ptr::addr_of_mut!(BUF0);
        let b1 = &mut *core::ptr::addr_of_mut!(BUF1);
        pdm.start(&mut b0.0, &mut b1.0)
    };

    // DIAGNOSTIC: with PDM running and driving CLK, watch the live DAT line. If it
    // toggles, the mic is alive and clocked (so any silence is decimation config);
    // if it is stuck, the chain CLK->mic->DAT is broken (power/wiring/clock).
    let (hi, lo) = pdm::probe_pin_activity(DIN, 200_000);
    let _ = log.write(if hi > 0 && lo > 0 {
        b"pdm-mic: DAT line TOGGLING - mic is alive and clocked\n" as &[u8]
    } else if hi > 0 {
        b"pdm-mic: DAT line stuck HIGH - no mic data (check power/wiring/clock)\n"
    } else {
        b"pdm-mic: DAT line stuck LOW - no mic data (check power/wiring/clock)\n"
    });

    let _ = log.write(b"pdm-mic: streaming\n");

    loop {
        let bytes = stream.next_buffer();
        pcm.write(bytes);
    }
}
