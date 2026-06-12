#![no_std]
#![no_main]

//! Keyword spotting on the nRF54LM20-DK.
//!
//! PDM mic (P1.23 CLK / P1.24 DIN) -> 16 kHz mono PCM via EasyDMA, one 320
//! sample (20 ms) hop per buffer -> MFCC on the M33 (mfcc.rs, mirrors
//! training/features.py) -> sliding 49x10 feature window -> int8 DS-CNN on the
//! Axon NPU -> smoothed logits -> detections on RTT channel "log".
//!
//! At boot, if a test vector is linked (generated/testvec.rs), three
//! self-tests run before audio starts: NPU-vs-TFLite logits on a known input,
//! Rust-vs-Python feature extraction on a known clip, and the end-to-end
//! combination. This separates "model is wrong" from "frontend is wrong"
//! before any live audio is involved.

use core::fmt::Write;
use core::sync::atomic::{AtomicU32, Ordering};

use cortex_m_rt::{entry, exception};
use panic_halt as _;
use rtt_target::{rtt_init, ChannelMode, UpChannel};

mod bindings;
mod libm_shims;
mod mfcc;
mod pdm;
mod platform;

#[cfg(has_testvec)]
#[path = "../generated/testvec.rs"]
#[allow(dead_code)]
mod testvec;

use mfcc::{FRAME_HOP, FRAME_LEN, NUM_FRAMES, NUM_MFCC};
use pdm::{Pdm, Pin};

const CLK: Pin = Pin { port: 1, pin: 23 }; // P1.23: dedicated clock pin (PDM_CLK)
const DIN: Pin = Pin { port: 1, pin: 24 }; // P1.24: PDM_DIN

const NUM_CLASSES: usize = 12;
const FIRST_WORD_CLASS: usize = 2; // 0 = silence, 1 = unknown
static LABELS: [&str; NUM_CLASSES] = [
    "silence", "unknown", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
];

// Detection tuning. Logits are int8; margins are in averaged int8-logit units.
const INFER_EVERY_HOPS: u32 = 5; // one inference per 100 ms
const SMOOTH: usize = 4; // average the last 4 inferences (400 ms)
const DETECT_MARGIN: i32 = 8; // top1 - top2 gap required; tune on hardware
const COOLDOWN_HOPS: u32 = 50; // min 1 s between detections
const STATUS_EVERY_HOPS: u32 = 250; // 5 s heartbeat

const TAIL: usize = FRAME_LEN - FRAME_HOP; // 160 samples carried between hops
const FEAT_LEN: usize = NUM_FRAMES * NUM_MFCC;

// Storage backing the C `extern uint32_t nrf_axon_interlayer_buffer[]` and
// `nrf_axon_psum_buffer[]` declared in nrf_axon_platform.h. Sizes (in bytes)
// must match the -D defines in build.rs and be >= the model's reported needs.
const INTERLAYER_BUFFER_BYTES: usize = 16384;
const PSUM_BUFFER_BYTES: usize = 4096;

#[no_mangle]
pub static mut nrf_axon_interlayer_buffer: [u32; INTERLAYER_BUFFER_BYTES / 4] =
    [0; INTERLAYER_BUFFER_BYTES / 4];

#[no_mangle]
pub static mut nrf_axon_psum_buffer: [u32; PSUM_BUFFER_BYTES / 4] = [0; PSUM_BUFFER_BYTES / 4];

#[cfg(has_model)]
extern "C" {
    fn axon_active_model() -> *const bindings::nrf_axon_nn_compiled_model_s;
}

// --- Device interrupt vector table -------------------------------------------
// No PAC crate is used, so we supply the table cortex-m-rt expects from one
// (feature "device"). The driver blob enables the AXONS IRQ in the NVIC for
// EVENT-mode inference; without a real entry at position 86 the completion
// interrupt vectors into code bytes (INVSTATE UsageFault, found the hard way).

const AXONS_IRQN: usize = 86; // from the nRF54LM20B MDK

unsafe extern "C" fn default_irq_handler() {
    loop {
        cortex_m::asm::bkpt();
    }
}

unsafe extern "C" fn axons_irq_handler() {
    // Documented ISR contract: clears the interrupt at its source and, if
    // completion work is pending, cascades generate_driver_event ->
    // process_driver_event -> generate_user_event (see platform.rs).
    bindings::nrf_axon_handle_interrupt();
}

const fn vector_table() -> [unsafe extern "C" fn(); AXONS_IRQN + 1] {
    let mut t = [default_irq_handler as unsafe extern "C" fn(); AXONS_IRQN + 1];
    t[AXONS_IRQN] = axons_irq_handler;
    t
}

#[no_mangle]
#[link_section = ".vector_table.interrupts"]
pub static __INTERRUPTS: [unsafe extern "C" fn(); AXONS_IRQN + 1] = vector_table();

// --- Inference watchdog ------------------------------------------------------
// A session killed mid-inference (debugger reflash) can leave the Axon engine
// wedged in a way that survives soft reset; the next boot then hangs forever
// inside the first blocking infer_sync. SysTick counts while an inference is
// in flight; if one exceeds the budget, reset the chip -- the boot-time power
// cycle in platform::init clears the engine.

const WDOG_DISARMED: u32 = u32::MAX;
const WDOG_LIMIT_TICKS: u32 = 50; // 50 x 10 ms = 500 ms per inference
static WDOG_TICKS: AtomicU32 = AtomicU32::new(WDOG_DISARMED);

#[exception]
fn SysTick() {
    let t = WDOG_TICKS.load(Ordering::Relaxed);
    if t != WDOG_DISARMED {
        if t >= WDOG_LIMIT_TICKS {
            cortex_m::peripheral::SCB::sys_reset();
        }
        WDOG_TICKS.store(t + 1, Ordering::Relaxed);
    }
}

struct WdogGuard;

impl WdogGuard {
    fn arm() -> Self {
        WDOG_TICKS.store(0, Ordering::Relaxed);
        WdogGuard
    }
}

impl Drop for WdogGuard {
    fn drop(&mut self) {
        WDOG_TICKS.store(WDOG_DISARMED, Ordering::Relaxed);
    }
}

// EasyDMA needs word-aligned RAM buffers. One buffer = one MFCC hop.
#[repr(C, align(4))]
struct Buf([i16; FRAME_HOP]);

static mut BUF0: Buf = Buf([0; FRAME_HOP]);
static mut BUF1: Buf = Buf([0; FRAME_HOP]);
static mut TABLES: mfcc::Tables = mfcc::Tables::ZEROED;

#[entry]
fn main() -> ! {
    let channels = rtt_init! {
        up: {
            // NoBlockSkip: with no RTT host attached, BlockIfFull would freeze
            // the firmware once the buffer fills. 2048 B holds the whole boot
            // sequence, so an attaching host still sees the selftest results.
            0: { size: 2048, mode: ChannelMode::NoBlockSkip, name: "log" }
        }
    };
    let mut log = channels.up.0;
    writeln!(log, "kws: boot").ok();

    // Cycle counter for inference timing; SysTick for the inference watchdog.
    if let Some(mut cp) = cortex_m::Peripherals::take() {
        cp.DCB.enable_trace();
        cp.DWT.enable_cycle_counter();
        cp.SYST
            .set_clock_source(cortex_m::peripheral::syst::SystClkSource::Core);
        cp.SYST.set_reload(1_280_000 - 1); // 10 ms at 128 MHz
        cp.SYST.clear_current();
        cp.SYST.enable_interrupt();
        cp.SYST.enable_counter();
    }

    let rc = platform::init();
    writeln!(log, "kws: axon init rc={}", rc).ok();

    let tables = unsafe {
        let t = &mut *core::ptr::addr_of_mut!(TABLES);
        t.init();
        &*t
    };

    #[cfg(has_model)]
    run(log, tables);

    #[cfg(not(has_model))]
    {
        let _ = tables;
        writeln!(log, "kws: no model linked; idle").ok();
        // Pin the inference API so the full FFI graph stays link-checked.
        core::hint::black_box(bindings::nrf_axon_nn_model_validate as *const ());
        core::hint::black_box(bindings::nrf_axon_nn_model_infer_sync as *const ());
        loop {
            cortex_m::asm::wfi();
        }
    }
}

#[cfg(has_model)]
fn quantize(x: f32, mult: f32, zp: i32) -> i8 {
    (libm::roundf(x * mult) as i32 + zp).clamp(-128, 127) as i8
}

#[cfg(has_model)]
fn infer(
    model: *const bindings::nrf_axon_nn_compiled_model_s,
    input: &[i8; FEAT_LEN],
) -> [i8; NUM_CLASSES] {
    let mut out = [0i8; NUM_CLASSES];
    let _wdog = WdogGuard::arm();
    unsafe {
        bindings::nrf_axon_nn_model_infer_sync(model, input.as_ptr(), out.as_mut_ptr());
    }
    out
}

fn argmax(v: &[i32; NUM_CLASSES]) -> (usize, usize) {
    let (mut top1, mut top2) = (0usize, 1usize);
    if v[1] > v[0] {
        (top1, top2) = (1, 0);
    }
    for c in 2..NUM_CLASSES {
        if v[c] > v[top1] {
            top2 = top1;
            top1 = c;
        } else if v[c] > v[top2] {
            top2 = c;
        }
    }
    (top1, top2)
}

#[cfg(all(has_model, has_testvec))]
fn selftest(
    log: &mut UpChannel,
    model: *const bindings::nrf_axon_nn_compiled_model_s,
    tables: &mfcc::Tables,
    in_mult: f32,
    in_zp: i32,
) {
    // 1. NPU vs int8 TFLite interpreter on the exact quantized input.
    let nn = infer(model, &testvec::TV_INPUT_Q);
    let mut maxdiff = 0i32;
    for c in 0..NUM_CLASSES {
        maxdiff = maxdiff.max((nn[c] as i32 - testvec::TV_LOGITS[c] as i32).abs());
    }
    let mut as_i32 = [0i32; NUM_CLASSES];
    for c in 0..NUM_CLASSES {
        as_i32[c] = nn[c] as i32;
    }
    let (top, _) = argmax(&as_i32);
    writeln!(
        log,
        "selftest nn: argmax={} want={} logit-maxdiff={}",
        LABELS[top],
        LABELS[testvec::TV_LABEL],
        maxdiff
    )
    .ok();

    // 2. Rust MFCC vs Python features on the embedded PCM clip.
    writeln!(log, "selftest mfcc: computing").ok();
    let mut feats = [0.0f32; FEAT_LEN];
    let mut maxerr = 0.0f32;
    for f in 0..NUM_FRAMES {
        let start = f * FRAME_HOP;
        let frame: &[i16; FRAME_LEN] =
            testvec::TV_PCM[start..start + FRAME_LEN].try_into().unwrap();
        let row = tables.process(frame);
        for j in 0..NUM_MFCC {
            feats[f * NUM_MFCC + j] = row[j];
            let err = (row[j] - testvec::TV_FEATS[f * NUM_MFCC + j]).abs();
            if err > maxerr {
                maxerr = err;
            }
        }
    }
    // Report in millis to avoid float formatting.
    writeln!(log, "selftest mfcc: maxerr={}e-3", (maxerr * 1000.0) as i32).ok();

    // 3. End to end: our features, quantized with the model's own params.
    let mut q = [0i8; FEAT_LEN];
    for i in 0..FEAT_LEN {
        q[i] = quantize(feats[i], in_mult, in_zp);
    }
    let nn = infer(model, &q);
    for c in 0..NUM_CLASSES {
        as_i32[c] = nn[c] as i32;
    }
    let (top, _) = argmax(&as_i32);
    writeln!(
        log,
        "selftest e2e: argmax={} want={}",
        LABELS[top],
        LABELS[testvec::TV_LABEL]
    )
    .ok();
}

#[cfg(has_model)]
fn run(mut log: UpChannel, tables: &'static mfcc::Tables) -> ! {
    let model = unsafe { axon_active_model() };
    let rc = unsafe { bindings::nrf_axon_nn_model_validate(model) };
    writeln!(log, "kws: model validate rc={}", rc.0).ok();

    // Input quantization parameters from the model descriptor:
    //   q = round(x * quant_mult / 2^quant_round) + quant_zp
    let (in_mult, in_zp) = unsafe {
        let ext = (*model).external_input_ndx as usize;
        let inp = &(*model).inputs[ext];
        (
            inp.quant_mult as f32 / (1u32 << inp.quant_round) as f32,
            inp.quant_zp as i32,
        )
    };

    #[cfg(has_testvec)]
    selftest(&mut log, model, tables, in_mult, in_zp);

    writeln!(log, "kws: starting pdm").ok();
    // SAFETY: single-threaded; the static buffers outlive the stream.
    let mut stream = unsafe {
        let p = Pdm::init(CLK, DIN);
        let b0 = &mut *core::ptr::addr_of_mut!(BUF0);
        let b1 = &mut *core::ptr::addr_of_mut!(BUF1);
        p.start(&mut b0.0, &mut b1.0)
    };

    // With PDM driving CLK, a live mic toggles DIN.
    let (hi, lo) = pdm::probe_pin_activity(DIN, 200_000);
    writeln!(
        log,
        "kws: mic {}",
        if hi > 0 && lo > 0 { "alive (DIN toggling)" } else { "DEAD (DIN stuck - check wiring)" }
    )
    .ok();

    let mut tail = [0i16; TAIL];
    let mut frame = [0i16; FRAME_LEN];
    let mut feats = [[0.0f32; NUM_MFCC]; NUM_FRAMES]; // ring, oldest at `head`
    let mut head = 0usize;
    let mut filled = 0usize;
    let mut hist = [[0i8; NUM_CLASSES]; SMOOTH];
    let mut hist_n = 0usize;
    let mut hist_i = 0usize;
    let mut hops: u32 = 0;
    let mut last_detect: u32 = 0;
    let mut infer_cycles: u32 = 0;

    writeln!(log, "kws: listening").ok();

    loop {
        let samples = stream.next_buffer(); // blocks until a 20 ms hop is full

        frame[..TAIL].copy_from_slice(&tail);
        frame[TAIL..].copy_from_slice(samples);
        tail.copy_from_slice(&samples[FRAME_HOP - TAIL..]);

        feats[head] = tables.process(&frame);
        head = (head + 1) % NUM_FRAMES;
        if filled < NUM_FRAMES {
            filled += 1;
        }
        hops = hops.wrapping_add(1);

        if filled == NUM_FRAMES && hops % INFER_EVERY_HOPS == 0 {
            // Assemble the window in chronological order (oldest = head).
            let mut q = [0i8; FEAT_LEN];
            for f in 0..NUM_FRAMES {
                let row = &feats[(head + f) % NUM_FRAMES];
                for j in 0..NUM_MFCC {
                    q[f * NUM_MFCC + j] = quantize(row[j], in_mult, in_zp);
                }
            }

            let t0 = cortex_m::peripheral::DWT::cycle_count();
            let logits = infer(model, &q);
            infer_cycles = cortex_m::peripheral::DWT::cycle_count().wrapping_sub(t0);

            hist[hist_i] = logits;
            hist_i = (hist_i + 1) % SMOOTH;
            if hist_n < SMOOTH {
                hist_n += 1;
            }

            let mut sums = [0i32; NUM_CLASSES];
            for h in 0..hist_n {
                for c in 0..NUM_CLASSES {
                    sums[c] += hist[h][c] as i32;
                }
            }
            let (top1, top2) = argmax(&sums);
            let margin = (sums[top1] - sums[top2]) / hist_n as i32;

            if top1 >= FIRST_WORD_CLASS
                && margin >= DETECT_MARGIN
                && hops.wrapping_sub(last_detect) >= COOLDOWN_HOPS
            {
                writeln!(log, "DETECT {} margin={}", LABELS[top1], margin).ok();
                last_detect = hops;
            }
        }

        if hops % STATUS_EVERY_HOPS == 0 {
            writeln!(
                log,
                "kws: t={}s overruns={} infer={}us",
                hops / 50,
                stream.overruns,
                infer_cycles / 128
            )
            .ok();
        }
    }
}
