//! Flash the PDM-MIC firmware over the DK's debug probe, then record the RTT PCM
//! stream into a .wav. One command owns the single probe end to end: flash, reset,
//! attach RTT, and capture until Ctrl-C.
//!
//!   cargo run --release -- <firmware.elf> [out.wav]
//!
//! The firmware exposes RTT up-channel 1 ("pcm") as raw little-endian i16 mono
//! samples at 16 kHz; channel 0 ("log") carries status text.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use object::{Object, ObjectSymbol};
use probe_rs::config::Registry;
use probe_rs::probe::list::Lister;
use probe_rs::rtt::{Rtt, ScanRegion};
use probe_rs::{flashing, Core, Permissions};

const CHIP: &str = "nRF54LM20B";
// probe-rs has no built-in nRF54LM20B target; register our cloned definition.
const CHIP_DESCRIPTION: &str = include_str!("../../targets/nRF54LM20B.yaml");
const SAMPLE_RATE: u32 = 16_000;
const PCM_CHANNEL: usize = 1;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let elf = args
        .next()
        .ok_or_else(|| anyhow!("usage: capture <firmware.elf> [out.wav]"))?;
    let wav = args.next().unwrap_or_else(|| "capture.wav".to_string());

    // Open the (single) debug probe and attach to the target.
    let lister = Lister::new();
    let probes = lister.list_all();
    let info = probes
        .first()
        .ok_or_else(|| anyhow!("no debug probe found (is the DK plugged in?)"))?;
    let probe = lister.open(info).context("opening probe")?;

    let mut registry = Registry::from_builtin_families();
    registry
        .add_target_family_from_yaml(CHIP_DESCRIPTION)
        .context("registering nRF54LM20B target")?;
    let mut session = probe
        .attach_with_registry(CHIP, Permissions::default(), &registry)
        .context("attaching to target")?;

    // Locate the RTT control block by symbol (like the probe-rs CLI does) so we
    // attach at an exact address instead of scanning all of RAM.
    let rtt_region = match rtt_symbol_address(&elf)? {
        Some(addr) => {
            eprintln!("RTT control block at {addr:#010x}");
            ScanRegion::Exact(addr)
        }
        None => {
            eprintln!("_SEGGER_RTT symbol not found; falling back to RAM scan");
            ScanRegion::Ram
        }
    };

    eprintln!("flashing {elf} ...");
    flashing::download_file(&mut session, &elf, flashing::FormatKind::Elf).context("flashing")?;

    // Reset into the freshly flashed image. Flashing leaves the core halted with
    // reset-vector catch armed, so a bare reset() can re-halt before main runs (and
    // RTT never initializes). reset_and_halt to a known state, then run.
    {
        let mut core = session.core(0)?;
        core.reset_and_halt(Duration::from_millis(500))
            .context("reset_and_halt")?;
        core.run().context("run")?;
    }

    // Attach RTT (retry: the firmware needs a moment to publish its control block).
    let mut core = session.core(0)?;
    let mut rtt = attach_rtt(&mut core, &rtt_region)?;

    // Attaching can leave the core halted; make sure it is running so the firmware
    // actually produces samples.
    if core.core_halted().unwrap_or(false) {
        core.run().context("resuming core after RTT attach")?;
    }

    // Inventory the channels and find "pcm"/"log" by name (don't trust the index).
    let names: Vec<Option<String>> = rtt
        .up_channels()
        .iter()
        .map(|c| c.name().map(|s| s.to_string()))
        .collect();
    eprintln!("RTT up-channels: {names:?}");
    let pcm_idx = names
        .iter()
        .position(|n| n.as_deref() == Some("pcm"))
        .unwrap_or(PCM_CHANNEL);
    let log_idx = names.iter().position(|n| n.as_deref() == Some("log"));
    eprintln!("recording channel {pcm_idx} to {wav} (press Ctrl-C to stop)");

    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst))
            .context("installing Ctrl-C handler")?;
    }

    let spec = WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(&wav, spec).context("creating wav")?;

    let mut buf = [0u8; 4096];
    let mut logbuf = [0u8; 256];
    let mut carry: Option<u8> = None; // odd-byte boundary between reads
    let mut total: u64 = 0;
    let mut last = Instant::now();

    while !stop.load(Ordering::SeqCst) {
        // Echo any firmware status text (tells us the firmware reached "streaming").
        if let Some(li) = log_idx {
            if let Some(ch) = rtt.up_channel(li) {
                let n = ch.read(&mut core, &mut logbuf).unwrap_or(0);
                if n > 0 {
                    eprint!("[fw] {}", String::from_utf8_lossy(&logbuf[..n]));
                }
            }
        }

        let channel = rtt
            .up_channel(pcm_idx)
            .ok_or_else(|| anyhow!("firmware has no RTT pcm channel"))?;
        let n = channel.read(&mut core, &mut buf).context("reading RTT")?;
        if n == 0 {
            std::thread::sleep(Duration::from_millis(2));
            continue;
        }

        let mut bytes = &buf[..n];
        // Stitch a leftover low byte from the previous read to the first high byte.
        if let Some(lo) = carry.take() {
            writer.write_sample(i16::from_le_bytes([lo, bytes[0]]))?;
            total += 1;
            bytes = &bytes[1..];
        }
        let mut it = bytes.chunks_exact(2);
        for s in it.by_ref() {
            writer.write_sample(i16::from_le_bytes([s[0], s[1]]))?;
        }
        total += (bytes.len() / 2) as u64;
        if let [lo] = it.remainder() {
            carry = Some(*lo);
        }

        if last.elapsed() >= Duration::from_secs(1) {
            eprintln!(
                "  {} samples ({:.1}s)",
                total,
                total as f64 / SAMPLE_RATE as f64
            );
            last = Instant::now();
        }
    }

    writer.finalize().context("finalizing wav")?;
    eprintln!(
        "done: {} samples, {:.2}s -> {}",
        total,
        total as f64 / SAMPLE_RATE as f64,
        wav
    );
    Ok(())
}

fn attach_rtt(core: &mut Core, region: &ScanRegion) -> Result<Rtt> {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        match Rtt::attach_region(core, region) {
            Ok(rtt) => return Ok(rtt),
            Err(_) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(anyhow!("RTT attach failed: {e}")),
        }
    }
}

/// Read the address of the `_SEGGER_RTT` symbol (the RTT control block) from the
/// firmware ELF, if present.
fn rtt_symbol_address(elf_path: &str) -> Result<Option<u64>> {
    let data = std::fs::read(elf_path).with_context(|| format!("reading {elf_path}"))?;
    let file = object::File::parse(&*data).context("parsing ELF")?;
    Ok(file
        .symbols()
        .find(|s| s.name() == Ok("_SEGGER_RTT"))
        .map(|s| s.address()))
}
