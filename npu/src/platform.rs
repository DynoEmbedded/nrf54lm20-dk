//! Bare-metal implementation of the Axon platform interface
//! (`nrf_axon_platform.h`). The pre-compiled driver calls these; we provide the
//! RTOS-equivalent primitives without Zephyr.
//!
//! Configured for a single-threaded application using synchronous inference.
//! Reservation is trivial (one owner), the user-event is a simple flag, and a
//! driver-event is processed inline (the header explicitly permits calling
//! `nrf_axon_process_driver_event()` directly on bare metal).

use core::ffi::c_void;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::bindings;

/// Base address of the AXONS peripheral (secure alias).
///
/// From the nRF54LM20B MDK: `NRF_AXONS_S_BASE = 0x50056000`
/// (non-secure alias `NRF_AXONS_NS_BASE = 0x40056000`). The CPU boots secure
/// without TrustZone/SPU setup, so use the secure alias.
pub const AXON_BASE_ADDR: usize = 0x5005_6000;

/// AXONS interrupt line (`AXONS_IRQn`). Needed only for async/event-driven
/// inference; the synchronous-polling path used here does not require the IRQ.
#[allow(dead_code)]
pub const AXONS_IRQN: u16 = 86;

/// ENABLE register offset within the AXONS block (MDK: ENABLE @ 0x400, EN = bit 0).
const AXON_ENABLE_OFFSET: usize = 0x400;
const AXON_ENABLE_EN_BIT: u32 = 1; // AXONS_ENABLE_EN_Msk

static USER_EVENT: AtomicBool = AtomicBool::new(false);

/// One-time bring-up: power/clock the NPU, then initialize and power on the
/// driver. Returns the driver result code (0 == success).
///
/// RRAM note: the Zephyr platform votes to keep RRAM in standby
/// (`nrf_sys_event_register`) so the engine can read model constants during
/// inference. After reset RRAM is active (code runs from it), so no action is
/// needed for a simple app. Only if you enter low-power modes that power-gate
/// RRAM must you hold it in standby across inference.
pub fn init() -> i32 {
    unsafe {
        let enable = (AXON_BASE_ADDR + AXON_ENABLE_OFFSET) as *mut u32;
        core::ptr::write_volatile(enable, core::ptr::read_volatile(enable) | AXON_ENABLE_EN_BIT);

        let r = bindings::nrf_axon_driver_init(AXON_BASE_ADDR as *mut c_void);
        if r.0 != 0 {
            return r.0;
        }
        bindings::nrf_axon_driver_power_on().0
    }
}

// --- Interrupt masking: lightweight critical sections for the driver. ---

#[no_mangle]
pub extern "C" fn nrf_axon_platform_disable_interrupts() -> u32 {
    let was_enabled = cortex_m::register::primask::read().is_active();
    cortex_m::interrupt::disable();
    was_enabled as u32
}

#[no_mangle]
pub extern "C" fn nrf_axon_platform_restore_interrupts(restore_value: u32) {
    if restore_value != 0 {
        // SAFETY: only re-enables if interrupts were enabled before the matching
        // disable call, preserving nesting.
        unsafe { cortex_m::interrupt::enable() };
    }
}

// --- Hardware reservation: single owner, so always granted. ---

#[no_mangle]
pub extern "C" fn nrf_axon_platform_reserve_for_user() -> bool {
    true
}

#[no_mangle]
pub extern "C" fn nrf_axon_platform_free_reservation_from_user() {}

#[no_mangle]
pub extern "C" fn nrf_axon_platform_reserve_for_driver() -> bool {
    true
}

#[no_mangle]
pub extern "C" fn nrf_axon_platform_free_reservation_from_driver() {}

// --- Event signalling. ---

#[no_mangle]
pub extern "C" fn nrf_axon_platform_generate_user_event() {
    USER_EVENT.store(true, Ordering::SeqCst);
}

#[no_mangle]
pub extern "C" fn nrf_axon_platform_wait_for_user_event() {
    while !USER_EVENT.swap(false, Ordering::SeqCst) {
        cortex_m::asm::nop();
    }
}

#[no_mangle]
pub extern "C" fn nrf_axon_platform_generate_driver_event() {
    // Bare-metal: process inline rather than signalling a driver thread.
    unsafe { bindings::nrf_axon_process_driver_event() };
}

// --- Timing (used only by the driver's profiling/test paths). ---

#[no_mangle]
pub extern "C" fn nrf_axon_platform_get_clk_hz() -> u32 {
    // CPU/Axon clock; used to convert profiling ticks. Adjust if you change clocks.
    128_000_000
}

#[no_mangle]
pub extern "C" fn nrf_axon_platform_get_ticks() -> u32 {
    cortex_m::peripheral::DWT::cycle_count()
}
