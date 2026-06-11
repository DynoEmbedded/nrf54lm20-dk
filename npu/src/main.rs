#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;

mod bindings;
mod libm_shims;
mod platform;

// Storage backing the C `extern uint32_t nrf_axon_interlayer_buffer[]` and
// `nrf_axon_psum_buffer[]` declared in nrf_axon_platform.h. Sizes (in bytes)
// must match the -D defines in build.rs and be >= your largest model's needs.
const INTERLAYER_BUFFER_BYTES: usize = 2048;
const PSUM_BUFFER_BYTES: usize = 2048;

#[no_mangle]
pub static mut nrf_axon_interlayer_buffer: [u32; INTERLAYER_BUFFER_BYTES / 4] =
    [0; INTERLAYER_BUFFER_BYTES / 4];

#[no_mangle]
pub static mut nrf_axon_psum_buffer: [u32; PSUM_BUFFER_BYTES / 4] = [0; PSUM_BUFFER_BYTES / 4];

// Accessor emitted by build.rs's model glue when a compiled model is present in
// vendor/include/generated/. Returns a pointer to the `model_<name>` descriptor.
#[cfg(has_model)]
extern "C" {
    fn axon_active_model() -> *const bindings::nrf_axon_nn_compiled_model_s;
}

#[entry]
fn main() -> ! {
    // Power on the NPU and initialize Nordic's driver.
    let rc = platform::init();
    let _ = rc; // 0 == success; wire to RTT/UART logging to observe.

    #[cfg(has_model)]
    {
        let _result = run_inference(1.5708); // pi/2; for hello_axon, sin -> ~1.0
        let _ = _result;
    }

    #[cfg(not(has_model))]
    {
        // No model yet: pin the inference API so --gc-sections keeps it and the
        // full FFI graph (API -> driver blob -> shims -> libm) is link-checked.
        core::hint::black_box(bindings::nrf_axon_nn_model_validate as *const ());
        core::hint::black_box(bindings::nrf_axon_nn_model_infer_sync as *const ());
        core::hint::black_box(bindings::nrf_axon_nn_model_infer_async as *const ());
        core::hint::black_box(bindings::nrf_axon_nn_model_async_init as *const ());
    }

    loop {
        cortex_m::asm::wfi();
    }
}

/// Run one synchronous inference on the linked model with a single scalar input,
/// returning the dequantized scalar output. Shaped for hello_axon (1->1); adapt
/// the input/output handling to your model's tensor dimensions.
#[cfg(has_model)]
fn run_inference(sample: f32) -> f32 {
    unsafe {
        let model = axon_active_model();
        let _ = bindings::nrf_axon_nn_model_validate(model); // 0 == ok

        // Quantize the input using the model's input parameters:
        //   q = sample * (quant_mult / 2^quant_round) + quant_zp
        let ext = (*model).external_input_ndx as usize;
        let inp = &(*model).inputs[ext];
        let in_scale = inp.quant_mult as f32 / (1u32 << inp.quant_round) as f32;
        let q = (sample * in_scale) as i32 + inp.quant_zp as i32;
        let input: [i8; 1] = [q.clamp(-128, 127) as i8];

        let mut output: [i8; 1] = [0];
        let _ = bindings::nrf_axon_nn_model_infer_sync(model, input.as_ptr(), output.as_mut_ptr());

        // Dequantize: y = (out - dequant_zp) * (dequant_mult / 2^dequant_round)
        let out_scale =
            (*model).output_dequant_mult as f32 / (1u32 << (*model).output_dequant_round) as f32;
        (output[0] as i32 - (*model).output_dequant_zp as i32) as f32 * out_scale
    }
}
