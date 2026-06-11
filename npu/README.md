# nRF54LM20B Axon NPU — bare-metal Rust

Access the Axon NPU from bare-metal Rust (no Zephyr) by FFI into Nordic's
pre-compiled driver. The NPU is a command-buffer executor: you compile an
int8-quantized TFLite model offline, link the generated command buffer, and call
the inference API.

## Why this shape

The Axon programming model (command-buffer submission register + instruction
encoding) is not public — only `ENABLE`/`STATUS` are documented. Nordic ships the
logic as a pre-compiled blob plus open C wrappers. The blob is RTOS-agnostic
(verified: its only external deps are `memcpy`/`memset` and the `nrf_axon_platform_*`
shims), so Zephyr is not required — we implement the platform layer in Rust.
See `NOTES.md` for the full analysis.

## Layout

    vendor/lib/   libnrf-axon-driver-internal-fpu.a   pre-compiled driver blob (Nordic IP)
    vendor/src/   nrf_axon_nn_infer.c, ..._op_extensions.c   open high-level API
    vendor/include/   Axon headers
    csrc/glue.c   variadic nrf_axon_platform_printf (can't be defined in stable Rust)
    src/platform.rs   the ~10 platform-interface shims + init()
    src/libm_shims.rs exp/expf/round/roundf from the libm crate (no newlib)
    src/main.rs   entry, Axon buffers, inference call site
    build.rs      cc-compiles the C, links the blob, runs bindgen

`vendor/` is git-ignored (Nordic LicenseRef-Nordic-5-Clause). Re-populate from the
Edge AI add-on (github.com/nrfconnect/sdk-edge-ai, branch matching NCS v3.3.0):

    cp <sdk-edge-ai>/lib/axon/bin/arm/libnrf-axon-driver-internal-fpu.a vendor/lib/
    cp -r <sdk-edge-ai>/include/axon vendor/include/axon
    cp -r <sdk-edge-ai>/include/drivers vendor/include/drivers
    cp <sdk-edge-ai>/drivers/axon/nrf_axon_nn_infer.c vendor/src/
    cp <sdk-edge-ai>/drivers/axon/nrf_axon_nn_op_extensions.c vendor/src/

## Toolchain

- Rust target `thumbv8m.main-none-eabihf` (`rustup target add thumbv8m.main-none-eabihf`)
- `arm-none-eabi-gcc` (compiles the vendored C)
- libclang for bindgen — path set via `LIBCLANG_PATH` in `.cargo/config.toml`
- `probe-rs` for flashing

## Build

    cargo build            # compiles + links the full FFI graph

The current `main()` powers on the NPU and pins the inference API as a link anchor.

## Before running on hardware — two values to confirm

1. **AXONS base address** — `src/platform.rs::AXON_BASE_ADDR` is a placeholder.
   Confirm from the nRF54LM20B product specification / MDK (`NRF_AXONS_BASE`); on
   Zephyr it is the devicetree `&axon` node `reg`. `ENABLE` is at `base + 0x400`.
2. **RRAM standby** — ensure RRAM stays powered during inference (the engine reads
   model constants from it). Zephyr votes via `nrf_sys_event_register`; on bare
   metal, don't power-gate RRAM, or replicate the POWER/MEMORY write in
   `platform::init`. See the TODO there.

## Adding a model

The Axon Compiler runs in a container (python3.11 + tensorflow) and turns an
int8-quantized `.tflite` into a header containing the weights + compiled command
buffer + a `model_<name>` descriptor. The engine defaults to **podman**
(`CONTAINER_ENGINE=docker` to override).

    tools/compile-model.sh path/to/model.tflite my_model [interlayer_bytes] [psum_bytes]

Note: in VS Code's snap-confined terminal, `XDG_DATA_HOME` is redirected into the
snap sandbox and podman reports a "database configuration mismatch". The script
auto-corrects this; if you run podman manually there, prefix with
`XDG_DATA_HOME=$HOME/.local/share`.

This drops `nrf_axon_model_my_model_.h` into `vendor/include/generated/`.
`build.rs` then auto-detects it, generates a glue TU exposing
`axon_active_model()`, compiles + links it, and enables the `has_model` cfg, so
`src/main.rs::run_inference` runs against it. No manual wiring.

If the model's `*_buffer_needed` exceeds the current 2048 B, bump
`INTERLAYER_BUFFER_SIZE`/`PSUM_BUFFER_SIZE` in `build.rs` and the matching
constants in `src/main.rs`.

A worked example is bundled: Nordic's `hello_axon` (a sin(x) regression) is
vendored under `vendor/include/generated/`, so a default build already links and
runs a real inference. `run_inference` is shaped for a 1->1 scalar model; adapt
the input quantization / output dequantization to your tensor dimensions (read
them from the `model_<name>` descriptor at runtime, as the function shows).

## Flash

    probe-rs chip list | grep -i 54l     # find the exact chip id
    # set it in .cargo/config.toml runner, then:
    cargo run
